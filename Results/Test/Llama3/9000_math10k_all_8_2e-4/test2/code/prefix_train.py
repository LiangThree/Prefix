import os
import sys
from datasets import load_dataset, Dataset
from typing import Optional, Union

import pdb
from template import *
import json
import torch.optim as optim

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

import logging
import fire
from datasets import load_dataset, concatenate_datasets
from transformers import  AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import torch
from transformers import AutoModelForCausalLM
from model import ActivationLLama
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers.trainer_callback import TrainerCallback
import torch

MAX_INPUT_LENGTH = 1024
MAX_LENGTH = 1024

device_map = "auto"

def load_RED_model(model_path, op_position, layer_type, n_prefix):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = ActivationLLama(model, op_position=op_position, layer_type=layer_type, prefix=-1)
    return model

class CustomModelSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_path = os.path.join(checkpoint_path, "delta_vector.pth")

        kwargs["model"].save_model(save_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path) :
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
        if  "model.safetensors" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "model.safetensors"))

class ActivationScalingMonitor(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.prev_scaling = None  # 记录上一次的 activation_scaling 值
        self.prev_bias = None


    def on_step_end(self, args, state, control, **kwargs):
        # 获取当前的 activation_scaling 值
        current_scaling = 0
        current_bias = 0

        for layer in range(32):
            if self.model.op_position == "attn_q":
                module = self.model.base_model.model.layers[layer].self_attn.q_proj
            elif self.model.op_position == "attn_k":
                module = self.model.base_model.model.layers[layer].self_attn.k_proj
            elif self.model.op_position == "attn_v":
                module = self.model.base_model.model.layers[layer].self_attn.v_proj
            elif self.model.op_position == "attn_o":
                module = self.model.base_model.model.layers[layer].self_attn.o_proj
            elif self.model.op_position == "ffn_up":
                module = self.model.base_model.model.layers[layer].mlp.up_proj
            elif self.model.op_position == "ffn_down":
                module = self.model.base_model.model.layers[layer].mlp.down_proj

            if hasattr(module.delta_vector, 'activation_scaling'):
                delta = module.delta_vector.activation_scaling.detach().to(torch.float).cpu().numpy()
                current_scaling += delta

            if hasattr(module.delta_vector, 'activation_bias'):
                current_bias += module.delta_vector.activation_bias.detach().to(torch.float).cpu().numpy()

        if hasattr(module.delta_vector, 'activation_scaling'):
            print(f"Step {state.global_step}: activation_scaling mean: {current_scaling.mean()}")

        if hasattr(module.delta_vector, 'activation_bias'):
            print(f"Step {state.global_step}: activation_bias mean: {current_bias.mean()}")


def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:

    file_path = json_path

    # 加载原始数据集
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.select(range(data_num))

    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
        return dataset
    elif split == 'all' or split is None:
        return dataset
    else:
        raise ValueError("split参数应为float(0-1)或None")
        
def save_params_to_json(output_path: str, **kwargs):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=4, ensure_ascii=False)

        
def train(
        model_path: str = "",
        data_path: str = "",
        output_dir: str = "",
        data_num: int = None,
        n_prefix: int = None,
        op_position: str = "",
        learning_rate: str = 2e-5,
        num_train_epochs:  int = 3,
        template_index: int = None,
        layer_type: str=""
):
    str_lr = str(learning_rate)
    learning_rate = float(learning_rate)

    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")

    path = "/share/kunluo/Models/"
    if not os.path.exists(path):
        path = "/mnt/usercache/huggingface/"
    model_path = path + model_path

    if not os.path.exists(model_path):
        model_path = model_path.replace('usercache', 'publiccache')

    if 'math10k' in data_path:
        data_type = 'math10k'
    elif 'prm800k' in data_path:
        data_type = 'prm800k'

    output_dir = os.path.join(output_dir, f'{data_num}_{data_type}_{layer_type}_{n_prefix}_{str_lr}')
    model = load_RED_model(model_path=model_path, op_position=op_position, layer_type=layer_type, n_prefix=n_prefix)
    train_config = {
        "model_path": model_path,
        "data_path": data_path,
        "data_num": data_num,
        "op_position": op_position,
        "template_index": template_index
    }
    output_dir = os.path.join(output_dir, f"epoch{num_train_epochs}")
    log_dir = os.path.join(output_dir, "log")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "train_config.json")
    print(train_config)
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    def process_ultra_preference(example, task_type, k=10):
        question = example["Question"]
        output = example["Output"]

        template = prompt_template[template_index] % question
        output = f"{output}\n\n "

        example["prompt"] = template
        example["output"] = output
        example["text"] = example["prompt"] + example["output"]

        inputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=True
        )
        input_ids = inputs["input_ids"]
        example["text_length"] =  len(inputs.input_ids)

        # 计算prompt长度
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False).input_ids
        prompt_length = len(prompt_ids)
        example["prompt_length"] = prompt_length

        # 初始化labels
        labels = [-100] * len(input_ids)

        # 设置output部分标签
        output_start = prompt_length
        output_end = len(input_ids) - 1  # 假设EOS已添加

        if task_type == "prefix":
            # 设置前k个token
            for i in range(output_start, output_end):
                if (i - output_start) < k:
                    labels[i] = input_ids[i]
                else:
                    labels[i] = -100
            labels[-1] = input_ids[-1]  # EOS
        else:
            # 全部output
            for i in range(output_start, len(input_ids)):
                labels[i] = input_ids[i]
        
        # 更新字段
        example.update({
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "task_type": task_type
        })
        return example

    train_data = load_custom_dataset(data_path, data_num)
    
    # 处理前缀数据集
    train_data = train_data.map(
        lambda ex: process_ultra_preference(ex, "prefix", k=n_prefix),
        num_proc=8
    )

    combined_data = train_data.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["text_length"] <= MAX_LENGTH)
    custom_saving_callback = CustomModelSavingCallback()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif os.path.isfile(output_dir+"/train.log"):
        with open(output_dir+"/train.log", 'w') as f:
            pass

    class LogSaverCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=True, **kwargs):
            if logs:
                with open(output_dir+"/train.log", "a") as f:
                    f.write(f"Step {state.global_step}: {logs}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        report_to="none",
        logging_strategy="steps",
        weight_decay=1e-2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=1024,
        train_dataset=combined_data,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[custom_saving_callback, ActivationScalingMonitor(model), LogSaverCallback()],
    )

    trainer.train()
    model.save_model(os.path.join(output_dir, "delta_vector.pth"))
    save_params_to_json(config_path, **train_config)


if __name__ == "__main__":
    fire.Fire(train)