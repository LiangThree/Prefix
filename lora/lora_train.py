import argparse
import torch, transformers
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import json
import pdb
from datasets import load_dataset, Dataset
from typing import Optional, Union
from template import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
device = "cuda"

def get_data(data_files:str, data_num:int, split:Optional[Union[str, float]] = None):

    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.select(range(data_num))
    
    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

def train_model(model_path:str, data_path:str, layer:int, output_dir:str, data_num:int, template_index:int, learning_rate: float = 2e-5,):

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    path = "/mnt/userdata/MyProject/" 
    if not os.path.exists(path):
        path = "/mnt/userdata/liangsirui/MyProject/"
    data_path = path + data_path

    print("template_index:", template_index)

    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")
    prompt = prompt_template[template_index]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, trust_remote_code=True)

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=4, lora_alpha=32, target_modules=["o_proj"], layers_to_transform=[layer],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    train_data, eval_data = get_data(data_path, data_num, 0.8)

    # 将数据转换为 Hugging Face 的 Dataset 格式
    train_dataset = Dataset.from_dict({
        "input": [prompt % example['instruction'] for example in train_data],
        "output": [example['output'] for example in train_data]
    })

    valid_dataset = Dataset.from_dict({
        "input": [prompt % example['instruction'] for example in eval_data],
        "output": [example['output'] for example in eval_data]
    })

    def tokenize_function(example):
        return tokenizer(example["input"], example["output"], truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./lora_output",
        overwrite_output_dir=True,
        fp16=True, #converts to float precision 16 using bitsandbytes
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # 保存微调后的模型
    output_dir = os.path.join(output_dir, f"{data_num}_lora_{learning_rate}", f"template{template_index}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'model saved in {output_dir}')

def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    parser.add_argument('-model_path', '--model_path', type=str, default=None)
    parser.add_argument('-data_path', '--data_path', type=str, default=None)
    parser.add_argument('-output_dir', '--output_dir', type=str)
    parser.add_argument('-data_num', '--data_num', type=int)
    parser.add_argument('-layer', '--layer', type=int, default=None)
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=None)
    parser.add_argument('-template_index', '--template_index', type=int, default=None)
    args = parser.parse_args()

    train_model(**vars(args))

    # model.base_model.model.model.layers[15].self_attn.o_proj.lora_A.default.weight


if __name__ == '__main__':
    main()