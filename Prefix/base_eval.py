import fire
import argparse
import torch, transformers
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

import json
import pdb
from datasets import load_dataset, Dataset
from typing import Optional, Union
from peft import PeftModel
from tqdm import *
from template import *
from make_answer_json import make_answer
from transformers import GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from vllm import LLM, SamplingParams

def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    data_list = list(data_dict.values())

    if data_num is not None:
        data_list = data_list[:min(data_num, len(data_list))]

    dataset = Dataset.from_list(data_list)

    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
        return dataset
    elif split == 'all' or split is None:
        return dataset
    else:
        raise ValueError("split参数应为float(0-1)或None")

def get_data(data_files:str, data_num:int, split:Optional[Union[str, float]] = None):

    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.select(range(data_num))
    
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")

def save_list_to_json(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)  # 使用 indent 格式化 JSON
    print(f"数据已成功存储到 {file_path}")

def train_model(model_path:str, data_num:int, template_index:str, output_path:str, dataset:str, vllm:bool):

    vllm = False

    print(f'Model: {model_path}')


    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    path = "/mnt/userdata/MyProject/" 
    if not os.path.exists(path):
        path = "/mnt/userdata/liangsirui/MyProject/"

    data_path = os.path.join(output_path, f'{template_index}_{dataset}_eval.json')

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(data_path):
        make_answer(dataset, data_path)

    if vllm:

        model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=4,  # 暂时禁用tensor并行
            gpu_memory_utilization=0.9,
            disable_custom_all_reduce=True  # 添加此参数
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=300,
            stop=[], 
            skip_special_tokens=True
        )

        tokenizer = model.get_tokenizer()

    else:
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, model_max_length=2048,
            padding_side="left", use_fast=False)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()

    template = get_prompt(tokenizer, template_index)
    
    def process_alpacaeval(example):
        question = example['Question']
        prompt = template.replace("%s", question)
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(data_path, data_num)
    dataset = dataset.map(process_alpacaeval)
    
    answer_dict = read_json_file(data_path)

    if vllm:
        prompts = []
        for i in range(len(dataset)):
            question = dataset[i]['Question']
            prompt = template % question
            prompts.append(prompt)

        outputs = model.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            answer_dict[str(i)]['base'] = generated_text
    
    else:
        batch_size = 32
        all_prompts = [ex["prompt"] for ex in dataset]
        total_samples = len(all_prompts)

        generation_config = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=1,
        )

        for batch_idx in tqdm(range(0, total_samples, batch_size)):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                generation_config=generation_config
            )
            
            completions = tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            
            for i, (prompt, completion) in enumerate(zip(batch_prompts, completions)):
                global_idx = batch_idx + i
                completion = completion.replace(prompt,"")
                answer_dict[str(global_idx)]["base"] = completion

    save_list_to_json(answer_dict, data_path)



if __name__ == '__main__':
    fire.Fire(train_model)