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
from peft import PeftModel
from tqdm import *
from template import *
from make_answer_json import make_answer

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
device = "cuda"

def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    
    # 加载 JSON 文件（原始格式是字典）
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # 将字典转换为列表格式
    data_list = list(data_dict.values())

    # 限制数据条数，避免超界
    if data_num is not None:
        data_list = data_list[:min(data_num, len(data_list))]

    # 转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_list(data_list)

    # # 加载原始数据集
    # dataset = load_dataset('json', data_files=json_path, split='train')
    # dataset = dataset.select(range(data_num))
    
    # 处理数据集划分
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
    
    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

def read_json_file(file_path):
    try:
        # 以只读模式打开文件，使用 UTF-8 编码
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载 JSON 数据
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

def train_model(run_type:str, model_path:str, lora_path:str, data_num:int, template_index:int, dataset:str):

    print("template_index:", template_index)
    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")
    template = prompt_template[template_index]


    print(f'------------------- {run_type} ------------------- ')
    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    path = "/mnt/userdata/MyProject/" 
    if not os.path.exists(path):
        path = "/mnt/userdata/liangsirui/MyProject/"
    
    data_path = os.path.join(lora_path, f'template{template_index}', f'{dataset}_eval.json')
    lora_path = os.path.join(lora_path, f'template{template_index}')

    if not os.path.isfile(data_path):
        make_answer(dataset, data_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, trust_remote_code=True)

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()

    def process_alpacaeval(example):
        question = example['Question']
        prompt = template % question
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(data_path, data_num)
    dataset = dataset.map(process_alpacaeval)

    answer_dict = read_json_file(data_path)

    for i in tqdm(range(len(dataset))):
        
        generate_dict = {}
        prompt = dataset[i]["prompt"]
        prompt_ids = tokenizer(prompt, return_tensors='pt').to(model.base_model.device)
        
        outputs = model.generate(
            **prompt_ids,
            max_new_tokens=300,
            do_sample=False,  # 禁用采样
            num_beams=1,      # 贪婪搜索
            use_cache=True,    # 启用KV缓存
        )

        # pdb.set_trace()

        completion_good = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("------------------------ Question ----------------------------")
        print('Question:', dataset[i]['Question'])
        print('Output:', dataset[i]['Output'])
        print('Answer:', dataset[i]['Answer'])

        print("------------------------ Answer ----------------------------")
        print(completion_good)
        answer_dict[str(i)]['lora'] = completion_good

        print("------------------------ END ----------------------------\n\n")
    
    save_list_to_json(answer_dict, data_path)

def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    parser.add_argument('-run_type', '--run_type', type=str, default=None)
    parser.add_argument('-model_path', '--model_path', type=str, default=None)
    parser.add_argument('-lora_path', '--lora_path', type=str, default=None)
    parser.add_argument('-data_num', '--data_num', type=int)
    parser.add_argument('-template_index', '--template_index', type=int, default=None)
    parser.add_argument('-dataset', '--dataset', type=str)
    args = parser.parse_args()

    train_model(**vars(args))


if __name__ == '__main__':
    main()