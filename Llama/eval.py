import json
import os
import fire
import pdb
import glob
import os

def find_json_files(folder_path):
    # 使用递归模式匹配所有子目录的.json文件
    search_pattern = os.path.join(folder_path, "**", "*.json")
    return glob.glob(search_pattern, recursive=True)


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
    # print(f"数据已成功存储到 {file_path}")


def extract_response(text: str, data_path:str) -> str:
    start_marker = "### Response:"
    start_idx = text.find(start_marker)
    
    if start_idx == -1:
        return ""
    
    response_text = text[start_idx + len(start_marker):].strip()
    
    end_marker = "assistant"
    end_idx = response_text.find(end_marker)

    if 'template0' in data_path:
        if end_idx != -1:
            response_text = response_text[:end_idx].strip()

    else:
        if end_idx != -1:
            second_end_idx = response_text.find(end_marker, end_idx + len(end_marker))
            if second_end_idx != -1:
                response_text = response_text[:second_end_idx].strip()
    
    response_text.replace('assistant', '')
    response_text.strip()
    
    return response_text


def eval_one_data(
    model_answer: str,
    answer: str,
):
    int_answer = float(answer)

    if int_answer.is_integer():
        int_answer = str(int(int_answer))
        formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)
    
    if answer in model_answer or int_answer in model_answer or formatted_answer in model_answer:
        return True
    else:
        return False
    
    
def eval_data(
        data_path: str = "",
        train_config: dict=None,
        eval_num: int = None
):
    data = read_json_file(data_path)
    
    if "attn_o" in data_path:
        eval_type = "attn_o"
    elif "attn_q" in data_path:
        eval_type = "attn_q"
    elif "attn_k" in data_path:
        eval_type = "attn_k"
    elif "attn_v" in data_path:
        eval_type = "attn_v"
    elif "ffn" in data_path:
        eval_type = "ffn"
    elif "lora" in data_path:
        eval_type = "lora"
    elif "base" in data_path:
        eval_type = "base"
    elif "lofit" in data_path:
        eval_type = "lofit"
    
    correct_count = 0
    question_count = 0
    
    for index, key in enumerate(data.keys()):

        one_data = data[key]
        one_data["eval"] = False

        if eval_type not in one_data.keys() or int(index) >= int(eval_num):
            break

        answer = one_data["Answer"]
        model_answer = one_data[eval_type]
        question_count += 1

        # model_answer = extract_response(model_answer, data_path)
        
        int_answer = float(one_data["Answer"])

        if int_answer.is_integer():
            int_answer = str(int(int_answer))
            formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)
        
        if answer in model_answer or int_answer in model_answer or formatted_answer in model_answer:
            one_data["eval"] = True
            correct_count += 1
    
    if question_count==0:
        return
    
    op_position = train_config["op_position"]
    data_num = train_config["data_num"]
    template_index = train_config["template_index"]
    lr = train_config["lr"]
    acc= (correct_count/question_count)*100
    if template_index != "template0":
        print(f"|{op_position}|{data_num}|{template_index}|{lr}|{correct_count}|{question_count}|{acc:.1f}|")
    save_list_to_json(data, data_path)


import re
from typing import List, Dict

def parse_training_config(path: str) -> Dict[str, str]:
    path_parts = path.split("/")

    config_path = path_parts[2]
    template = path_parts[3]

    if 'base' in path.lower():
        train_config = {
            "data_num": "base",
            "op_position": "base",
            "template_index": template,
            "lr": "None"
        }
        return train_config
    
    elif 'lora' in path:
        parts = config_path.split("_")
        train_config = {
            "data_num": parts[0],
            "op_position": "lora",
            "template_index": template,
            "lr": parts[-1]
        }
    
    else:
        parts = config_path.split("_")
        train_config = {
            "data_num": parts[0],
            "op_position": "red",
            "template_index": template,
            "lr": parts[-1]
        }

        train_config["op_position"] = "base"

        if "bias" in config_path:
            train_config["op_position"] = "bias"
        elif "scaling" in config_path:
            train_config["op_position"] = "scaling"
        elif "attn_o" in config_path:
            train_config["op_position"] = "attn_o"
        elif "attn_q" in config_path:
            train_config["op_position"] = "attn_q"
        elif "attn_k" in config_path:
            train_config["op_position"] = "attn_k"
        elif "attn_v" in config_path:
            train_config["op_position"] = "attn_v"
        elif "ffn_up" in config_path:
            train_config["op_position"] = "ffn_up"
        elif "ffn_down" in config_path:
            train_config["op_position"] = "ffn_down"
        elif "ffn" in config_path:
            train_config["op_position"] = "ffn"
        elif "lora" in config_path:
            train_config["op_position"] = "lora"
        elif "base" in config_path:
            train_config["op_position"] = "base"
        elif "lofit" in config_path:
            train_config["op_position"] = "lofit"
    
    return train_config


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,default=None)
parser.add_argument('--eval_num', type=str,default=None) 
args = parser.parse_args()

if __name__ == "__main__":

    json_files = find_json_files(args.data_path)
    print(f"找到 {len(json_files)} 个JSON文件:")
    print('|op_position|data_num|template_index|lr|correct_count|question_count|accuracy|')
    print('|:--:|:--:|:--:|:--:|:--:|:--:|:--:|')
    eval_list = []
    for file in json_files:
        train_config = parse_training_config(file)
        eval_data(file, train_config, args.eval_num)
    
    # eval_list.append(file)
    # print(eval_list)
    # fire.Fire(train)