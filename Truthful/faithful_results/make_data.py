import sys
import os
import pickle
import numpy as np
import json
import pdb
from tqdm import *
import random
import json, time, random
from openai import OpenAI
from tqdm import *

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


def save_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            # 将每个元素转为 JSON 字符串并写入文件（末尾加换行符）
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def is_num(num_data):
    for i in num_data:
        if i not in '0123456789':
            return False
    return True

def random_data(num):
    float_num = float(num)
    if float_num.is_integer():
        return random.randint(float_num//2, float_num*2)

def make_data():
    data_list = read_json_file('Truthful/faithful_results/math10k_faithful_data.json')
    
    prob_data = []
    for data in data_list:
        instruction = data['instruction']
        num_data = data['num_data']
        for num_sentence in num_data.keys():
            num = num_data[num_sentence]
            
            correct_answers = []
            incorrect_answers = []
            if type(num) == str and is_num(num) and '[NUM]' in num_sentence:
                correct_answers.append(num_sentence.replace('[NUM]', num))
                incorrect_answers.append(num_sentence.replace('[NUM]', str(random_data(num))))
            
            {
                "question": instruction,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers
            }
        
        if len(correct_answers) == 0 or len(incorrect_answers) == 0:
            continue

        prob_data.append({
                            "question": instruction,
                            "correct_answers": correct_answers,
                            "incorrect_answers": incorrect_answers
                        })
    
    save_list_to_jsonl(prob_data,'dataset/Faithful/faithful.jsonl')
                

if __name__ == "__main__":
    make_data()