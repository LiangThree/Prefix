import json
import os
import fire
import pdb
import glob
import os


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

def eval_one_data(
    model_answer: str,
    answer: str,
):
    if answer in ['A', 'B', 'C', 'D', 'E']:
        return True
    
    int_answer = float(answer)
    formatted_answer = answer

    if int_answer.is_integer():
        int_answer = int(int_answer)
        formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)
    
    if answer in model_answer or str(int_answer) in model_answer or formatted_answer in model_answer:
        return True
    else:
        return False

if __name__ == "__main__":
    data_path = "dataset/math10k/train.json"
    data = read_json_file(data_path)
    
    count = 0
    
    for one_data in data:
        if eval_one_data(one_data['output'], one_data['answer']):
            pass
        else:
            count += 1
            print(one_data)
    
    print(count) 

