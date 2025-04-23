import json

def add_index_to_json(json_data):
    for idx, item in enumerate(json_data):  # 遍历列表，同时获取索引
        item['index'] = idx  # 添加 index 属性
    return json_data

def save_json_to_file(json_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON data saved to {file_path}")

def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 主程序
if __name__ == "__main__":
    # 文件路径
    file_path = 'test.json'
    
    # 加载 JSON 数据
    json_data = load_json_from_file(file_path)

    # 添加 index 属性
    updated_json_data = add_index_to_json(json_data)

    # 保存更新后的 JSON 数据
    save_json_to_file(updated_json_data, file_path)