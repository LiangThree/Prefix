import json

# 读取JSONL文件并处理数据
data = []
with open('4.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        processed = {
            "instruction": f"What is the sum of {entry['a']} and {entry['b']}?",
            "output": int(entry['golden']),
            "answer": int(entry['golden'])
        }
        data.append(processed)

# 保存为JSON文件
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("转换完成, test.json")