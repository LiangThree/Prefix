import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle
from functools import partial
import pdb
import json
from datasets import load_dataset
from baukit import Trace, TraceDict

import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MODEL_NAME="/mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct"  
if not os.path.exists(MODEL_NAME):
    MODEL_NAME="/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct"  
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS=32  # LLaMA-7B的层数
NUM_HEADS=32   # 每层的注意力头数
SEED=42
BATCH_SIZE=16
MAX_SEQ_LEN=64

def load_and_format_data():
    """加载TruthfulQA数据集并格式化为问答对"""

    # 下载数据集并存储到本地
    # dataset = load_dataset('truthfulqa/truthful_qa','generation') # 'generation' or 'multiple_choice'
    # dataset = dataset['validation']

    # with open("/mnt/userdata/liangsirui/MyProject/Prefix/dataset/TruthfulQA/truthful_qa.jsonl", "w") as f:
    #     for i in range(len(dataset)):
    #         row = dataset[i]
    #         f.write(json.dumps(row) + "\n")
    
    # 读取数据
    data_path = "/mnt/userdata/liangsirui/MyProject/Prefix/dataset/TruthfulQA/truthful_qa.jsonl"
    if not os.path.exists(data_path):
        data_path = "/mnt/userdata/MyProject/Prefix/dataset/TruthfulQA/truthful_qa.jsonl"
    
    dataset = load_dataset("json", data_files=data_path)["train"]

    # 转换为 Dataset 对象
    print(dataset[0])
    
    formatted_data = []
    for item in dataset:
        # 处理正确答案
        for ans in item['correct_answers']:
            formatted_data.append({
                'prompt': format_truthfulqa(item['question'], ans),
                'label': 1
            })
        
        # 处理错误答案
        for ans in item['incorrect_answers']:
            formatted_data.append({
                'prompt': format_truthfulqa(item['question'], ans),
                'label': 0
            })
    
    return formatted_data

def format_truthfulqa(question, answer):
    return f"Q: {question}\nA: {answer}"

class ActivationExtractor:
    def __init__(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)
        self.model.eval()
        
    def get_activations(self, prompts):
        """批量提取各层的注意力头激活"""
        all_layer_activations = []

        # filter prompts above max length
        # filter_prompt = [] 
        # for i in tqdm(range(0, len(prompts))):
        #     inputs = self.tokenizer(
        #         prompts[i], 
        #         return_tensors="pt", 
        #         truncation=True,
        #         max_length=MAX_SEQ_LEN, 
        #         padding="max_length"
        #     ).to(DEVICE)

        #     if len(inputs['input_ids']) > MAX_SEQ_LEN:
        #         continue
        #     else:
        #         filter_prompt.append(prompts[i])

        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            
            # 编码文本
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LEN, 
                padding="max_length"
            ).to(DEVICE)

            # 提取激活
            with torch.no_grad():
                with TraceDict(self.model, [f"model.layers.{i}.self_attn.o_proj" for i in range(NUM_LAYERS)]) as td:
                    outputs = self.model(**inputs)
                    
                    # 收集各层各头的激活
                    batch_activations = []
                    for layer in range(NUM_LAYERS):
                        layer_activations = td[f"model.layers.{layer}.self_attn.o_proj"].output
                        batch_activations.append(layer_activations.cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
                    
                    
                    batch_activations = np.stack(batch_activations) # 拼接: [layers, batch, seq_len, features]
                    batch_activations = np.transpose(batch_activations, (1, 0, 2, 3))  # [batch, layers, seq, features]
                    all_layer_activations.append(batch_activations) # [data_num, batch, layers, seq_len, features]
        
        # 合并批次: [data_num, layers, seq_len, features]
        return np.concatenate(all_layer_activations, axis=0)

class ProbeTrainer:
    def __init__(self, num_layers, num_heads):
        self.num_layers = num_layers
        self.num_heads = num_heads
        
    def train_probes(self, activations, labels, train_ratio=0.8):
        np.random.seed(SEED)
        
        # 数据分割
        indices = np.random.permutation(len(labels))
        split_idx = int(len(indices) * train_ratio)
        
        # 初始化存储
        probes = {}
        accuracies = np.zeros(self.num_layers)
        
        for layer in tqdm(range(self.num_layers)):
            # 提取特定头的激活特征
            head_features = activations[:, layer, -1, :]  # 取最后一个token的激活
            
            # 分割数据集
            X_train = head_features[indices[:split_idx]]
            y_train = labels[indices[:split_idx]]
            X_val = head_features[indices[split_idx:]]
            y_val = labels[indices[split_idx:]]
            
            # 训练逻辑回归
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X_train, y_train)
            
            # 评估并存储
            val_acc = clf.score(X_val, y_val)
            probes[(layer)] = clf
            accuracies[layer] = val_acc
                
        return probes, accuracies

if __name__ == "__main__":
    print("\nLoading data...")
    data = load_and_format_data()

    prompts = [d['prompt'] for d in data]
    labels = np.array([d['label'] for d in data])
    
    print("\nExtracting activations...")
    extractor = ActivationExtractor(MODEL_NAME)
    activations = extractor.get_activations(prompts)  # [data_num, layers, seq_len, features]
    
    print("Training probes...")
    trainer = ProbeTrainer(NUM_LAYERS, NUM_HEADS)
    probes, acc_matrix = trainer.train_probes(activations, labels)
    
    with open("truthfulqa_probes.pkl", "wb") as f:
        pickle.dump({
            'probes': probes,
            'accuracies': acc_matrix,
            'layer_head_mapping': (NUM_LAYERS)
        }, f)
    
    print(f"Probe layers results:")
    sorted_indices = np.argsort(acc_matrix)[::-1]  # acc_matrix现在是1D数组
    for idx in sorted_indices:
        print(f"Layer {idx}: Accuracy={acc_matrix[idx]:.4f}")