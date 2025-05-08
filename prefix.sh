#!/bin/bash
# pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U vllm
# layer_type: scaling/bias/all
# 当学习率达到0.01量级时scaling才会发生变化 

# Meta-Llama-3-8B-Instruct  template: 3
# Qwen2.5-Math-7B-Instruct  template: 4

condition=$1

if [ "$condition" == "prefix" ]; then

    CUDA_VISIBLE_DEVICES=0 python prefix_train.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --data_path  "dataset/math10k/clean_train.json" \
    --output_dir "Results/Test/Qwen2.5-instruct/" \
    --data_num 9000 \
    --n_prefix 10 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --template_index 4 \
    --layer_type "all" \
    --num_train_epochs 3

elif [ "$condition" == "eval" ]; then

    python Llama/eval.py \
    --data_path "Results/Test/Qwen2.5-instruct" \
    --eval_num 500

elif [ "$condition" == "answer" ]; then

    python Llama/answer.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --template_index 4 \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 10 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5-instruct/9000_math10k_all_10_2e-4/spilt9_1" \
    --repetition_penalty 1.1 \
    --data_num 500

elif [ "$condition" == "make_train" ]; then
    
    python Llama/make_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --output_path "Results/Base_Results/Llama3/" \
    --dataset "math10k" \
    --template_index 3 \
    --data_num 10000 \
    --vllm True

elif [ "$condition" == "lora_train" ]; then
    
    CUDA_VISIBLE_DEVICES=1 python lora/lora_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/clean_train.json" \
    --output_dir "Results/Lora_Results/Llama3/" \
    --learning_rate 1e-3 \
    --template_index 3 \
    --data_num 9000 \
    --layer 15

elif [ "$condition" == "run" ]; then

    CUDA_VISIBLE_DEVICES=1 python prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/clean_train.json" \
    --output_dir "Results/Prefix_Base/Llama3/" \
    --data_num 9000 \
    --n_prefix 14 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --template_index 3 \
    --layer_type "all" \
    --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=1 python prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/clean_train.json" \
    --output_dir "Results/Prefix_Base/Llama3/" \
    --data_num 9000 \
    --n_prefix 16 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --template_index 3 \
    --layer_type "all" \
    --num_train_epochs 3
    
fi