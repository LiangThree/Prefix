#!/bin/bash
# pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# layer_type: scaling/bias/all
# 当学习率达到0.01量级时scaling才会发生变化 

condition=$1

if [ "$condition" == "train" ]; then

    CUDA_VISIBLE_DEVICES=0 python Llama/train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "ReFT/dataset/math_10k/train.json" \
    --output_dir "RED_Results/Llama3/" \
    --data_num 9000 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --template_index 1 \
    --layer_type "all" \
    --num_train_epochs 3

elif [ "$condition" == "prefix" ]; then

    CUDA_VISIBLE_DEVICES=0 python prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math_10k/train.json" \
    --output_dir "Prefix_Results/Llama3/" \
    --data_num 9000 \
    --op_position "attn_o" \
    --learning_rate 2e-5 \
    --template_index 1 \
    --layer_type "all" \
    --num_train_epochs 2

elif [ "$condition" == "eval" ]; then

    python Llama/eval.py \
    --data_path "Base_Results/Llama3" \
    --eval_num 10000 

elif [ "$condition" == "answer" ]; then
    
    python RED/Llama/answer.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --template_index 1 \
    --dataset "gsm8k" \
    --peft "RED" \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Llama3/9000_attn_o_bias_2e-05/epoch2" \
    --repetition_penalty 1.1 \
    --data_num 300 

elif [ "$condition" == "lora_train" ]; then
    
    python RED/Llama/lora_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "ReFT/dataset/math_10k/train.json" \
    --output_dir "Results/Llama/" \
    --data_num 9000 \
    --template_index 1 \
    --layer 15

elif [ "$condition" == "lora_eval" ]; then
    
    python RED/Llama/lora_eval.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --run_type "lora" \
    --dataset "gsm8k" \
    --lora_path "Results/Llama/9000_lora"\
    --template_index 1 \
    --data_num 500 

elif [ "$condition" == "make_train" ]; then
    
    python Llama/make_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --output_path "Base_Results/Llama3/" \
    --dataset "prm800k" \
    --template_index 3 \
    --data_num 100 \
    --vllm True

elif [ "$condition" == "base_eval" ]; then
    
    python Llama/base_eval.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --output_path "Base_Results/Llama3/"\
    --dataset "prm800k" \
    --template_index 1 \
    --data_num 10000 \
    --vllm True \

elif [ "$condition" == "run" ]; then

    python RED/Llama/answer.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --template_index 1 \
    --dataset "gsm8k" \
    --peft "RED" \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Llama3/9000_attn_o_scaling_1e-2" \
    --repetition_penalty 1.1 \
    --data_num 300 

fi