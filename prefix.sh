#!/bin/bash
# pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# layer_type: scaling/bias/all
# 当学习率达到0.01量级时scaling才会发生变化 

# Meta-Llama-3-8B-Instruct  template: 3
# Meta-Llama-3.1-8B-Instruct 
# Qwen2.5-Math-7B-Instruct  template: 4
# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token
# 训练时n_prefix内置为-1

condition=$1

if [ "$condition" == "red" ]; then

    # CUDA_VISIBLE_DEVICES=0 python red_train.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --data_path  "dataset/math10k/train.json" \
    # --output_dir "Results/RED/Llama3/" \
    # --data_num 9000 \
    # --op_position "attn_o" \
    # --learning_rate 2e-4 \
    # --layer_type "all" \
    # --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=0 python red_train.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/RED/Qwen2.5-instruct/" \
    --data_num 9000 \
    --n_prefix 32 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 3


elif [ "$condition" == "prefix" ]; then

    CUDA_VISIBLE_DEVICES=0 python prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test/Llama3/" \
    --data_num 9000 \
    --n_prefix 32 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=0 python prefix_train.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test/Qwen2.5-Math/" \
    --data_num 9000 \
    --n_prefix 32 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=0 python prefix_train.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test/Qwen2.5/" \
    --data_num 9000 \
    --n_prefix 32 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 3

elif [ "$condition" == "eval" ]; then

    python Llama/eval.py \
    --data_path "Results/Test/base" \
    --eval_num 500

    python Llama/eval.py \
    --data_path "Results/Test/Qwen2.5" \
    --eval_num 500

    python Llama/eval.py \
    --data_path "Results/Test/Qwen2.5-Math" \
    --eval_num 500

    python Llama/eval.py \
    --data_path "Results/Test/Llama3" \
    --eval_num 500


elif [ "$condition" == "answer" ]; then

    # python Llama/answer.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --dataset "gsm8k" \
    # --peft "RED" \
    # --n_prefix 8 \
    # --is_train_return False \
    # --no_repeat_ngram_size 5 \
    # --peft_path "Results/Test/Llama3/9000_math10k_all_32_2e-4/attn_o" \
    # --repetition_penalty 1.1 \
    # --data_num 500


    python Llama/answer.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 8 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 10 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 12 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 14 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 8 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5-Math/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 10 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5-Math/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 12 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5-Math/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    python Llama/answer.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 14 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Qwen2.5-Math/9000_math10k_all_32_2e-4/attn_o" \
    --repetition_penalty 1.1 \
    --data_num 500

    
elif [ "$condition" == "lora_train" ]; then
    
    python lora/lora_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Lora/Llama3/" \
    --learning_rate 1e-3 \
    --data_num 9000 \
    --layer 15

    python lora/lora_train.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Lora/Qwen2.5/" \
    --learning_rate 1e-3 \
    --data_num 9000 \
    --layer 15

    python lora/lora_train.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Lora/Qwen2.5-Math/" \
    --learning_rate 1e-3 \
    --data_num 9000 \
    --layer 15

elif [ "$condition" == "base_eval" ]; then
    
    # python Llama/base_eval.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --output_path "Results/Test/Llama3/9000_math10k_base_0_0/gsm8k_eval_base.json" \
    # --dataset "gsm8k" \
    # --data_num 500 \
    # --vllm False 

    python Llama/base_eval.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --output_path "Results/Test/Qwen2.5-Math_attn_o/9000_math10k_all_0_0/gsm8k_eval_base.json" \
    --dataset "gsm8k" \
    --data_num 500 \
    --vllm False 

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