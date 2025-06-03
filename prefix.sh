#!/bin/bash
# old: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# layer_type: scaling/bias/all 当学习率达到0.01量级时scaling才会发生变化 
# mistral v0.3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3

# Meta-Llama-3-8B-Instruct | Qwen2.5-7B-Instruct | Mistral-7B-Instruct-v0.2 | DeepSeek-R1-Distill-Qwen-7B
# template: llama3 | qwen_base | qwen_base_fewshot | mistral | none

# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token
# 训练时n_prefix内置为-1


condition=$1

if [ "$condition" == "red" ]; then

    CUDA_VISIBLE_DEVICES=0 python Prefix/red_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/RED/Llama3" \
    --data_num 9000 \
    --op_position "attn_o" \
    --learning_rate 2e-4 \
    --template_index "llama3" \
    --layer_type "all" \
    --num_train_epochs 3

    # CUDA_VISIBLE_DEVICES=0 python Prefix/red_train.py \
    # --model_path "Qwen2.5-7B-Instruct" \
    # --data_path  "dataset/math10k/train.json" \
    # --output_dir "Results/RED/Qwen2.5-Instruct" \
    # --data_num 9000 \
    # --op_position "attn_o" \
    # --learning_rate 2e-4 \
    # --template_index "qwen_base" \
    # --layer_type "all" \
    # --num_train_epochs 3

    # CUDA_VISIBLE_DEVICES=0 python Prefix/red_train.py \
    # --model_path "Mistral-7B-Instruct-v0.3" \
    # --data_path  "dataset/math10k/train.json" \
    # --output_dir "Results/RED/Mistral-v0.3" \
    # --data_num 9000 \
    # --op_position "attn_o" \
    # --learning_rate 2e-4 \
    # --template_index "mistral" \
    # --layer_type "all" \
    # --num_train_epochs 3

elif [ "$condition" == "prefix" ]; then

    # python Prefix/prefix_train.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --data_path  "dataset/math10k/train.json" \
    # --output_dir "Results/Test/Llama3/" \
    # --data_num 9000 \
    # --n_prefix 64 \
    # --op_position "ffn_up" \
    # --learning_rate 2e-4 \
    # --layer_type "all" \
    # --num_train_epochs 3 \
    # --template_index "llama3"

    python Prefix/prefix_train.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test/Qwen2.5/" \
    --data_num 9000 \
    --n_prefix 64 \
    --op_position "ffn_up" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --template_index 'qwen_base' \
    --num_train_epochs 3

    python Prefix/prefix_train.py \
    --model_path "Mistral-7B-Instruct-v0.3" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test/Mistral-v0.3/" \
    --data_num 9000 \
    --n_prefix 32 \
    --op_position "ffn_up" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --template_index 'mistral' \
    --num_train_epochs 3


elif [ "$condition" == "eval" ]; then

    python Prefix/eval.py \
    --data_path "Results/Base/base" \
    --eval_num 500

    python Prefix/eval.py \
    --data_path "Results/Lora" \
    --eval_num 500

    python Prefix/eval.py \
    --data_path "Results/RED" \
    --eval_num 500

    python Prefix/eval.py \
    --data_path "Results/Test" \
    --eval_num 500

elif [ "$condition" == "answer" ]; then

    python Prefix/answer.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 8 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test/Llama3/9000_math10k_all_64_2e-4/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "llama3" \
    --data_num 500

    # python Prefix/answer.py \
    # --model_path "Qwen2.5-7B-Instruct" \
    # --dataset "gsm8k" \
    # --peft "RED" \
    # --n_prefix 10 \
    # --is_train_return False \
    # --no_repeat_ngram_size 5 \
    # --peft_path "Results/Test/Qwen2.5/9000_math10k_all_64_2e-4/ffn_up" \
    # --repetition_penalty 1.1 \
    # --template_index "qwen_base" \
    # --data_num 500


elif [ "$condition" == "lora_train" ]; then
    
    python lora/lora_train.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Lora/Qwen2.5/" \
    --learning_rate 5e-4 \
    --data_num 9000 \
    --template_index "chat_template" \
    --layer 15

elif [ "$condition" == "lora_eval" ]; then
    
    # python lora/lora_eval.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --lora_path "Results/Lora/Llama3/9000_math10k_lora_1e-3"\
    # --dataset "gsm8k" \
    # --data_num 500 \
    # --template_index "llama3"

    python lora/lora_eval.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --lora_path "Results/Lora/Qwen2.5/9000_math10k_lora_5e-4"\
    --dataset "gsm8k" \
    --data_num 500 \
    --template_index "chat_template" 

    python lora/lora_eval.py \
    --model_path "Qwen2.5-7B-Instruct" \
    --lora_path "Results/Lora/Qwen2.5/9000_math10k_lora_5e-4"\
    --dataset "gsm8k" \
    --data_num 500 \
    --template_index "fewshot_chat_template"

elif [ "$condition" == "base_eval" ]; then

    python Prefix/base_eval.py \
    --model_path "Qwen3-8B" \
    --output_path "Results/Base/base/9000_Qwen3_all_0_0" \
    --dataset "gsm8k" \
    --data_num 500 \
    --template_index "chat_template" \
    --vllm False
   
elif [ "$condition" == "llama" ]; then

    python Prefix/base_eval.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --output_path "Results/Test/base/9000_Llama3_all_0_0/gsm8k_eval_base.json" \
    --dataset "gsm8k" \
    --data_num 500 \
    --template_index "llama3_fewshot" \
    --vllm False 
    
fi