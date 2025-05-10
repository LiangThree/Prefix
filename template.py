
prompt_template = {
    0:"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:""",

    1:"""<|start_header_id|>user<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",

    2:"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n""" ,

    "llama3": """<|start_header_id|>user<|end_header_id|>\n\n%s Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution.\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",

    "qwen_reft": """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n%s.<|im_end|>\n<|im_start|>assistant\n""",

    "qwen_base": """'<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n%s.<|im_end|>\n<|im_start|>assistant\n'""",

    "qwen_math": """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"""
}