
prompt_template = {
    0:"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:""",

    1:"""<|start_header_id|>user<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",

    2:"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:%s\n\n### Response:<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n""" ,

    3:"""<|start_header_id|>user<|end_header_id|>\n\n%s Please provide the initial step towards resolving the question. This
step may serve as a foundation but might not encompass the entire solution.\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
}