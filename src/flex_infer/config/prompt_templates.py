PROMPT_TEMPLATES = {
    "no_style": {
        "format": "{}",
        "format_system": "{system_prompt} {prompt}",
        "end_of_seq": "",
    },
    "llama2": {
        "format": "<s>[INST] {} [/INST]",
        "format_system": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",  # noqa: E501
        "end_of_seq": "</s>",
    },
    "zephyr": {
        "format": "<|user|>\n{}</s>\n<|assistant|>\n",
        "format_system": "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",  # noqa: E501
        "end_of_seq": "</s>",
    },
    "chatml": {
        "format": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
        "format_system": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",  # noqa: E501
        "end_of_seq": "<|im_end|>",
    },
    "alpaca": {
        "format": "### Instruction:\n{}\n\n### Response:\n",
        "format_system": "### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n{prompt}",  # noqa: E501
        "end_of_seq": "",
    },
    "alpaca_human": {
        "format": "### HUMAN:\n{}\n\n### RESPONSE:\n",
        "format_system": "### HUMAN:\n{system_prompt}\n\n### INPUT:\n{prompt}\n\n### RESPONSE:\n{prompt}",  # noqa: E501
        "end_of_seq": "",
    },
    "vicuna": {
        "format": "USER: {} ASSISTANT:",
        "format_system": "{system_prompt} USER: {prompt} ASSISTANT:",
        "end_of_seq": "",
    },
}
