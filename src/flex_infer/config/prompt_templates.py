PROMPT_TEMPLATES = {
    "no_style": {
        "prompt_template": "{}",
        "system_prompt_template": "{system_prompt} {prompt}",
        "eos_token": "",
    },
    "llama2": {
        "prompt_template": "<s>[INST] {} [/INST]",
        "system_prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",  # noqa: E501
        "eos_token": "</s>",
    },
    "zephyr": {
        "prompt_template": "<|user|>\n{}</s>\n<|assistant|>\n",
        "system_prompt_template": "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",  # noqa: E501
        "eos_token": "</s>",
    },
    "chatml": {
        "prompt_template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
        "system_prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",  # noqa: E501
        "eos_token": "<|im_end|>",
    },
    "alpaca": {
        "prompt_template": "### Instruction:\n{}\n\n### Response:\n",
        "system_prompt_template": "### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n{prompt}",  # noqa: E501
        "eos_token": "",
    },
    "alpaca_human": {
        "prompt_template": "### HUMAN:\n{}\n\n### RESPONSE:\n",
        "system_prompt_template": "### HUMAN:\n{system_prompt}\n\n### INPUT:\n{prompt}\n\n### RESPONSE:\n{prompt}",  # noqa: E501
        "eos_token": "",
    },
    "vicuna": {
        "prompt_template": "USER: {} ASSISTANT:",
        "system_prompt_template": "{system_prompt} USER: {prompt} ASSISTANT:",
        "eos_token": "",
    },
}
