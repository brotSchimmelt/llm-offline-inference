PROMPT_FORMATS = {
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
    "chatml": {
        "prompt_template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
        "system_prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",  # noqa: E501
        "eos_token": "<|im_end|>",
    },
    "chatml_dbrx": {
        "prompt_template": "<|im_start|>user\n{}<|im_end|><|endoftext|>\n<|im_start|>assistant",  # noqa: E501
        "system_prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|><|endoftext|>\n<|im_start|>user\n{prompt}<|im_end|><|endoftext|>\n<|im_start|>assistant",  # noqa: E501
        "eos_token": "<|endoftext|>",
    },
    "cohere": {
        "prompt_template": "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",  # noqa: E501
        "system_prompt_template": "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",  # noqa: E501
        "eos_token": "<|END_OF_TURN_TOKEN|>",
    },
    "gemma": {
        "prompt_template": "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",  # noqa: E501
        "system_prompt_template": "<bos><start_of_turn>user\n{system_prompt}\n{prompt}<end_of_turn>\n<start_of_turn>model\n",  # noqa: E501
        "eos_token": "<eos>",
    },
    "llama2": {
        "prompt_template": "<s>[INST] {} [/INST]",
        "system_prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",  # noqa: E501
        "eos_token": "</s>",
    },
    "llama3": {
        "prompt_template": "<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
        "system_prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
        "eos_token": "<|eot_id|>",
    },
    "no_style": {
        "prompt_template": "{}",
        "system_prompt_template": "{system_prompt}\n{prompt}",
        "eos_token": "",
    },
    "vicuna": {
        "prompt_template": "USER: {} ASSISTANT:",
        "system_prompt_template": "{system_prompt} USER: {prompt} ASSISTANT:",
        "eos_token": "",
    },
    "zephyr": {
        "prompt_template": "<|user|>\n{}</s>\n<|assistant|>\n",
        "system_prompt_template": "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",  # noqa: E501
        "eos_token": "</s>",
    },
}
