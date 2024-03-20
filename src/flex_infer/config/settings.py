LOGGING = {
    "logger_name": "flex_infer",
    "disable_icecream": False,
}

RANDOM_SEED = 42

SUPPORTED_QUANTIZATION_MODES = ["awq", "gptq"]

TEST_MODEL_SETTINGS = {
    "model_name": "gemma-2b-instruct",
    "model_path": "/bigwork/nhwpknet/models/gemma-2b-it",
    "prompt_format": "llama2",
}
