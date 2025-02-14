# LLM Offline Inference

This is a comprehensive toolkit designed to streamline offline inference for large language models (LLMs). It focuses on processing large batches of prompts efficiently by wrapping popular libraries such as [vLLM](https://github.com/vllm-project/vllm) and Hugging Face [Transformers](https://github.com/huggingface/transformers), and it is built with future extensibility in mind — with plans to integrate additional libraries like [SGLang](https://github.com/sgl-project/sglang).

## Overview

### Key Features

- **Multi-Library Integration:** Seamless support for various LLM inference libraries, enabling you to switch or combine frameworks as needed.

- **Dynamic & Manual Batching:** Optimize throughput with efficient batch processing strategies that allow for both dynamically determined and manually specified batch sizes.

- **Guided Generation:** Leverage advanced guided generation techniques through JSON schemas or regular expression-based choices for more controlled and deterministic outputs with the [outlines](https://github.com/dottxt-ai/outlines) library.

- **Unified Generation Parameters:** Fine-tune generation settings with a unified set of parameters over different libraries to control model behavior.

### Performance and Scalability

Built for high-throughput offline inference, the toolkit's robust batching mechanisms and performance logging ensure efficient processing of large volumes of prompts. Its design focuses on maximizing resource utilization and delivering scalable performance even in resource-intensive scenarios.

## Installation

Install the library with:

```bash
git clone https://github.com/brotSchimmelt/llm-offline-inference.git
cd llm-offline-inference

pip install -e .
```

## Usage

```python
from pydantic import BaseModel
from llm_offline_inference import VLLM, GenerationParams

# initialize the model
llm = VLLM(
  name="human friendly model name",
  model_path="/path/to/model",
  prompt_format="llama2" # supported prompt formats are found in config/prompt_formats.py
)

# set the parameters for model generation
generation_params = GenerationParams(
  temperature=0.5,
  max_tokens=16,
  # list with all parameters is found in generation_params.py
)

# setup JSON return schema (optional)
class City(BaseModel):
  city: str


output = llm.generate(
    "What is the capital of Iceland?",
    generation_params,
    return_string=True,
    json_schema=City, # optional
    system_prompt="You are a helpful assistant.", # optional
)
# output[0]: { "city": "Reykjavik" }
```

## License

This project is licensed under the MIT license.
