# Flex-Infer

flex-infer is a comprehensive toolkit for streamlining and simplifying the inference process for LLMs across various models and libraries.

## Installation

Clone the repository to your machine. If you clone it inside your project folder, please remember to adjust your ```.gitignore``` file.

```bash
git clone https://github.com/brotSchimmelt/flex-infer.git /path/to/flex-infer
```

Install the package with pip in editable mode in a fresh virtual environment.

```bash
pip install /path/to/flex-infer -e
```

By installing the package in editable mode, we can easily make changes to the code without re-installing it.

## Example Usage

```python
from pydantic import BaseModel
from flex_infer import VLLM, GenerationParams

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

## ToDo

- add support for custom prompt templates in LLM
- add examples
  - inference with vllm or transformers
  - guided inference
  - custom template
- add refusal detection with Sentence-Transformers
- add [tqdm for transformers](https://github.com/huggingface/transformers/issues/14789)
- add support for Self-Consistency
