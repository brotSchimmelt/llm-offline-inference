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

## ToDo

- add output validation for guided generation
- add support for custom prompt templates
- add examples
  - inference with vllm or transformers
  - guided inference
  - custom template
- add evaluation for classification (f1, precision, recall)
