import pytest

from src.flex_infer import GenerationParams, TransformersLLM
from src.flex_infer.config.settings import TEST_MODEL_SETTINGS

GENERATION_PARAMS = GenerationParams(temperature=0.1, max_tokens=16, n=2)


@pytest.fixture
def valid_llm():
    """Provides a valid LLM instance for testing."""

    # configure the test model settings in the config/settings.py file
    return TransformersLLM(
        name=TEST_MODEL_SETTINGS["model_name"],
        model_path=TEST_MODEL_SETTINGS["model_path"],
        prompt_format=TEST_MODEL_SETTINGS["prompt_format"],
    )


def test_format_prompts_single_str_prompt(valid_llm):
    template = valid_llm.prompt_template
    prompt = " Single String Prompt  "
    expected = template.format(prompt.strip())
    assert valid_llm.format_prompts(prompt) == [expected]


def test_format_prompts_list_of_str_prompts(valid_llm):
    template = valid_llm.prompt_template
    prompts = ["Test prompt 1", "Test prompt 2"]
    expected = [template.format(p.strip()) for p in prompts]
    assert valid_llm.format_prompts(prompts) == expected


def test_format_prompts_with_system_prompt(valid_llm):
    template = valid_llm.system_prompt_template
    system_prompt = "I am a system prompt"
    prompts = ["Test prompt 1", "Test prompt 2"]
    expected = [template.format(system_prompt=system_prompt, prompt=p.strip()) for p in prompts]
    assert valid_llm.format_prompts(prompts, system_prompt=system_prompt) == expected


def test_get_model_settings(valid_llm):
    model_settings = valid_llm.get_model_settings()
    assert model_settings["name"] == TEST_MODEL_SETTINGS["model_name"]
    assert model_settings["model_path"] == TEST_MODEL_SETTINGS["model_path"]
    assert model_settings["prompt_format"] == TEST_MODEL_SETTINGS["prompt_format"]
    assert model_settings["num_gpus"] == 1
    assert model_settings["seed"] == 42
    assert model_settings["quant"] is None
