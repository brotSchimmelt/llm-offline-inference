import pytest

from src.flex_infer.generation_params import GenerationParams


def test_valid_initialization():
    params = GenerationParams()
    assert params.n == 1
    assert params.best_of == 1


def test_best_of_less_than_n_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(n=2, best_of=1)


def test_negative_presence_penalty_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(presence_penalty=-3)


def test_high_repetition_penalty_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(repetition_penalty=3)


def test_negative_temperature_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(temperature=-1)


def test_top_p_out_of_bounds_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(top_p=2)


def test_top_k_invalid_value_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(top_k=-2)


def test_min_p_out_of_bounds_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(min_p=-1)


def test_max_tokens_invalid_value_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(max_tokens=0)


def test_logprobs_invalid_value_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(logprobs=-1)


def test_prompt_logprobs_invalid_value_raises_error():
    with pytest.raises(ValueError):
        GenerationParams(prompt_logprobs=-1)


def test_beam_search_with_invalid_best_of():
    with pytest.raises(ValueError):
        GenerationParams(use_beam_search=True, best_of=1)


def test_beam_search_with_nonzero_temperature():
    with pytest.raises(ValueError):
        GenerationParams(use_beam_search=True, temperature=1)


def test_get_vllm_params():
    params = GenerationParams(n=1, best_of=1)
    vllm_params = params.get_vllm_params()
    assert isinstance(vllm_params, dict)
    assert vllm_params["n"] == 1


def test_get_transformers_params():
    params = GenerationParams()
    transformers_params = params.get_transformers_params()
    assert isinstance(transformers_params, dict)
    assert transformers_params["max_new_tokens"] == 16
