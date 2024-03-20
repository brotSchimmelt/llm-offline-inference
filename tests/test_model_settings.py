import pytest

from src.flex_infer.llm import ModelSettings

PROMPT_FORMATS = ["llama2"]
SUPPORTED_QUANTIZATION_MODES = ["awq"]


@pytest.fixture
def valid_settings():
    """Provides a valid ModelSettings instance for testing."""
    return ModelSettings(
        name="TestModel",
        model_path="./",
        prompt_format=PROMPT_FORMATS[0],
        num_gpus=1,
        seed=42,
        quant=SUPPORTED_QUANTIZATION_MODES[0],
        trust_remote_code=True,
    )


def test_initialization(valid_settings):
    """Test if a ModelSettings instance is initialized properly."""
    assert valid_settings.name == "TestModel"
    assert valid_settings.model_path == "./"
    assert valid_settings.prompt_format == PROMPT_FORMATS[0]
    assert valid_settings.num_gpus == 1
    assert valid_settings.seed == 42
    assert valid_settings.quant == SUPPORTED_QUANTIZATION_MODES[0]
    assert valid_settings.trust_remote_code is True


def test_invalid_path():
    """Test initialization with an invalid path."""
    with pytest.raises(ValueError):
        ModelSettings(
            name="TestModel",
            model_path="/invalid/path",
            prompt_format=PROMPT_FORMATS[0],
            num_gpus=1,
            seed=42,
            quant=SUPPORTED_QUANTIZATION_MODES[0],
            trust_remote_code=True,
        )


def test_invalid_prompt_format():
    """Test initialization with an invalid prompt format."""
    with pytest.raises(ValueError):
        ModelSettings(
            name="TestModel",
            model_path="./",
            prompt_format="invalid_format",
            num_gpus=1,
            seed=42,
            quant=SUPPORTED_QUANTIZATION_MODES[0],
            trust_remote_code=True,
        )


def test_invalid_quantization_mode():
    """Test initialization with an invalid quantization mode."""
    with pytest.raises(ValueError):
        ModelSettings(
            name="TestModel",
            model_path="/valid/path",
            prompt_format=PROMPT_FORMATS[0],
            num_gpus=1,
            seed=42,
            quant="invalid_quant",
            trust_remote_code=True,
        )
