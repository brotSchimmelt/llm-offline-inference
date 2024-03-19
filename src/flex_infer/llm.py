import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import vllm
from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
from pydantic import BaseModel
from vllm.outputs import RequestOutput

from .config import PROMPT_TEMPLATES, RANDOM_SEED, SUPPORTED_QUANTIZATION_MODES
from .utils import get_time


@dataclass
class ModelSettings:
    """Manages configuration settings for a model, including validation.

    Attributes:
        name (str): The name of the model.
        path (str): Filesystem path to the model.
        prompt_template_name (str): Template identifier for model prompts.
        num_gpus (int): Number of GPUs allocated for the model. Defaults to 1.
        seed (int): Seed value for random number generation. Defaults to a
        module-level `RANDOM_SEED`.
        quant (Optional[str]): Quantization mode for the model. Optional; defaults
        to None.
    """

    name: str
    path: str
    prompt_template_name: str
    num_gpus: int = 1
    seed: int = field(default_factory=lambda: RANDOM_SEED)
    quant: Optional[str] = None

    def __post_init__(self) -> None:
        if not Path(self.path).exists():
            raise ValueError(f"Invalid path: {self.path}")

        if self.prompt_template_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template: {self.prompt_template}")

        if self.quant is not None:
            if self.quant not in SUPPORTED_QUANTIZATION_MODES:
                raise ValueError(f"Invalid quantization mode: {self.quant}")

        self.prompt_template = PROMPT_TEMPLATES[self.prompt_template_name]


class LLM(ABC):
    def __init__(self, model_settings: Dict[str, Any]) -> None:
        self._settings = ModelSettings(**model_settings)
        self.prompt_template = self._settings.prompt_template["format"]
        self.eos_token = self._settings.prompt_template["end_of_seq"]
        self.system_prompt_template = self._settings.prompt_template["format_system"]
        self.name = self._settings.name

    @abstractmethod
    def generate(self) -> Union[List[str], List[Any]]:
        """Generate a response from the model."""
        pass

    def format_prompts(self, prompts: Union[List[str], str]) -> List[str]:
        """Format prompts using the model's template."""
        if isinstance(prompts, str):
            prompts = [prompts]
        return [self.prompt_template.format(p.strip()).strip() for p in prompts]

    def get_model_settings(self) -> Dict[str, Any]:
        """Getter for the model settings."""
        return asdict(self._settings)

    def __str__(self) -> str:
        settings_str = "\n".join(
            [f"{k}: {v}" for k, v in self.get_model_settings().items()]
        )
        return f"LLM: {self.name}\n{settings_str}"

    def __repr__(self) -> str:
        return self.__str__()


class VLLM(LLM):
    """This class extends LLM, adapting it to utilize specific functionalities
    of vLLM models including support for dynamically batched and guided generation.
    """

    def __init__(self, model_settings: Dict[str, Any]) -> None:
        super().__init__(model_settings)
        self.model = vllm.LLM(
            model=self._settings.path,
            seed=self._settings.seed,
            tensor_parallel_size=self._settings.num_gpus,
            quantization=self._settings.quant,
        )

    @get_time
    def generate(
        self,
        prompts: Union[List[str], str],
        sampling_params: vllm.SamplingParams,
        return_string: bool = True,
        json_schema: BaseModel = None,
        choices: List[str] = None,
        batch_size: int = 0,
        use_tqdm: bool = True,
    ) -> Union[List[str], List[RequestOutput]]:
        """
        Generates text based on the given prompts and sampling parameters, with optional
        support for guided generation using either a JSON schema or regular expression
        choices.

        Args:
            prompts (Union[List[str], str]): The prompt(s) to generate text for.
                sampling_params (vllm.SamplingParams): Parameters to control the
                sampling behavior.
            return_string (bool, optional): The format of the generated output (string
                or vllm.RequestOutput). Defaults to True.
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.
            batch_size (int, optional): The number of prompts to process in each batch.
                Use 0 for dynamic batching. Defaults to 0.
            use_tqdm (bool, optional): Whether to display a progress bar during
                generation. Defaults to True.

        Raises:
            ValueError: If both json_schema and choices are provided, or if an invalid
                batch_size is given.

        Returns:
            Union[List[str], List[RequestOutput]]: The generated text or structured
                output based on return_type.
        """
        # validate inputs
        if json_schema and choices:
            raise ValueError("Cannot use guided generation for both JSON and RegEx.")

        if batch_size < 0 or not isinstance(batch_size, int):
            raise ValueError(f"Invalid batch size: {batch_size}. batch_size > 0!")

        # format prompts with the model's template
        prompts = self.format_prompts(prompts)

        # create logits processors for guided generation
        if json_schema or choices:
            sampling_params = self._configure_guided_generation(
                sampling_params, json_schema, choices
            )

        # generate responses
        if batch_size > 0:
            outputs = self._manually_batched_generation(
                prompts, sampling_params, batch_size, use_tqdm
            )
        else:
            outputs = self._dynamically_batched_generation(
                prompts, sampling_params, use_tqdm
            )

        # convert outputs to string if necessary
        if return_string:
            return [o.outputs[0].text for o in outputs]
        return outputs

    def _manually_batched_generation(
        self,
        prompts: List[str],
        sampling_params: vllm.SamplingParams,
        batch_size: int,
        use_tqdm: bool,
    ) -> List[RequestOutput]:
        """
        Generates responses for a list of prompts in manually specified batches.

        Args:
            prompts (List[str]): The prompts to generate responses for.
            sampling_params (vllm.SamplingParams): Sampling parameters to use for
                generation.
            batch_size (int): The size of each batch.
            use_tqdm (bool): Whether to display a progress bar.

        Returns:
            List[RequestOutput]: A list of generated responses.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            current_prompts = prompts[i : i + batch_size]
            output = self.model.generate(current_prompts, sampling_params, use_tqdm)
            results.extend(output)
        return results

    def _dynamically_batched_generation(
        self, prompts: List[str], sampling_params: vllm.SamplingParams, use_tqdm: bool
    ) -> List[RequestOutput]:
        """
        Generates responses for a list of prompts using dynamic batching based on the
        model's capacity.

        Args:
            prompts (List[str]): The prompts to generate responses for.
            sampling_params (vllm.SamplingParams): Sampling parameters to use for
                generation.
            use_tqdm (bool): Whether to display a progress bar.

        Returns:
            List[RequestOutput]: A list of generated responses.
        """
        return self.model.generate(prompts, sampling_params, use_tqdm)

    def _configure_guided_generation(
        self,
        sampling_params: vllm.SamplingParams,
        json_schema: BaseModel,
        choices: List[str],
    ) -> vllm.SamplingParams:
        """
        Configures the sampling parameters for guided generation based on a JSON schema
        or a list of choices.

        Args:
            sampling_params (vllm.SamplingParams): The original sampling parameters.
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.

        Returns:
            vllm.SamplingParams: Updated sampling parameters with logits processors for
                guided generation.
        """
        if json_schema:
            logits_processor = JSONLogitsProcessor(
                schema=json_schema, llm=self.model, whitespace_pattern=r" ?"
            )

        else:
            choices_regex = "(" + "|".join([re.escape(c) for c in choices]) + ")"
            logits_processor = RegexLogitsProcessor(
                regex_string=choices_regex, llm=self.model
            )

        sampling_params.logits_processors = [logits_processor]
        return sampling_params


class TransformersLLM(LLM):
    """This class extends LLM, adapting it to utilize specific functionalities
    of Transformers models including support for guided generation.
    """

    pass
