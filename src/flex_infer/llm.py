import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import vllm
from icecream import ic
from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
from pydantic import BaseModel
from vllm.outputs import RequestOutput

from .config import PROMPT_FORMATS, RANDOM_SEED, SUPPORTED_QUANTIZATION_MODES
from .generation_params import GenerationParams
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
    model_path: str
    prompt_format: str
    num_gpus: int = 1
    seed: int = field(default_factory=lambda: RANDOM_SEED)
    quant: Optional[str] = None

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise ValueError(f"Invalid path: {self.model_path}")

        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"Invalid prompt template: {self.prompt_format}")

        if self.quant is not None:
            if self.quant not in SUPPORTED_QUANTIZATION_MODES:
                raise ValueError(f"Invalid quantization mode: {self.quant}")


class LLM(ABC):
    def __init__(
        self,
        name: str,
        model_path: str,
        prompt_format: str,
        num_gpus: int = 1,
        seed: int = RANDOM_SEED,
        quant: str = None,
    ) -> None:
        # load and validate model settings
        self._settings = ModelSettings(
            name, model_path, prompt_format, num_gpus, seed, quant
        )

        # load prompt settings
        self._prompt_settings = PROMPT_FORMATS[self._settings.prompt_format]
        self.prompt_template = self._prompt_settings["prompt_template"]
        self.system_prompt_template = self._prompt_settings["system_prompt_template"]
        self.eos_token = self._prompt_settings["eos_token"]

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
        return f"LLM: {self._settings.name}\n{settings_str}"

    def __repr__(self) -> str:
        return self.__str__()


class VLLM(LLM):
    """This class extends LLM, adapting it to utilize specific functionalities
    of vLLM models including support for dynamically batched and guided generation.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        prompt_format: str,
        num_gpus: int = 1,
        seed: int = RANDOM_SEED,
        quant: str = None,
    ) -> None:
        super().__init__(name, model_path, prompt_format, num_gpus, seed, quant)

        # load the model
        self.model = vllm.LLM(
            model=self._settings.model_path,
            seed=self._settings.seed,
            tensor_parallel_size=self._settings.num_gpus,
            quantization=self._settings.quant,
        )

    @get_time
    def generate(
        self,
        prompts: Union[List[str], str],
        generation_params: GenerationParams,
        return_string: bool = True,
        json_schema: BaseModel = None,
        choices: List[str] = None,
        batch_size: int = 0,
        use_tqdm: bool = True,
    ) -> Union[List[str], List[RequestOutput]]:
        """
        Generates text based on the given prompts and generation parameters, with
        optional support for guided generation using either a JSON schema or regular
        expression choices.

        Args:
            prompts (Union[List[str], str]): The prompt(s) to generate text for.
            generation_params (GenerationParams): Parameters to control the generation
                behavior.
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

        # convert generation params to vllm.SamplingParams
        sampling_params = self._create_sampling_params(generation_params)

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

    def _create_sampling_params(
        self, generation_params: GenerationParams
    ) -> vllm.SamplingParams:
        """
        Converts a GenerationParams object to a vllm.SamplingParams object.

        Args:
            generation_params (GenerationParams): Parameters to control the generation
                behavior.

        Returns:
            vllm.SamplingParams: The converted sampling parameters.
        """
        return vllm.SamplingParams(**generation_params.get_vllm_params())

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
            output = self.model.generate(
                prompts=current_prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
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
        return self.model.generate(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=use_tqdm
        )

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
