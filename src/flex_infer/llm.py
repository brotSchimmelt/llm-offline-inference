import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import transformers
import vllm
from icecream import ic
from outlines.integrations.transformers import (
    JSONPrefixAllowedTokens,
    RegexPrefixAllowedTokens,
)
from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
from pydantic import BaseModel
from vllm.outputs import RequestOutput

from .config import LOGGING, PROMPT_FORMATS, RANDOM_SEED, SUPPORTED_QUANTIZATION_MODES
from .generation_params import GenerationParams
from .utils import get_time, validate_choice, validate_json

##### SETUP LOGGING #####
if LOGGING["disable_icecream"]:
    ic.disable()
logger = logging.getLogger(LOGGING["logger_name"])
##### SETUP LOGGING #####


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
    num_gpus: int
    seed: int
    quant: str
    trust_remote_code: bool

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise ValueError(f"Invalid path: {self.model_path}")

        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"Invalid prompt template: {self.prompt_format}")

        if self.quant is not None:
            if self.quant not in SUPPORTED_QUANTIZATION_MODES:
                raise ValueError(f"Invalid quantization mode: {self.quant}")


class LLM(ABC):
    def __init__(self, **kwargs) -> None:
        self._type, self.model = None, None

        # load and validate model settings
        self._settings = ModelSettings(**kwargs)
        ic(self._settings)
        logger.info(f"Loaded model settings: {self._settings}")

        # load prompt settings
        self._prompt_settings = PROMPT_FORMATS[self._settings.prompt_format]
        self.prompt_template = self._prompt_settings["prompt_template"]
        self.system_prompt_template = self._prompt_settings["system_prompt_template"]
        self.eos_token = self._prompt_settings["eos_token"]
        ic(self._prompt_settings)
        logger.info(f"Loaded prompt settings: {self._prompt_settings}")

    @abstractmethod
    def generate(self) -> Union[List[str], List[Any]]:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def unpack_output(self) -> Union[List[str], List[List[str]]]:
        """Unpack the model output."""
        pass

    def self_consistency_generate(
        self,
        prompts: Union[List[str], str],
        generation_params_list: List[GenerationParams],
        **kwargs,
    ) -> List[str]:
        """
        Generates outputs for given prompts using multiple generation parameters for
        self-consistency. This method is designed to apply a majority vote mechanism
        across different outputs to find the most consistent results. It is recommended
        to use an odd number of generation parameters to avoid ties.

        Args:
            prompts (Union[List[str], str]): A single prompt or a list of prompts for
                generation.
            generation_params_list (List[GenerationParams]): A list of GenerationParams
                objects. Each object specifies a unique set of parameters to be used for
                generating text.

        Returns:
            List[str]: A list of most frequently generated output for each prompt.
        """
        if not isinstance(generation_params_list, list):
            logger.warning("Self-consistency requires a list of generation params.")

        if len(generation_params_list) % 2 == 0:
            logger.warning(
                "Self-consistency uses majority voting. Provide an odd number of "
                "generation params."
            )

        if kwargs.get("json_schema") is None and kwargs.get("choices") is None:
            logger.warning("Using self-consistency without guided generation!")

        if kwargs.get("return_string") is None:
            kwargs["return_string"] = True

        outputs = []
        for generation_params in generation_params_list:
            # for self-consistency, we want to generate only one output
            generation_params.n = 1

            outputs.append(
                self.generate(
                    prompts=prompts,
                    generation_params=generation_params,
                    **kwargs,
                )
            )

        return self._majority_vote(outputs)

    def _majority_vote(self, outputs: List[List[str]]) -> List[str]:
        """
        Determines the most frequently occurring string in each position across a list
        of output lists. In the event of a tie, the lexicographically earliest string
        is chosen.

        Args:
            outputs (List[List[str]]): A list where each element is a list of strings
            generated by the model under different parameters for the same prompt. Each
            inner list is expected to be the same length.

        Returns:
            List[str]: A list of strings where each element is the most common string.
        """
        result = []
        for items in zip(*outputs):
            counts = Counter(items)
            max_count = max(counts.values())
            tied_items = [item for item, count in counts.items() if count == max_count]
            # break ties alphabetically
            result.append(sorted(tied_items)[0])

        return result

    def format_prompts(
        self, prompts: Union[List[str], str], system_prompt: str = None
    ) -> List[str]:
        """Format prompts using the model's template."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if system_prompt:
            return [
                self.system_prompt_template.format(
                    system_prompt=system_prompt, prompt=p.strip()
                ).strip()
                for p in prompts
            ]
        return [self.prompt_template.format(p.strip()).strip() for p in prompts]

    def validate_model_output(
        self,
        output: Union[str, List[str]],
        json_schema: BaseModel = None,
        choices: List[str] = None,
    ) -> bool:
        """
        Validates the output of a model against specified criteria.

        Args:
            output (Union[str, List[str]]): The model's output to validate, can be a
                single string or a list of strings.
            json_schema (BaseModel, optional): A Pydantic BaseModel representing the
                JSON schema against which to validate the output. Defaults to None.
            choices (List[str], optional): A list of strings representing the valid
                choices against which to validate the output. Defaults to None.

        Returns:
            Tuple[bool, Optional[List[Tuple[int, str]]]]: A tuple where the first
                element is a boolean indicating whether the output is valid, and the
                second element is None if the output is valid or a list of tuples
                (index, value) for each invalid output element if the output is invalid.
        """
        if not output:
            raise ValueError("No output to validate.")

        if isinstance(output, str):
            output = [output]

        if isinstance(output[0], str):
            output = [[o] for o in output]

        malformed = []
        for output_idx, inner_output in enumerate(output):
            for candidate_idx, o in enumerate(inner_output):
                if json_schema:
                    if not validate_json(o):
                        malformed.append((output_idx, candidate_idx, o))
                elif choices:
                    if not validate_choice(o, choices):
                        malformed.append((output_idx, candidate_idx, o))
                else:
                    if not isinstance(o, str):
                        malformed.append((output_idx, candidate_idx, o))

        if malformed:
            return False, malformed
        return True, None

    def _post_process_model_output(
        self,
        outputs: List[Any],
        return_string: bool,
        json_schema: BaseModel = None,
        choices: List[str] = None,
    ) -> Union[List[str], List[Any]]:
        """
        Adds post-processing steps to the model output, such as converting it to a list
        and validating the model output.

        Args:
            outputs (List[Any]): Model output to post-process.
            return_string (bool): Whether to return the output as a string.
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.

        Returns:
            Union[List[str], List[Any]]: The post-processed model output.
        """
        unpacked_output = self.unpack_output(outputs)

        valid, malformed = self.validate_model_output(
            unpacked_output, json_schema, choices
        )
        if not valid:
            logger.warning("Invalid model output found. See logs for details.")
            logger.info(f"Malformed output: {malformed}")

        if return_string:
            return unpacked_output
        return outputs

    def get_model_settings(self) -> Dict[str, Any]:
        """Getter for the model settings."""
        return asdict(self._settings)

    def __str__(self) -> str:
        settings_str = "\n".join(
            [f"{k}: {v}" for k, v in self.get_model_settings().items()]
        )
        return f"{self._type} Instance\n{settings_str}"

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
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            model_path=model_path,
            prompt_format=prompt_format,
            num_gpus=num_gpus,
            seed=seed,
            quant=quant,
            trust_remote_code=trust_remote_code,
        )

        self._type = self.__class__.__name__

        # load the model
        self.model = vllm.LLM(
            model=self._settings.model_path,
            seed=self._settings.seed,
            tensor_parallel_size=self._settings.num_gpus,
            quantization=self._settings.quant,
            trust_remote_code=self._settings.trust_remote_code,
        )

    def unpack_output(
        self, output: List[RequestOutput]
    ) -> Union[List[str], List[List[str]]]:
        """
        Extracts text data from a list of RequestOutput objects.

        Args:
            output (List[RequestOutput]): A list of RequestOutput objects.
        Returns:
            Union[List[str], List[List[str]]]: Unpacked text data.
        """
        if len(output[0].outputs) > 1:
            return [
                [o.text for o in request_output.outputs] for request_output in output
            ]
        return [o.outputs[0].text for o in output]

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
        system_prompt: str = None,
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
        if json_schema and choices:
            raise ValueError("Cannot use guided generation for both JSON and RegEx.")

        if batch_size < 0 or not isinstance(batch_size, int):
            raise ValueError(f"Invalid batch size: {batch_size}. batch_size > 0!")

        # convert generation params to vllm.SamplingParams
        sampling_params = self._create_sampling_params(generation_params)
        ic(sampling_params)

        # format prompts with the model's template
        prompts = self.format_prompts(prompts, system_prompt)
        ic(prompts[0])
        logger.info(f"First formatted prompt: {prompts[0]}")

        if json_schema or choices:
            sampling_params = self._configure_guided_generation(
                sampling_params, json_schema, choices
            )

        if batch_size > 0:
            outputs = self._manually_batched_generation(
                prompts, sampling_params, batch_size, use_tqdm
            )
        else:
            outputs = self._dynamically_batched_generation(
                prompts, sampling_params, use_tqdm
            )

        logger.info(f"Generated {len(outputs)} outputs.")
        logger.info(f"First output: {outputs[0].outputs[0].text}")

        return self._post_process_model_output(
            outputs, return_string, json_schema, choices
        )

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
        logger.info("Generation with manual batching.")
        logger.info(f"Manually batching generation with batch size: {batch_size}")
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
        logger.info("Generation with dynamic batching.")
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
        logger.info("Configured sampling parameters for guided generation.")
        return sampling_params


class TransformersLLM(LLM):
    """This class extends LLM, adapting it to utilize specific functionalities
    of Transformers models including support for guided generation.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        prompt_format: str,
        num_gpus: int = 1,
        seed: int = RANDOM_SEED,
        quant: str = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            model_path=model_path,
            prompt_format=prompt_format,
            num_gpus=num_gpus,
            seed=seed,
            quant=quant,
            trust_remote_code=trust_remote_code,
        )

        self._type = self.__class__.__name__

        self.model_kwargs = {
            "model": self._settings.model_path,
            "trust_remote_code": self._settings.trust_remote_code,
        }

        if self._settings.num_gpus == 1:
            self.model_kwargs["device"] = 0
        else:
            self.model_kwargs["device_map"] = "auto"

        transformers.set_seed(self._settings.seed)

        self.model = transformers.pipeline("text-generation", **self.model_kwargs)

    def unpack_output(
        self, outputs: List[List[Dict[str, str]]]
    ) -> Union[List[str], List[List[str]]]:
        """
        Extracts text data from a list of Dicts.

        Args:
            outputs (List[List[Dict[str, str]]]): A list of Dicts.
        Returns:
            Union[List[str], List[List[str]]]: Unpacked text data.
        """
        if len(outputs[0]) > 1:
            return [[o["generated_text"] for o in output] for output in outputs]
        return [o[0]["generated_text"] for o in outputs]

    @get_time
    def generate(
        self,
        prompts: Union[List[str], str],
        generation_params: GenerationParams,
        return_string: bool = True,
        json_schema: BaseModel = None,
        choices: List[str] = None,
        use_tqdm: bool = True,
        system_prompt: str = None,
    ) -> Union[List[str], List[List[Dict[str, str]]]]:
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
            use_tqdm (bool, optional): Whether to display a progress bar during
                generation. Defaults to True.

        Raises:
            ValueError: If both json_schema and choices are provided.

        Returns:
            Union[List[str], List[List[Dict[str, str]]]]: The generated text or
                structured output based on return_type.
        """
        if json_schema and choices:
            raise ValueError("Cannot use guided generation for both JSON and RegEx.")

        generation_params = self._get_generation_params(generation_params)
        ic(generation_params)

        # set the pad token id to eos token id to suppress transformer warnings
        generation_params["pad_token_id"] = self.model.tokenizer.eos_token_id

        prompts = self.format_prompts(prompts, system_prompt)
        ic(prompts[0])
        logger.info(f"First formatted prompt: {prompts[0]}")

        if json_schema or choices:
            generation_params = self._configure_guided_generation(
                generation_params, json_schema, choices
            )

        transformers.set_seed(self._settings.seed)

        outputs = self.model(prompts, **generation_params)

        logger.info(f"Generated {len(outputs)} outputs.")

        return self._post_process_model_output(
            outputs, return_string, json_schema, choices
        )

    def _get_generation_params(
        self, generation_params: GenerationParams
    ) -> Dict[str, Any]:
        """
        Converts a GenerationParams object to a dictionary of parameters compatible with
        the Transformers library.

        Args:
            generation_params (GenerationParams): Parameters to control the generation
                behavior.

        Returns:
            Dict[str, Any]: The converted generation parameters.
        """
        return generation_params.get_transformers_params()

    def _configure_guided_generation(
        self,
        generation_params: Dict[str, Any],
        json_schema: BaseModel,
        choices: List[str],
    ) -> Dict[str, Any]:
        """
        Configures the generation parameters for guided generation based on a JSON
        schema or a list of choices.

        Args:
            generation_params (Dict[str, Any]): The original generation parameters.
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.

        Returns:
            Dict[str, Any]: Updated generation parameters with logits processors for
                guided generation.
        """
        if json_schema:
            prefix_allowed_tokens_fn = JSONPrefixAllowedTokens(
                schema=json_schema,
                tokenizer_or_pipe=self.model,
                whitespace_pattern=r" ?",
            )
        else:
            choices_regex = "(" + "|".join([re.escape(c) for c in choices]) + ")"
            prefix_allowed_tokens_fn = RegexPrefixAllowedTokens(
                regex_string=choices_regex, tokenizer_or_pipe=self.model
            )

        # currently there is a bug in the integration for outlines in transformers that
        # causes the generation of more than one sequence to fail when using a guided
        # generation. outlines==0.0.36
        if generation_params["num_return_sequences"] > 1:
            generation_params["num_return_sequences"] = 1
            logger.warning(
                "Setting 'num_return_sequences' (n) to 1 due to a bug in outlines "
                "0.0.36 with transformers integration. Change to vLLM for multiple "
                "return sequence support."
            )

        generation_params["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        logger.info("Configured generation parameters for guided generation.")
        return generation_params
