import logging
import re
from typing import Any, Dict, List, Union

import transformers
from icecream import ic
from outlines.integrations.transformers import (
    JSONPrefixAllowedTokens,
    RegexPrefixAllowedTokens,
)
from pydantic import BaseModel

from .config import LOGGING, RANDOM_SEED
from .generation_params import GenerationParams
from .llm import LLM, ModelOutput
from .utils import get_time

##### SETUP LOGGING #####
if LOGGING["disable_icecream"]:
    ic.disable()
logger = logging.getLogger(LOGGING["logger_name"])
##### SETUP LOGGING #####


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
        batch_size: int = 0,
        use_tqdm: bool = True,
        system_prompt: str = None,
        max_logprobs: int = 0,  # TODO implement functionality
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
            system_prompt (str, optional): A system prompt to prepend to the prompt.
            batch_size (int, optional): NOT SUPPORTED YET. Defaults to 0.

        Raises:
            ValueError: If both json_schema and choices are provided.

        Returns:
            Union[List[str], List[List[Dict[str, str]]]]: The generated text or
                structured output based on return_type.
        """
        if json_schema and choices:
            raise ValueError("Cannot use guided generation for both JSON and RegEx.")

        if batch_size != 0:
            logger.info("Batch size is not supported yet.")

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

        return self._post_process_model_output(outputs, return_string, json_schema, choices)

    def _get_generation_params(self, generation_params: GenerationParams) -> Dict[str, Any]:
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

    def aggregate_outputs(self, outputs: Any) -> ModelOutput:
        raise NotImplementedError("Method not implemented for TransformersLLM.")
