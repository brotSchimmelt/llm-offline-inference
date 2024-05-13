import logging
import math
import re
from collections import defaultdict
from typing import List, Union

import vllm
from icecream import ic
from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm.outputs import RequestOutput

from .config import LOGGING, RANDOM_SEED
from .generation_params import GenerationParams
from .llm import LLM, ModelOutput
from .utils import get_time

##### SETUP LOGGING #####
if LOGGING["disable_icecream"]:
    ic.disable()
logger = logging.getLogger(LOGGING["logger_name"])
##### SETUP LOGGING #####


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
        max_logprobs: int = 10,
        **kwargs,
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
            max_logprobs=max_logprobs,
            **kwargs,
        )

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._settings.model_path)

    def unpack_output(self, output: List[RequestOutput]) -> Union[List[str], List[List[str]]]:
        """
        Extracts text data from a list of RequestOutput objects.

        Args:
            output (List[RequestOutput]): A list of RequestOutput objects.
        Returns:
            Union[List[str], List[List[str]]]: Unpacked text data.
        """
        if len(output[0].outputs) > 1:
            return [[o.text for o in request_output.outputs] for request_output in output]
        return [o.outputs[0].text for o in output]

    @get_time
    def generate(
        self,
        prompts: Union[List[str], str],
        generation_params: GenerationParams,
        return_type: str = "str",
        json_schema: BaseModel = None,
        choices: List[str] = None,
        custom_regex: str = None,
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
            return_type (str, optional): The format of the generated output. string "str",
                vllm.RequestOutput "request_output" or ModelOutput "model_output".
                Defaults to "str".
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.
            custom_regex (str, optional): A custom regular expression to guide the generation.
                Defaults to None.
            batch_size (int, optional): The number of prompts to process in each batch.
                Use 0 for dynamic batching. Defaults to 0.
            use_tqdm (bool, optional): Whether to display a progress bar during
                generation. Defaults to True.
            system_prompt (str, optional): A system prompt to prepend to each input.
                Defaults to None.

        Raises:
            ValueError: If both json_schema and choices are provided, or if an invalid
                batch_size is given.

        Returns:
            Union[List[str], List[RequestOutput]]: The generated text or structured
                output based on return_type.
        """
        if sum(opt is not None for opt in [json_schema, choices, custom_regex]) > 1:
            raise ValueError("Cannot use multiple guided generation methods.")

        if batch_size < 0 or not isinstance(batch_size, int):
            raise ValueError(f"Invalid batch size: {batch_size}. batch_size > 0!")

        # convert generation params to vllm.SamplingParams
        sampling_params = self._create_sampling_params(generation_params)
        ic(sampling_params)

        # format prompts with the model's template
        prompts = self.format_prompts(prompts, system_prompt)
        ic(prompts[0])
        logger.info(f"First formatted prompt: {prompts[0]}")

        if json_schema or choices or custom_regex:
            sampling_params = self._configure_guided_generation(
                sampling_params, json_schema, choices, custom_regex
            )

        if batch_size > 0:
            outputs = self._manually_batched_generation(
                prompts, sampling_params, batch_size, use_tqdm
            )
        else:
            outputs = self._dynamically_batched_generation(prompts, sampling_params, use_tqdm)

        logger.info(f"Generated {len(outputs)} outputs.")
        logger.info(f"First output: {outputs[0].outputs[0].text}")

        return self._post_process_model_output(outputs, return_type, json_schema, choices)

    def _create_sampling_params(self, generation_params: GenerationParams) -> vllm.SamplingParams:
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
        custom_regex: str,
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
            custom_regex (str, optional): A custom regular expression to guide the generation.
                Defaults to None.

        Returns:
            vllm.SamplingParams: Updated sampling parameters with logits processors for
                guided generation.
        """
        if json_schema:
            logits_processor = JSONLogitsProcessor(
                schema=json_schema, llm=self.model, whitespace_pattern=r" ?"
            )

        elif custom_regex:
            logits_processor = RegexLogitsProcessor(regex_string=custom_regex, llm=self.model)

        else:
            choices_regex = "(" + "|".join([re.escape(c) for c in choices]) + ")"
            logits_processor = RegexLogitsProcessor(regex_string=choices_regex, llm=self.model)

        sampling_params.logits_processors = [logits_processor]
        logger.info("Configured sampling parameters for guided generation.")
        return sampling_params

    def aggregate_outputs(self, raw_outputs: List[RequestOutput]) -> ModelOutput:
        prompts, outputs, output_token_ids, cum_probs, token_probs = [], [], [], [], []

        for output in raw_outputs:
            prompts.append(output.prompt)
            outputs.append(output.outputs[0].text)
            output_token_ids.append(output.outputs[0].token_ids)
            cum_probs.append(output.outputs[0].cumulative_logprob)

            local_probs = []
            for prob in output.outputs[0].logprobs:
                prob_dict = defaultdict(float)

                for value in prob.values():
                    prob_dict[value.decoded_token] += math.exp(value.logprob)

                local_probs.append(dict(prob_dict))
            token_probs.append(local_probs)

        return ModelOutput(outputs, prompts, output_token_ids, cum_probs, token_probs)
