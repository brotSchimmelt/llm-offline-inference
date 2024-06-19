import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from icecream import ic
from pydantic import BaseModel

from .config import LOGGING, PROMPT_FORMATS, SUPPORTED_QUANTIZATION_MODES
from .generation_params import GenerationParams
from .utils import correct_json_output, is_valid_json, save_df_to_csv, validate_choice

##### SETUP LOGGING #####
if LOGGING["disable_icecream"]:
    ic.disable()
logger = logging.getLogger(LOGGING["logger_name"])
##### SETUP LOGGING #####


@dataclass
class ModelOutput:
    """Manages the output of a model, including prompts and probabilities.

    Attributes:
        output (List[str]): The model's output.
        prompts (List[str]): The prompts used to generate the output.
        output_token_ids (List[List[int]]): List of token IDs for the output.
        cumulative_probabilities (List[float]): The cumulative probabilities of each output.
        token_probabilities (List[Dict[str, float]]): The probabilities of each token.

    Examples:
        token_probabilities for 1 output with number of logprobs set to 2:
        >>> token_probabilities[0] -> [{'pred': 0.95633781516, 'pre': 0.0337628525670}, {...}]

        token_probabilities for the n-th token of the m-th generated output:
        >>> token_probabilities[m][n] -> {'pred': 0.95633781516, 'pre': 0.0337628525670}
    """

    output: List[str]
    prompts: List[str]
    output_token_ids: List[List[int]]
    cumulative_probabilities: List[float]
    token_probabilities: List[List[Dict[str, float]]]

    def __post_init__(self) -> None:
        self.cumulative_probabilities = list(np.exp(self.cumulative_probabilities))

    def convert_to_pd_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "outputs": self.output,
                "prompts": self.prompts,
                "output_token_ids": self.output_token_ids,
                "cumulative_probabilities": self.cumulative_probabilities,
                "token_probabilities": self.token_probabilities,
            }
        )

    def write_to_csv(self, path: str) -> None:
        df = self.convert_to_pd_dataframe()

        df["outputs"] = df["outputs"].str.replace("\n", " ", regex=False)
        df["prompts"] = df["prompts"].str.replace("\n", " ", regex=False)

        save_df_to_csv(df, path, index=False)


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

    @abstractmethod
    def aggregate_outputs(self, outputs: Any) -> ModelOutput:
        """Aggregate the model outputs into a single object with prompts, output and
        probabilities."""
        pass

    def convert_output_str_to_json(
        self, model_output: ModelOutput
    ) -> Dict[str, List[Union[str, Dict[str, str]]]]:
        """
        Converts the output string from a model to a JSON format and identifies invalid outputs.

        Args:
            model_output (ModelOutput): An object containing the output string from a model.

        Returns:
            Dict[str, List[Union[str, Dict[str, str]]]]: A dictionary with two keys:
                - "json_output": A list of the correctly formatted JSON outputs. Each element is
                    either a string or a dictionary with string keys and values.
                - "invalid_outputs": A list of the original outputs that could not be converted to
                    JSON, retaining their original string format.
        """
        original_output = model_output.output
        json_output = correct_json_output(original_output)

        invalid_outputs = []
        for idx, output in enumerate(json_output):
            if isinstance(output, str):
                invalid_outputs.append(original_output[idx])

        return {
            "json_output": json_output,
            "invalid_outputs": invalid_outputs,
        }

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

    def _find_answer_token_probability_distribution(
        self, model_output: ModelOutput, answer_choices: List[str]
    ) -> List[Dict[str, float]]:
        """
        Identifies and returns the probability of the first token in the model's output that matches
        any of the specified answer choices.

        Args:
            model_output (ModelOutput): Contains tokens and their probabilities as output by the
                model.
            answer_choices (List[str]): A list of strings that represent possible answers to search
                within the tokens.

        Returns:
            List[Dict[str, float]]: A list of all token probabilities that match any of the
                answer labels.
        """
        if isinstance(answer_choices, str):
            answer_choices = [answer_choices]

        results = []
        for row, token_ids in enumerate(model_output.output_token_ids):
            list_of_tokens = [self.tokenizer.decode(t) for t in token_ids]

            found_probability_flag = False
            for idx, token in enumerate(list_of_tokens):
                if token and token in "".join(answer_choices):
                    results.append(model_output.token_probabilities[row][idx])
                    # we only care for the first token that matches the answer since the llm already
                    # decided the answer at that point
                    found_probability_flag = True
                    break

            if not found_probability_flag:
                results.append({choice: 0.0 for choice in answer_choices})

        if len(results) != len(model_output.output):
            raise ValueError(
                f"The number of token distributions ({len(results)}) does not match "
                f"the number of outputs ({len(model_output.output)})."
            )

        return results

    def get_output_distribution(
        self, model_output: ModelOutput, answer_choices: List[str]
    ) -> List[Dict[str, float]]:
        """
        Computes and returns a list of dictionaries representing the probability distribution over
        provided answer choices based on the tokens and their probabilities from the model output.

        Args:
            model_output (ModelOutput): Contains tokens and their probabilities as output by the
                model.
            answer_choices (List[str]): A list of strings that represent possible answers to
                evaluate against the tokens.

        Returns:
            List[Dict[str, float]]: Each dictionary in the list corresponds to a set of answer
                choices and their cumulative probabilities calculated from the model's token
                outputs.
        """

        # List[Dict[str, float]]
        answer_token_distributions = self._find_answer_token_probability_distribution(
            model_output, answer_choices
        )

        results = []
        for token_probability_distribution in answer_token_distributions:
            aggregated_distribution = {label: 0.0 for label in answer_choices}

            for token, prob in token_probability_distribution.items():
                for choice in answer_choices:
                    if token and token in choice:
                        aggregated_distribution[choice] += prob

            results.append(aggregated_distribution)

        return results

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
                    if not is_valid_json(o):
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
        return_type: str,
        json_schema: BaseModel = None,
        choices: List[str] = None,
    ) -> Union[List[str], List[Any]]:
        """
        Adds post-processing steps to the model output, such as converting it to a list
        and validating the model output.

        Args:
            outputs (List[Any]): Model output to post-process.
            return_type (str): Type of the model output.
            json_schema (BaseModel, optional): A Pydantic model representing a JSON
                schema for guided generation. Defaults to None.
            choices (List[str], optional): A list of strings to guide the generation via
                regular expressions. Defaults to None.

        Returns:
            Union[List[str], List[Any]]: The post-processed model output.
        """
        unpacked_output = self.unpack_output(outputs)

        valid, malformed = self.validate_model_output(unpacked_output, json_schema, choices)
        if not valid:
            logger.warning("Invalid model output found. See logs for details.")
            logger.info(f"Malformed output: {malformed}")

        if return_type.lower() == "str":
            return unpacked_output
        elif return_type.lower() == "model_output":
            return self.aggregate_outputs(outputs)
        return outputs

    def get_model_settings(self) -> Dict[str, Any]:
        """Getter for the model settings."""
        return asdict(self._settings)

    def __str__(self) -> str:
        settings_str = "\n".join([f"{k}: {v}" for k, v in self.get_model_settings().items()])
        return f"{self._type} Instance\n{settings_str}"

    def __repr__(self) -> str:
        return self.__str__()
