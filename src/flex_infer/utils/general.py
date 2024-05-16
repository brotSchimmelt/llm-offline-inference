import ast
import json
import logging
import os
import re
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Dict, List

import pandas as pd

from flex_infer.config import LOGGING

logger = logging.getLogger(LOGGING["logger_name"])


def get_time(func: Callable) -> Callable:
    """Decorates a function to log its execution time.

    Args:
        func (Callable): The function to be measured and wrapped.

    Returns:
        Callable: A wrapper function that, when called, will execute the original func,
        measure and log its execution time, and then return the result of func.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()

        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        formatted_seconds = str(int(seconds)).zfill(2)

        logger.info(f"'{func.__name__}()' took {int(minutes)}:{formatted_seconds} min")
        return result

    return wrapper


def is_valid_json(json_string: str) -> bool:
    """
    Validates whether the given string is a properly formatted JSON.

    Args:
        json_string (str): The string to be validated as JSON. It should be a string
            representation of a JSON object.

    Returns:
        bool: A boolean value indicating whether the input string is a valid JSON
            format.
    """
    if not json_string or not isinstance(json_string, str):
        return False

    try:
        obj = ast.literal_eval(json_string)
        return isinstance(obj, dict)
    except (SyntaxError, ValueError):
        return False


def validate_choice(s: str, choices: List[str]) -> bool:
    """
    Validates if the given string is among a list of specified choices.

    Args:
        s (str): The string to validate against the list of choices.
        choices (List[str]): A list of strings representing the valid options.

    Raises:
        TypeError: Raised if the `choices` parameter is not a list, ensuring that the
            function operates on the expected types.

    Returns:
        bool: A boolean indicating whether the string `s` is found within the `choices`
            list.
    """
    if not s or not isinstance(s, str):
        return False

    if not isinstance(choices, list):
        raise TypeError("choices must be a list of strings")

    return s in choices


def is_valid_binary_sequence(seq: List[int]) -> bool:
    """
    Checks if a sequence consists only of integers and that these are only 0 and 1.

    Args:
        seq (list or array-like): The sequence to check.

    Returns:
        bool: True if valid, False otherwise.
    """
    return all(isinstance(item, int) for item in seq) and all(item in [0, 1] for item in seq)


def save_df_to_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Saves a pandas DataFrame to a CSV file, with checks for directory validity and
    file extension.

    Args:
        df (pd.DataFrame): The pandas DataFrame to save.
        file_path (str): The path (including file name and extension) where the CSV file
                         will be saved. If the file exists, it will be overwritten.
        index (bool, optional): Whether to include the DataFrame index in the CSV file.
                                Defaults to False, meaning the index will not be saved.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory) and directory != "":
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    if not file_path.lower().endswith(".csv"):
        file_path += ".csv"

    df.to_csv(file_path, index=index)


def correct_json_output(outputs: List[str]) -> Dict:
    """
    Corrects a list of JSON output strings and returns a list of dictionaries.

    Args:
        outputs (List[str]): A list of JSON strings to be corrected.

    Returns:
        List[Dict]: A list of dictionaries with the corrected JSON strings, or empty dictionaries if
            all fail.
    """
    if not isinstance(outputs, list):
        outputs = [outputs]

    def fix_json_string(json_string: str) -> str:
        """Try to fix a JSON string that is not properly formatted."""
        # remove extra whitespace and newlines
        json_string = re.sub(r"\s+", " ", json_string).strip()

        # fix unclosed braces
        if not json_string.startswith("{"):
            json_string = "{" + json_string
        if not json_string.endswith("}"):
            json_string = json_string + "}"

        # remove extra commas before closing braces
        json_string = re.sub(r",\s*([}\]])", r"\1", json_string)

        # ensure proper key-value structure with quoted keys and values
        json_string = re.sub(r"([,{]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_string)

        # ensure values are quoted properly
        json_string = re.sub(r'(:\s*")([^"]*?)(\s*[,}])', r'\1\2"\3', json_string)

        # fix any unclosed string values (e.g., missing closing quote)
        matches = re.findall(r'":\s*"[^"]*$', json_string)
        if matches:
            json_string = re.sub(r'(": "[^"]*)$', r'\1"', json_string)

        return json_string

    fixed_outputs = []
    for output in outputs:
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            corrected_output = fix_json_string(output)

            try:
                parsed_output = json.loads(corrected_output)
            except json.JSONDecodeError:
                # return string as is if it still fails
                parsed_output = output

        fixed_outputs.append(parsed_output)

    return fixed_outputs
