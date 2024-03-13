from functools import wraps
from time import perf_counter
from typing import Any, Callable

from ..config import PROMPT_TEMPLATES


def get_time(func: Callable) -> Callable:
    """Decorates a function to print its execution time.

    Args:
        func (Callable): The function to be measured and wrapped.

    Returns:
        Callable: A wrapper function that, when called, will execute the original func,
        measure and print its execution time, and then return the result of func.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        print(f"'{func.__name__}()' took {int(minutes)}:{seconds:.1f} min")
        return result

    return wrapper


def check_prompt_format(format_name: str) -> bool:
    """
    Checks if the given format name exists in the predefined list of prompt templates.

    Args:
        format_name (str): The name of the format to check.

    Raises:
        ValueError: If `format_name` is not found within `PROMPT_TEMPLATES`, a
                    ValueError is raised with a message indicating the invalid format
                    name and listing the available format names.

    Returns:
        bool: Returns True if `format_name` exists within `PROMPT_TEMPLATES`, indicating
              a valid format name. If a ValueError is raised due to an invalid format
              name, the function does not return a value.
    """
    if not format_name:
        raise ValueError(
            "No prompt format provided. Please provide a valid format name."
        )
        return False

    if format_name not in PROMPT_TEMPLATES.keys():
        raise ValueError(
            f"Invalid prompt format: {format_name}. "
            f"Available formats: {', '.join(list(PROMPT_TEMPLATES.keys()))}"
        )
        return False
    return True
