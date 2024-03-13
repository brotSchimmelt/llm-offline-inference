from functools import wraps
from time import perf_counter
from typing import Any, Callable


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
