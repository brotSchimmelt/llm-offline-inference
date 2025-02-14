from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .utils import is_valid_binary_sequence


def evaluate_binary_classification(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluates the performance of a binary classification model on given true labels and
    predictions.

    Args:
        y_true (list or array-like): The true labels of the instances. Must be binary
            (0 or 1).
        y_pred (list or array-like): The predicted labels as outputted by the
            classification model. Must be binary (0 or 1).

    Returns:
        dict: A dictionary containing the computed metrics ('f1', 'precision',
            'recall', 'accuracy') as keys and their respective scores as values.
    """
    if not is_valid_binary_sequence(y_true):
        raise ValueError("y_true must be a binary sequence of 0s and 1s")

    if not is_valid_binary_sequence(y_pred):
        raise ValueError("y_pred must be a binary sequence of 0s and 1s")

    return {
        "f1": f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
