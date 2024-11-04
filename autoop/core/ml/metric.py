from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "root_mean_squared_error",
    "precision",
    "recall"
]

def get_metric(name: str):
    """Factory function to get a metric by name.
    """
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "accuracy":
            return Accuracy()
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "root_mean_squared_error":
            return RootMeanSquaredError()
        case "precision":
            return Precision()
        case "recall":
            return Recall()
        case _:
            raise ValueError(f"Unknown metric: {name}")

class Metric(ABC):
    """Base class for all metrics.
    """
    @abstractmethod
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        pass

# add here concrete implementations of the Metric class
class MeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean((ground_truth - prediction) ** 2)
    
class Accuracy(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean(ground_truth == prediction)
    
class MeanAbsoluteError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean(np.abs(ground_truth - prediction))
    
class RootMeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.sqrt(np.mean((ground_truth - prediction) ** 2))
    
class Precision(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        true_positives = np.sum((ground_truth == 1) & (prediction == 1))
        false_positives = np.sum((ground_truth == 0) & (prediction == 1))
        return true_positives / (true_positives + false_positives)
    
class Recall(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        true_positives = np.sum((ground_truth == 1) & (prediction == 1))
        false_negatives = np.sum((ground_truth == 1) & (prediction == 0))
        return true_positives / (true_positives + false_negatives)
