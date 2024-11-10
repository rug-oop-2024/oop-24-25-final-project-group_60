from abc import ABC, abstractmethod
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
    """Factory function to retrieve a metric object by name.

    This function returns the corresponding metric class for the specified
    metric name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: A metric class corresponding to the input name.

    Raises:
        ValueError: If the provided metric name is not recognized.
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

    This class defines the interface for all metric classes that calculate
    the performance of a model by comparing predictions to ground truth values.

    Attributes:
        name (str): The name of the metric.
    """
    def __init__(self, name: str):
        self.name = "model_name"

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the metric value for given ground truth and predictions.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed metric value.
        """
        pass


class MeanSquaredError(Metric):
    """Mean Squared Error metric.

    This metric calculates the average squared difference between predicted
    and actual values.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "mean_squared_error"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the Mean Squared Error.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed mean squared error.
        """
        return np.mean((ground_truth - prediction) ** 2)


class Accuracy(Metric):
    """Accuracy metric.

    This metric calculates the proportion of correct predictions.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "accuracy"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the accuracy.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The accuracy of the predictions.
        """
        return np.mean(ground_truth == prediction)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric.

    This metric calculates the average absolute difference between predicted
    and actual values.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "mean_absolute_error"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the Mean Absolute Error.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed mean absolute error.
        """
        return np.mean(np.abs(ground_truth - prediction))


class RootMeanSquaredError(Metric):
    """Root Mean Squared Error metric.

    This metric calculates the square root of the average squared difference
    between predicted and actual values.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "root_mean_squared_error"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the Root Mean Squared Error.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed root mean squared error.
        """
        return np.sqrt(np.mean((ground_truth - prediction) ** 2))


class Precision(Metric):
    """Precision metric.

    This metric calculates the proportion of positive predictions that are
    correct.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "precision"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate precision.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed precision.
        """
        true_positives = np.sum((ground_truth == 1) & (prediction == 1))
        false_positives = np.sum((ground_truth == 0) & (prediction == 1))
        return (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )


class Recall(Metric):
    """Recall metric.

    This metric calculates the proportion of actual positives that were
    correctly identified.

    Args:
        name (str): The name of the metric.
    """
    def __init__(self):
        self.name = "recall"

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate recall.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed recall.
        """
        true_positives = np.sum((ground_truth == 1) & (prediction == 1))
        false_negatives = np.sum((ground_truth == 1) & (prediction == 0))
        return (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0)
