from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy


class Model(Artifact, ABC):
    """Abstract base class for all machine learning models.

    This class must be extended by specific model implementations (e.g.,
    LinearRegressionModel, RandomForestModel).
    """

    def __init__(self, name: str, type: str, **kwargs):
        """Initializes the model with the given name, type, and optional
        parameters.

        Args:
            name (str): The name of the model.
            type (str): The type of the model (e.g., 'classification' or
                 'regression').
            **kwargs: Additional model-specific arguments.
        """
        super().__init__(name="model_name", type="model",
                         asset_path="asset_path", version="1_0_0",
                         data={}, **kwargs)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Trains the model using the provided observations and ground truth
        values.

        Args:
            observations (np.ndarray): The input data used for training.
            ground_truths (np.ndarray): The correct labels/values for the
                                        training data.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generates predictions for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: Predicted values based on the observations.
        """
        pass

    @property
    def parameters(self):
        """Gets the model's parameters.

        Returns:
            dict: A deep copy of the model's internal parameters.
        """
        return deepcopy(self.data)
