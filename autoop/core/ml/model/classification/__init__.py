from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autoop.core.ml.model.model import Model


class LogisticRegressionModel(Model):
    """Logistic Regression model for classification tasks."""
    
    def __init__(self, **kwargs):
        """Initializes the Logistic Regression model.

        Args:
            **kwargs: Keyword arguments passed to the LogisticRegression model.
        """
        self.model = LogisticRegression(**kwargs)
        self.type = "classification"  # Set model type
        self.name = "Logistic Regression"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth labels for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model with data
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts labels for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions on.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(observations)


class DecisionTreeClassificationModel(Model):
    """Decision Tree model for classification tasks."""

    def __init__(self, **kwargs):
        """Initializes the Decision Tree model.

        Args:
            **kwargs: Keyword arguments passed to the DecisionTreeClassifier model.
        """
        self.model = DecisionTreeClassifier(**kwargs)
        self.type = "classification"  # Set model type
        self.name = "Decision Tree"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth labels for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model with data
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts labels for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions on.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(observations)


class RandomForestClassificationModel(Model):
    """Random Forest model for classification tasks."""

    def __init__(self, **kwargs):
        """Initializes the Random Forest model.

        Args:
            **kwargs: Keyword arguments passed to the RandomForestClassifier model.
        """
        self.model = RandomForestClassifier(**kwargs)
        self.type = "classification"  # Set model type
        self.name = "Random Forest"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth labels for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model with data
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts labels for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions on.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(observations)