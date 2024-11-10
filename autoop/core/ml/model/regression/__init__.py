from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    """Linear Regression model for regression tasks."""

    def __init__(self, **kwargs):
        """Initializes the Linear Regression model.

        Args:
            **kwargs: Keyword arguments passed to the LinearRegression model.
        """
        self.model = LinearRegression(**kwargs)
        self.type = "regression"  # Set model type
        self.name = "Linear Regression"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth values for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts values for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions
            on.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(observations)


class DecisionTreeRegressionModel(Model):
    """Decision Tree model for regression tasks."""

    def __init__(self, **kwargs):
        """Initializes the Decision Tree model.

        Args:
            **kwargs: Keyword arguments passed to the DecisionTreeRegressor
            model.
        """
        self.model = DecisionTreeRegressor(**kwargs)
        self.type = "regression"  # Set model type
        self.name = "Decision Tree"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth values for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts values for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions
            on.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(observations)


class RandomForestRegressionModel(Model):
    """Random Forest model for regression tasks."""

    def __init__(self, **kwargs):
        """Initializes the Random Forest model.

        Args:
            **kwargs: Keyword arguments passed to the RandomForestRegressor
            model.
        """
        self.model = RandomForestRegressor(**kwargs)
        self.type = "regression"  # Set model type
        self.name = "Random Forest"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth values for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts values for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions
            on.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(observations)


class MultipleLinearRegression(Model):
    """Multiple Linear Regression model for regression tasks."""

    def __init__(self, **kwargs):
        """Initializes the Multiple Linear Regression model.

        Args:
            **kwargs: Keyword arguments passed to the LinearRegression model.
        """
        self.model = LinearRegression(**kwargs)
        self.type = "regression"  # Set model type
        self.name = "Multiple Linear Regression"  # Set model name

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): The input features for training.
            ground_truths (np.ndarray): The ground truth values for training.
        """
        self.model.fit(observations, ground_truths)  # Fit the model
        self._data = self.model.get_params()  # Save the model parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts values for the given input features.

        Args:
            observations (np.ndarray): The input features to make predictions
            on.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(observations)
