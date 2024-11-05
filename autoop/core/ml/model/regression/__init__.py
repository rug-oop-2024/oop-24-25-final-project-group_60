from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
from autoop.core.ml.model.model import Model

class LinearRegressionModel(Model):
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        self.type = "regression"
        self.name = "Linear Regression"
        super().__init__(name = self.name, type = self.type)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class DecisionTreeRegressionModel(Model):
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        self.type = "regression"
        self.name = "Decision Tree"
        super().__init__(name = self.name, type = self.type)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class RandomForestRegressionModel(Model):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        self.type = "regression"
        self.name = "Random Forest"
        super().__init__(name = self.name, type = self.type)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)
    
class MultipleLinearRegression(Model):
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        self.type = "regression"
        self.name = "Multiple Linear Regression"
        super().__init__(name = self.name, type = self.type)
    
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)
