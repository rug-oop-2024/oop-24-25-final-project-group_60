from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
from autoop.core.ml.model.model import Model

class LinearRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class DecisionTreeRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeRegressor(**kwargs)
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class RandomForestRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(**kwargs)
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)
    
class MultipleLinearRegression(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)
        self.type = "regression"
    
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)
