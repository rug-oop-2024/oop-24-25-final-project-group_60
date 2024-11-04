from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from model.model import Model

class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._parameters = self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class DecisionTreeClassificationModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths, )
        self._parameters = self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class RandomForestClassificationModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._parameters = self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)