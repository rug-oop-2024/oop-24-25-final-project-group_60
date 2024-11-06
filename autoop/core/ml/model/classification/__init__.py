from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from autoop.core.ml.model.model import Model

class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
        self.type = "classification"
        self.name = "Logistic Regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        print(ground_truths)
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class DecisionTreeClassificationModel(Model):
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
        self.type = "classification"
        self.name = "Decision Tree"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths, )
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)

class RandomForestClassificationModel(Model):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.type = "classification"
        self.name = "Random Forest"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        self.model.fit(observations, ground_truths)
        self._data= self.model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)