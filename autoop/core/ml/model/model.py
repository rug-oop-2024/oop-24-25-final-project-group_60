
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy

class Model(ABC, Artifact):
    """Base class for all models.
    """
    def __init__(self, **kwargs):
        super(Artifact, self).__init__(type="model", **kwargs)
        self._parameters = {}
        

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self):
        return deepcopy(self._parameters)
