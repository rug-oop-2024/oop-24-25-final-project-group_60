
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy

class Model(Artifact, ABC):
    """Base class for all models.
    """
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(name = "model_name", type = "model" , 
                         asset_path = "asset_path", version = "1_0_0", 
                         data = {},**kwargs)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self):
        return deepcopy(self.data)
