
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    """Base class for all models.
    """
    def __init__(self):
        self._parameters = {}

    @abstractmethod
    def fit(self, dataset: Artifact):
        pass
    
    @abstractmethod
    def predict(self, dataset: Artifact) -> np.ndarray:
        pass
    