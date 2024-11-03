
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(Artifact):
    def __init__(self, *args, **kwargs):
        super().__init__(type="model", *args, **kwargs)
        
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass
    