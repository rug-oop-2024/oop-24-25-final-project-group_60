
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name: str = Field()
    type: Literal["numerical", "categorical"] = Field()

    def __str__(self):
        return f"'{self.name}' of type ({self.type})"