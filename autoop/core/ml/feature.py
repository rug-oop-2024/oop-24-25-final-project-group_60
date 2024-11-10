from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """Represents a feature used in a dataset.

    A feature can either be of type 'numerical' or 'categorical', and
    contains metadata about the feature's name and type.
    """
    name: str = Field()
    type: Literal["numerical", "categorical"] = Field()

    def __str__(self) -> str:
        """Returns a string representation of the feature.

        Returns:
            str: The name of the feature.
        """
        return self.name
