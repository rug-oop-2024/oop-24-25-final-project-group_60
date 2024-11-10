import base64
from typing import Dict, Any, List
from copy import deepcopy


class Artifact:
    """Represents an artifact in the system.

    An artifact can store data, metadata, and tags, along with the associated
    versioning and asset path for identification.
    """

    def __init__(self, name: str, asset_path: str, version: str, data: bytes,
                 type: str, metadata: Dict[str, Any] = {"userid": "admin"},
                 tags: List[str] = ["default"]):
        """Initializes the Artifact with the specified parameters.

        Args:
            name (str): The name of the artifact.
            asset_path (str): The path of the asset to identify it.
            version (str): The version of the artifact.
            data (bytes): The raw data stored within the artifact.
            type (str): The type of the artifact (e.g., 'model', 'dataset').
            metadata (Dict[str, Any], optional): Metadata associated with the
                artifact. Defaults to {"userid": "admin"}.
            tags (List[str], optional): Tags associated with the artifact.
                Defaults to ["default"].
        """
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self._data = data
        self.metadata = metadata
        self.type = type
        self.tags = tags
        self.id = self.generate_id(asset_path, version)

    def generate_id(self, asset_path: str, version: str) -> str:
        """Generates a unique ID for the artifact based on its asset path and
        version.

        Args:
            asset_path (str): The asset path of the artifact.
            version (str): The version of the artifact.

        Returns:
            str: A unique identifier string for the artifact.
        """
        encoded_path = base64.b64encode(asset_path.encode("utf-8")).decode(
                                                                    "utf-8")
        encoded_path = encoded_path.rstrip("=")
        return f"{encoded_path}_{version}"

    def read(self) -> bytes:
        """Returns the data stored in the artifact.

        Returns:
            bytes: The raw data associated with the artifact.
        """
        return self.data

    def save(self, data: bytes) -> bytes:
        """Saves new data to the artifact.

        Args:
            data (bytes): The new data to be stored in the artifact.

        Returns:
            bytes: The saved data.
        """
        self.data = data
        return data

    @property
    def data(self) -> bytes:
        """Retrieves a deep copy of the artifact's data.

        Returns:
            bytes: A deep copy of the stored data.
        """
        return deepcopy(self._data)
