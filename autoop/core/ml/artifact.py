from pydantic import BaseModel, Field
import base64
from typing import Dict, Any, List
from copy import deepcopy

class Artifact:
    def __init__(self, name: str, asset_path: str, version: str, data: bytes,
                 type: str, metadata: Dict[str, Any] = {"userid": "admin"},
                 tags: List[str] = ["default"]):
        
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self._data = data
        self.metadata = metadata
        self.type = type
        self.tags = tags
        self.id = self.generate_id(asset_path, version)
        

    def generate_id(self, asset_path, version) -> str:
        encoded_path = base64.b64encode(asset_path.encode("utf-8")).decode("utf-8")
        encoded_path = encoded_path.rstrip("=")
        return f"{encoded_path}_{version}"
    
    def read(self) -> bytes:
        return self.data
    
    def save(self, data: bytes) -> bytes:
        self.data = data
        return data
    
    @property
    def data(self) -> bytes:
        return deepcopy(self._data)
