from pydantic import BaseModel, Field
import base64
from typing import Dict, Any, List

class Artifact(dict):
    def __init__(self, asset_path, version="1.0.0", **kwargs):
        items = kwargs
        items["asset_path"] = asset_path
        items["version"] = version
        super().__init__(items) 

        self._id = self.generate_id(asset_path, version)

    def generate_id(self, asset_path, version) -> str:
        encoded_path = base64.b64encode(asset_path.encode()).decode()
        return f"{encoded_path}:{version}"
    
    def read(self) -> bytes:
        return self.data
    
    def save(self, data: bytes) -> bytes:
        self.data = data
        return self.data
