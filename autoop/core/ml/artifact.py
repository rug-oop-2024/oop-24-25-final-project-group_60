from pydantic import BaseModel, Field
import base64
from typing import Dict, Any, List

class Artifact:
    def __init__(self, type: str, name: str, asset_path: str, data: bytes, 
                 version: str):
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.id = self.generate_id()

    def generate_id(self) -> str:
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def update_version(self, new_version: str):
        self.version = new_version
        self.id = self.generate_id()

