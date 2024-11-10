import json
from typing import Dict, Tuple, List, Union
import os

from autoop.core.storage import Storage

class Database():
    """A database class for storing, retrieving, and managing data.

    Args:
        storage (Storage): The storage backend used to persist data
    """

    def __init__(self, storage: Storage):
        """Initialize the database.

        Args:
            storage (Storage): The storage to use for persisting the data.
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """Store a new entry in a specified collection.

        Args:
            collection (str): The name of the collection where the data should be stored.
            id (str): The unique ID for the entry in the collection.
            entry (dict): The data to be stored.

        Returns:
            dict: The data that was stored in the database.
        
        Raises:
            AssertionError: If the entry is not a dictionary, or the collection or id is not a string.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """Retrieve an entry by its ID from a specified collection.

        Args:
            collection (str): The name of the collection to retrieve data from.
            id (str): The ID of the data to retrieve.

        Returns:
            Union[dict, None]: The data if found, or None if the entry does not exist.
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)
    
    def delete(self, collection: str, id: str):
        """Delete an entry by its ID from a specified collection.

        Args:
            collection (str): The name of the collection to delete the data from.
            id (str): The ID of the data to delete.

        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """List all entries in a specified collection.

        Args:
            collection (str): The name of the collection to list the data from.

        Returns:
            List[Tuple[str, dict]]: A list of tuples where each tuple contains the ID and data of an entry in the collection.
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self):
        """Reload the database by re-reading all data from the storage.
        """
        self._load()

    def _persist(self):
        """Persist the data to storage.

        This method saves each collection and its data to the storage. It also 
        removes any data from storage that is no longer present in the database.
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(json.dumps(item).encode(), f"{collection}{os.sep}{id}")

        # Remove deleted items from storage
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split(os.sep)[-2:]
            if not self._data.get(collection, id):
                self._storage.delete(f"{collection}{os.sep}{id}")
    
    def _load(self):
        """Load data from storage into memory.

        This method reads all the stored data from the storage and populates the in-memory database with it.
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{id}")
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
