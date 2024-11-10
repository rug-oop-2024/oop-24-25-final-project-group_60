from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a path is not found in storage.

    Args:
        path (str): The path that could not be found.
    """
    def __init__(self, path):
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage systems. Defines the methods for saving,
       loading, deleting, and listing data."""

    @abstractmethod
    def save(self, data: bytes, path: str):
        """Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path where the data should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a given path.

        Args:
            path (str): Path from which to load data.

        Returns:
            bytes: The data loaded from the specified path.
        """
        pass

    @abstractmethod
    def delete(self, path: str):
        """Delete data at a given path.

        Args:
            path (str): Path where the data should be deleted.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """List all paths under a given path.

        Args:
            path (str): The path to list.

        Returns:
            list: A list of paths under the specified directory.
        """
        pass


class LocalStorage(Storage):
    """Local file implementation of the Storage class."""

    def __init__(self, base_path: str = "./assets"):
        """Initialize LocalStorage with a base path.

        Args:
            base_path (str): The base directory where data will be stored.
            Defaults to './assets'.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str):
        """Save data to a file under a specified key (path).

        Args:
            data (bytes): The data to save.
            key (str): The file path where the data should be stored.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Load data from a file specified by a key (path).

        Args:
            key (str): The path of the file to load data from.

        Returns:
            bytes: The data loaded from the file.

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/"):
        """Delete a file or directory at the specified key (path).

        Args:
            key (str): The path of the file or directory to delete.

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """List all file paths under a given prefix (directory).

        Args:
            prefix (str): The directory to list files from.

        Returns:
            List[str]: A list of relative file paths.

        Raises:
            NotFoundError: If the directory does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys
                if os.path.isfile(p)]

    def _assert_path_exists(self, path: str):
        """Assert that a given path exists.

        Args:
            path (str): The path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Join the base path with a given path, ensuring compatibility across
           platforms.

        Args:
            path (str): The path to join with the base path.

        Returns:
            str: The full normalized path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
