from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Represents a dataset artifact in the system, extending the Artifact
    class.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the Dataset with specific parameters.

        Args:
            *args: Variable length argument list
            **kwargs: Keyword arguments
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1_0_0") -> 'Dataset':
        """Creates a Dataset from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be stored in the dataset.
            name (str): The name of the dataset.
            asset_path (str): The asset path of the dataset.
            version (str, optional): The version of the dataset.
                Defaults to "1_0_0".

        Returns:
            Dataset: A new Dataset object created from the DataFrame.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Reads the dataset and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data of the dataset as a DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Saves a pandas DataFrame as a dataset.

        Args:
            data (pd.DataFrame): The DataFrame to be saved in the dataset.

        Returns:
            bytes: The saved dataset data in bytes (CSV format).
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

    def __str__(self) -> str:
        """Returns a string representation of the Dataset.

        Returns:
            str: The name of the dataset.
        """
        return self.name
