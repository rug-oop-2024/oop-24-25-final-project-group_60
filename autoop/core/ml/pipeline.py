from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
from autoop.functional.feature import detect_feature_types
import numpy as np


class Pipeline():
    """A class to define and execute a machine learning pipeline.

    The pipeline is responsible for preprocessing data, splitting the dataset,
    training the model, and evaluating it using specified metrics.

    Args:
        metrics (List[Metric]): List of metrics to evaluate the model.
        dataset (Dataset): Dataset object containing the data.
        model (Model): The model to be used in the pipeline.
        input_features (List[Feature]): List of input features for the model.
        target_feature (Feature): The target feature for the model.
        split (float): The proportion of the dataset used for training
                       (default is 0.8).
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ):
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and \
           model.type != "classification":
            raise ValueError("Model type must be classification for \
                              categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous \
                              target feature")

    def __str__(self):
        """Return a string representation of the pipeline.

        Returns:
            str: The string representation of the pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self):
        """Returns the model used in the pipeline.

        Returns:
            Model: The model object.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Returns the artifacts generated during the pipeline execution.

        This includes models, encoders, scalers, and configuration data.

        Returns:
            List[Artifact]: List of artifact objects generated by the pipeline.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """Registers an artifact in the pipeline.

        Args:
            name (str): The name of the artifact.
            artifact (any): The artifact object to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """Preprocess the input features and target feature.

        This method transforms the features into a format suitable for
        training the model.
        It also registers the corresponding artifacts.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sorted by feature name
        # for consistency
        self._output_vector = np.argmax(target_data, axis=1)
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self):
        """Split the data into training and testing sets.
        """
        split = self._split
        split = int(split * len(self._output_vector))

        if split == 0:
            split = 1
        elif split == len(self._output_vector):
            split -= 1

        self._train_X = [vector[:split] for vector in
                         self._input_vectors]
        self._test_X = [vector[split:] for vector in
                        self._input_vectors]
        self._train_y = self._output_vector[
            :int(split)]
        self._test_y = self._output_vector[
            int(split):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenate the feature vectors into a single array.

        Args:
            vectors (List[np.array]): The list of feature vectors to
                                      concatenate.

        Returns:
            np.array: The concatenated feature matrix.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """Train the model on the training dataset.

        This method uses the training data to fit the model.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate_on_training_set(self):
        """Evaluate the model on the training dataset.

        This method calculates performance metrics on the training data and
        stores the results.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._trainmetrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._trainmetrics_results.append((metric.name, result))
        self._trainpredictions = predictions

    def _evaluate(self):
        """Evaluate the model on the test dataset.

        This method calculates performance metrics on the test data and stores
        the results.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric.name, result))
        self._predictions = predictions

    def execute(self):
        """Execute the entire pipeline (preprocessing, training, evaluation).

        Returns:
            dict: A dictionary containing the evaluation results, including
                  training metrics and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_on_training_set()
        return {
            "trainmetrics": self._trainmetrics_results,
            "trainpredictions": self._trainpredictions,
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    def predict(self, dataset: Dataset):
        """Make predictions on a new dataset using the trained model.

        Args:
            dataset (Dataset): The new dataset for prediction.

        Returns:
            tuple: A tuple containing the name of the target feature and the
                   predictions.
        """
        features = detect_feature_types(dataset)
        if features != self._input_features:
            raise ValueError

        input_results = preprocess_features(features, dataset)
        input_vectors = [data for (feature_name, data, artifact)
                         in input_results]
        X = self._compact_vectors(input_vectors)
        predictions = self._model.predict(X)

        return self._target_feature.name, predictions
