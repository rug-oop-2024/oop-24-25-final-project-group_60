
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import DecisionTreeClassificationModel
from autoop.core.ml.model.classification  import RandomForestClassificationModel
from autoop.core.ml.model.classification  import LogisticRegressionModel
from autoop.core.ml.model.regression  import DecisionTreeRegressionModel
from autoop.core.ml.model.regression import RandomForestRegressionModel
from autoop.core.ml.model.regression import LinearRegressionModel
from autoop.core.ml.model.regression import MultipleLinearRegression

# List of available regression models
REGRESSION_MODELS = [
    "linearregressionmodel",
    "decisiontreeregressionmodel",
    "randomforestregressionmodel",
    "multiplelinearregression"
]

# List of available classification models
CLASSIFICATION_MODELS = [
    "logisticregressionmodel",
    "decisiontreeclassificationmodel",
    "randomforestclassificationmodel"
] 

def get_model(model_name: str) -> Model:
    """Returns the model class based on the model name.

    Args:
        model_name (str): The name of the model to retrieve. This name is case-insensitive.

    Returns:
        Model: A model object that matches the requested model name.

    Raises:
        ValueError: If the model name is not implemented.
    """
    match model_name.lower():
        case "logisticregressionmodel":
            return LogisticRegressionModel()
        case "decisiontreeclassificationmodel":
            return DecisionTreeClassificationModel()
        case "randomforestclassificationmodel":
            return RandomForestClassificationModel()
        case "linearregressionmodel":
            return LinearRegressionModel()
        case "decisiontreeregressionmodel":
            return DecisionTreeRegressionModel()
        case "randomforestregressionmodel":
            return RandomForestRegressionModel()
        case "multiplelinearregression":
            return MultipleLinearRegression()
        case _:
            raise ValueError(f"Model '{model_name}' is not implemented.")
        