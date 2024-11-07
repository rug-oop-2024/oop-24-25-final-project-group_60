import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()
metrics = []
datasets = automl.registry.list(type="dataset")


dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox('Select a dataset:', dataset_names)
dataset_select = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
dataset_select.__class__ = Dataset
columns = [feature.name for feature in detect_feature_types(dataset_select)]
input_features = st.multiselect(
    label="Select input features",
    options=columns,
    default=columns[:-1],
    help="Select one or more features to be used as input features for the model."
)

target_feature = st.selectbox(
    label="Select target feature",
    options=columns,
    index=len(columns) - 1,  # Default to the last column
    help="Select a single feature to be used as the target for the model."
)


model_select = st.selectbox('Select a model:', ["linearregressionmodel",
    "decisiontreeregressionmodel","randomforestregressionmodel",
    "multiplelinearregression", "logisticregressionmodel",
    "decisiontreeclassificationmodel","randomforestclassificationmodel"])
metric_select = st.multiselect('Select your metrics:', ["mean_squared_error",
    "accuracy", "mean_absolute_error", "root_mean_squared_error",
    "precision","recall"])

datasets = automl.registry.list(type="dataset")

if st.button("Start Modelling") and model_select and metric_select:
    print(model_select, metric_select) 
    print(''.join(model_select))
    model = get_model(''.join(model_select))
    print(model.name)
    print(model.type)
    for metric in metric_select:
        metrics.append(get_metric(metric))
    print(metrics)
    for dataset in datasets:
        dataset.__class__ = Dataset
        features = detect_feature_types(dataset)
        input_features = features[:-1]
        target_feature = features[-1]
        dataset_pipeline = Pipeline(metrics=metrics, dataset=dataset, model=model, 
                                    input_features=input_features, 
                                    target_feature=target_feature)
        st.write(dataset_pipeline.execute())
else:
    st.write("Fill in both your wanted model and metrics before you start modelling")
