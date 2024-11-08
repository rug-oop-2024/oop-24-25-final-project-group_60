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
model_select = None
metrics_select = None
metrics = []


model_select = st.multiselect('Select a model:', ["linearregressionmodel",
    "decisiontreeregressionmodel","randomforestregressionmodel",
    "multiplelinearregression", "logisticregressionmodel",
    "decisiontreeclassificationmodel","randomforestclassificationmodel"], 
    max_selections=1)
metric_select = st.multiselect('Select your metrics:', ["mean_squared_error",
    "accuracy", "mean_absolute_error", "root_mean_squared_error",
    "precision","recall"])

datasets = automl.registry.list(type="dataset")

if st.button("Detect Features", key="detect_features"):
    for idx, dataset in enumerate(datasets):  # Enumerate to get unique index for each dataset
        dataset.__class__ = Dataset
        features = detect_feature_types(dataset)  # Assuming this function returns a list of features

        # Dynamically create keys based on dataset index
        input_features_key = f"input_features_{idx}"
        target_feature_key = f"target_feature_{idx}"

        # Feature selection widgets with dynamic keys
        input_features = st.multiselect("Select Input Features", features, key=input_features_key)
        target_feature = st.selectbox("Select Target Feature", features, key=target_feature_key)

        if input_features and target_feature:
            st.write(f"Selected Input Features: {input_features}")
            st.write(f"Selected Target Feature: {target_feature}")

# Modelling section
if st.button("Start Modelling"):
    model_selected = model_select is not None and model_select != []
    input_features_selected = input_features is not None and len(input_features) > 0
    target_selected = target_feature is not None and target_feature != []

    if model_selected and input_features_selected and target_selected:
        model = get_model(''.join(model_select))
        metrics = [get_metric(metric) for metric in metric_select or []]

        for dataset in datasets:
            train_data, test_data = dataset.split(0.8)  # Adjust split as needed

            # Initialize and execute the pipeline
            dataset_pipeline = Pipeline(
                metrics=metrics,
                dataset=dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                train_data=train_data,
                test_data=test_data
            )
            st.write(dataset_pipeline.execute())
    else:
        if not model_selected:
            st.write("Please select a model before starting the modelling process.")
        if not input_features_selected:
            st.write("Please select at least one input feature.")
        if not target_selected:
            st.write("Please select a target feature.")