import streamlit as st
import pandas as pd
import pickle
from time import sleep

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import get_model, REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.metric import get_metric, METRICS
from autoop.functional.feature import detect_feature_types
import re


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()
metrics = []
model_options = REGRESSION_MODELS + CLASSIFICATION_MODELS
datasets = automl.registry.list(type="dataset")

dataset_select = st.selectbox('Select a dataset:', datasets, format_func=lambda dataset: dataset.name)
if dataset_select is not None:
    print(dataset_select)
    dataset_select.__class__ = Dataset
    columns = detect_feature_types(dataset_select)

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

    if target_feature.type == "categorical":
        model_options = CLASSIFICATION_MODELS

    elif target_feature.type == "numerical":
        model_options = REGRESSION_MODELS

split = st.slider("Select your trainings-split:", min_value=0.0, max_value=1.0,
                  value=0.8)
model_select = st.selectbox('Select a model:', model_options)
metric_select = st.multiselect('Select your metrics:', ["mean_squared_error",
    "accuracy", "mean_absolute_error", "root_mean_squared_error",
    "precision","recall"], default=["accuracy"])

#datasets = automl.registry.list(type="dataset")


if model_select and metric_select and dataset_select:  

    # Model the pipeline
    if  st.button("Start Modelling"):
        st.session_state['start_modelling'] = True

    if st.session_state.get('start_modelling', False):  
        model = get_model(''.join(model_select))
        for metric in metric_select:
            metrics.append(get_metric(metric))

        dataset_select.__class__ = Dataset

        features = detect_feature_types(dataset_select)
        input_features = features[:-1]
        target_feature = features[-1]

        dataset_pipeline = Pipeline(metrics=metrics, dataset=dataset_select, 
                                    model=model, input_features=input_features, 
                                    target_feature=target_feature, 
                                    split = split)
        results = dataset_pipeline.execute()

        # Print the train metrics
        st.subheader("Train Metrics:")
        if results["trainmetrics"] is not None:
            train_metrics_df = pd.DataFrame(results["trainmetrics"], columns=["Metric", "Value"])
            st.table(train_metrics_df)

        # Print the train predictions
        st.subheader("Train Predictions:")
        if results["trainpredictions"] is not None:
            train_predictions_df = pd.DataFrame(results["trainpredictions"], columns=[f"{target_feature}"])
            st.table(train_predictions_df.head())

        # Print the evaluation metrics
        st.subheader("Metrics:")
        if results["metrics"] is not None:
            metrics_df = pd.DataFrame(results["metrics"], columns=["Metric", "Value"])
            st.table(metrics_df)

        # Print the predictions as a table
        st.subheader("Predictions:")
        if results["predictions"] is not None:
            predictions_df = pd.DataFrame(results["predictions"], columns=[f"{target_feature}"])
            st.table(predictions_df.head())

        st.write("### Save configuration:")
        # Save the Pipeline
        name = st.text_input('Enter a name for the pipeline:', 'my_pipeline')
        version = st.text_input('Enter a version for the pipeline:', '1_0_0')
        if st.button("Save Pipeline"):
            # Assert that the version is structured as digit_digit_digit
            if re.match(r'^\d+_\d+_\d+$', version):
                if  (name and version) != "":

                    pipeline_artifact = Artifact(name=name,
                                                    version=version,
                                                    asset_path=name,
                                                    data=pickle.dumps(dataset_pipeline),
                                                    type="pipeline")
                    
                    automl.registry.register(pipeline_artifact)
                    st.success(f"Pipeline {name} saved successfully")
                else:
                    st.write("Fill in a name and version if you want to save your pipeline.")

            else:
                st.error("Version must be structured as digit_digit_digit (e.g., 1_0_0)")
else:
    st.session_state['start_modelling'] = False