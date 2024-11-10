import streamlit as st
import pickle
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import get_model, REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.metric import get_metric, METRICS
from autoop.functional.feature import detect_feature_types

st.set_page_config(
    page_title="Deployment",
    page_icon="🚀",
)

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# 🚀 Deployment")
# keep the helper text short
write_helper_text("In this section, you can predict your data using your saved pipelines.")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

# select a pipeline
pipeline_select = st.selectbox('Select a pipeline:', pipelines, format_func=lambda pipeline: f"{pipeline.name}    v{pipeline.version}")

if pipeline_select is not None:
    pipeline_select.__class__ = Pipeline
    st.write(f"### Pipeline: {pipeline_select.name}")
    st.write(f"### Make predictions:")
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        dataset = Dataset.from_dataframe(name=uploaded_file.name, asset_path=uploaded_file.name, data=df)

        if st.button("Predict"):
            predictions = pipeline_select.predict(dataset)
            st.write(predictions)