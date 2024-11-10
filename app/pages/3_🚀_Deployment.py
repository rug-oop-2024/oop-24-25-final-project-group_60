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
pipeline_artifact = st.selectbox('Select a pipeline:', pipelines, format_func=lambda pipeline: f"{pipeline.name}    v{pipeline.version}")

if pipeline_artifact is not None:
    st.write(f"### Pipeline: {pipeline_artifact.name}")
    st.write(f"### Make predictions:")
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        dataset = Dataset.from_dataframe(name=uploaded_file.name, asset_path=uploaded_file.name, data=df)

        if st.button("Predict"):
            st.session_state['predict_pipeline'] = True
        
        if st.session_state.get('predict_pipeline', False):
            pipeline = pickle.loads(pipeline_artifact.read())
            prediction_name, predictions = pipeline.predict(dataset)
            st.subheader("Predictions")
            if predictions is not None:
                df[prediction_name] = predictions
                st.table(df.head())

                csv = df.to_csv(index=False).encode('utf-8')

                result_file_name = uploaded_file.name.replace('.csv', '_results.csv') if uploaded_file.name.endswith('.csv') else uploaded_file.name + '_results.csv'
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name=result_file_name,
                    mime='text/csv',
                )
