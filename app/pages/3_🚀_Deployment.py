import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
import pickle

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
pipeline_select = st.selectbox('Select a pipeline:', pipelines, format_func=lambda pipeline: pipeline.name)

if st.button('Show Pipeline: '):
    if pipeline_select is not None:
        st.subheader("Pipeline Configuration:")
        data = pickle.load(pipeline_select.data)
        st.write(data)
    else:
        st.write("Make sure you saved a pipeline on the modelling page and selected it above.") 