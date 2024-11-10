import streamlit as st

from app.core.system import AutoMLSystem

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