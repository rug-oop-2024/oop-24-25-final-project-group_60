import streamlit as st
import pandas as pd
from time import sleep

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Datasets", page_icon="📊")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# 📊 Datasets")
write_helper_text("In this section, you can add your own datasets to be modelled.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
st.write(automl.registry.list())

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset = Dataset.from_dataframe(name="uploaded_dataset", asset_path=uploaded_file.name, data=df)

    st.write(dataset.read().head())


    if st.button('Save dataset'):
        automl.registry.register(dataset)
        st.write('Dataset saved successfully')
        
