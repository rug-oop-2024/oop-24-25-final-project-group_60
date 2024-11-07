import streamlit as st
import pandas as pd

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


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset = Dataset.from_dataframe(name=uploaded_file.name, asset_path=uploaded_file.name, data=df)

    st.write(dataset.read().head())

    if st.button('Save dataset'):
        automl.registry.register(dataset)
        st.success(f"Dataset {dataset.name} registered successfully.")

st.write("#### Registered Datasets")

datasets = automl.registry.list(type="dataset")
datasets_checkboxes = []
if len(datasets) == 0:
    st.write("No datasets registered yet.")
else:
    for dataset in datasets:
        dataset_checkbox = st.checkbox(f"**{dataset.name}**")
        datasets_checkboxes.append(dataset_checkbox)

    if st.button("Delete selected datasets"):
        for dataset, checkbox in zip(datasets, datasets_checkboxes):
            if checkbox:
                automl.registry.delete(dataset.id)
        st.success("Selected datasets deleted successfully.")


