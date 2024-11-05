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

model = get_model("logisticregressionmodel")
metrics = [get_metric("accuracy")]



datasets = automl.registry.list(type="dataset")
for dataset in datasets:
    dataset.__class__ = Dataset
    features = detect_feature_types(dataset)
    input_features = features[:-1]
    target_feature = features[-1]
    dataset_pipeline = Pipeline(metrics=metrics, dataset=dataset, model=model, 
                                input_features=input_features, 
                                target_feature=target_feature)
    print(dataset_pipeline.execute())

# your code here
