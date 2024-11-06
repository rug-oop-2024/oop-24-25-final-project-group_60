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
metrics = []
metrics_select = None

model_select = st.multiselect('Select a model:', ["linearregressionmodel",
    "decisiontreeregressionmodel","randomforestregressionmodel",
    "multiplelinearregression", "logisticregressionmodel",
    "decisiontreeclassificationmodel","randomforestclassificationmodel"], 
    max_selections=1)
metric_select = st.multiselect('Select your metrics:', ["mean_squared_error",
    "accuracy", "mean_absolute_error", "root_mean_squared_error",
    "precision","recall"])

datasets = automl.registry.list(type="dataset")

if st.button("Start Modelling") and (model_select and metric_select) is not None: 
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
