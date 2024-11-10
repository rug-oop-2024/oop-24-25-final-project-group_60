import streamlit as st
import pickle
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Set the page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")


def write_helper_text(text: str):
    """Writes helper text to the Streamlit app with custom styling.

    Args:
        text (str): The text to display as a helper message.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# Page header and helper text
st.write("# 🚀 Deployment")
write_helper_text("In this section, you can predict your data using "
                  "your saved pipelines.")

# Get an instance of the AutoML system and retrieve pipelines
automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

# Dropdown to select a pipeline
pipeline_artifact = st.selectbox(
    'Select a pipeline:', pipelines,
    format_func=lambda pipeline: f"{pipeline.name}    v{pipeline.version}"
)

if pipeline_artifact is not None:
    # Load the selected pipeline
    pipeline = pickle.loads(pipeline_artifact.read())
    st.write(f"### Pipeline: {pipeline_artifact.name}")

    # Prepare the pipeline configuration for display
    config_dict = {
        "Model Type": pipeline._model.type,
        "Input Features": ', '.join(map(str, pipeline._input_features)),
        "Target Feature": str(pipeline._target_feature),
        "Data Split": pipeline._split,
        "Metrics": ', '.join([metric.name for metric in pipeline._metrics])
    }

    # Display pipeline configuration as a table
    config_df = pd.DataFrame(config_dict.items(),
                             columns=["Configuration", "Details"])
    st.table(config_df.set_index("Configuration"))

    # Section for uploading CSV files for prediction
    st.write("### Make predictions:")
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset = Dataset.from_dataframe(name=uploaded_file.name,
                                         asset_path=uploaded_file.name,
                                         data=df)

        # Button to start predictions
        if st.button("Predict"):
            st.session_state['predict_pipeline'] = True

        # If prediction button is clicked, make predictions
        if st.session_state.get('predict_pipeline', False):
            pipeline = pickle.loads(pipeline_artifact.read())

            try:
                prediction_name, predictions = pipeline.predict(dataset)
                st.subheader("Predictions")
                if predictions is not None:
                    df[prediction_name] = predictions
                    st.table(df.head())

                # Prepare download button for prediction results
                csv = df.to_csv(index=False).encode('utf-8')
                result_file_name = (
                            uploaded_file.name.replace('.csv', '_results.csv')
                            if uploaded_file.name.endswith('.csv')
                            else uploaded_file.name + '_results.csv'
                                   )

                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name=result_file_name,
                    mime='text/csv',
                )

            except ValueError:
                st.error("The dataset does not match the input features of the pipeline.")

    else:
        st.session_state['predict_pipeline'] = False