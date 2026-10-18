import sys
from pathlib import Path

# Ensure project root is in Python path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

from src.predict import (
    load_model_and_classes,
    predict_image_bytes,
)


# Page configuration

st.set_page_config(
    page_title="BrainScan",
    page_icon="(-_-)",
    layout="centered",
)


# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_brainscan.keras"
CLASSES_PATH = PROJECT_DIR / "models" / "class_names.json"


# Load model only once
@st.cache_resource
def get_model():
    """
    Load and cache the model.

    Streamlit reruns the script after every user interaction.
    Caching prevents the model from being loaded repeatedly.
    """
    return load_model_and_classes(
        MODEL_PATH,
        CLASSES_PATH
    )


# Interface
st.title("BrainScan")

st.write(
    "Upload a brain MRI image to obtain the model's prediction."
)



# Load the model and handle loading errors
try:
    model, class_names = get_model()

except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
)


if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    # Display the original uploaded image
    st.image(
        image_bytes,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    if st.button("Analyze image", type="primary"):
        try:
            with st.spinner("Analyzing the image..."):
                result = predict_image_bytes(
                    image_bytes,
                    model,
                    class_names
                )

            st.success("Analysis completed")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted class",
                    value=result["class_name"]
                )

            with col2:
                st.metric(
                    label="Model confidence",
                    value=f"{result['confidence']:.2%}"
                )

            st.subheader("Probabilities")

            # Sort classes from highest to lowest probability
            sorted_probabilities = sorted(
                result["probabilities"].items(),
                key=lambda item: item[1],
                reverse=True
            )

            for class_name, probability in sorted_probabilities:
                st.write(
                    f"**{class_name}: {probability:.2%}**"
                )
                st.progress(float(probability))

            st.info(
                "Confidence is the probability produced by the model. "
                "It is not a guarantee that the prediction is correct."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")


st.divider()

st.caption(
    "BrainScan => CNN educational project | "
    "Always consult a qualified medical professional..."
)