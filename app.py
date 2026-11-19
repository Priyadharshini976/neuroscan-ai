
import streamlit as st
import torch
import os
import numpy as np
from PIL import Image

from common import load_trained_model, DEVICE
from explain import predict_and_explain, METHODS

st.set_page_config(
    page_title="NeuroScan AI",
    layout="wide"
)

st.title("🧠 NeuroScan AI")
st.subheader("Explainable Stroke Detection Using Brain MRI")

st.write(
    "Upload a brain MRI image to detect and classify "
    "stroke using our trained ResNet50 model."
)

MODEL_PATH = "models/best_model.pt"


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI Image")
    st.image(image, width=350)

    method = st.selectbox(
        "Select Explainability Method",
        list(METHODS.keys())
    )

    if st.button("Analyse MRI Image", type="primary"):

        if not os.path.exists(MODEL_PATH):
            st.error("Trained model not found!")
            st.stop()

        try:
            with st.spinner("Analysing MRI image..."):

                model, classes = get_model()

                probs, pred, overlay, cam_map = (
                    predict_and_explain(
                        model,
                        image,
                        method=method
                    )
                )

            st.success("Analysis completed!")

            st.subheader("Stroke Classification Result")

            predicted_class = classes[pred]
            confidence = float(probs[pred]) * 100

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Class-wise Prediction")

            for i, class_name in enumerate(classes):
                st.write(
                    f"{class_name}: {probs[i] * 100:.2f}%"
                )

                st.progress(float(probs[i]))

            st.divider()

            st.subheader("Explainable AI - Grad-CAM")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Original MRI")
                st.image(image, use_container_width=True)

            with col2:
                st.write(method + " Visualization")
                st.image(
                    overlay,
                    caption="Highlighted model attention regions",
                    use_container_width=True
                )

            st.info(
                "The heatmap highlights image regions "
                "that influenced the model prediction. "
                "It does not confirm a medical lesion."
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")

st.divider()

st.caption(
    "NeuroScan AI | Academic Research Prototype | "
    "Not intended for clinical diagnosis."
)
