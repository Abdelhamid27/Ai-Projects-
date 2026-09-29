import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Ship Classifier",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Ship Classification Web App")
st.write("Upload an image of a ship to predict its category and model confidence.")

# Class dictionary mapped to the dataset labels
CLASSES = {
    1: 'Cargo',
    2: 'Military',
    3: 'Carrier',
    4: 'Cruise',
    5: 'Tankers'
}

MODEL_PATH = 'best_ship_model.keras'

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Verify if the model file exists in the directory
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' was not found. Please ensure it is placed in the project folder.")
    st.stop()

model = load_trained_model()

# Image file uploader widget
uploaded_file = st.file_uploader("Choose a ship image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image to match training input (224x224, normalized to [0, 1])
    target_size = (224, 224)
    img_resized = image.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction button
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array)[0]
            
            # Categories 1 to 5 correspond to indices 0 to 4
            best_idx = int(np.argmax(predictions))
            predicted_class_num = best_idx + 1
            predicted_label = CLASSES[predicted_class_num]
            confidence = float(predictions[best_idx]) * 100

            st.success(f"**Predicted Category:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Display probabilities breakdown
            st.write("---")
            st.subheader("Class Probabilities:")
            for idx, prob in enumerate(predictions):
                c_num = idx + 1
                st.write(f"**{CLASSES[c_num]}**: {prob * 100:.2f}%")
                st.progress(float(prob))