import os
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# --- Custom Layer Definition ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove unsupported 'groups' arg if present
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# --- Cached Functions ---
@st.cache_resource
def load_bird_model(model_path: str):
    """Load a Keras model with custom objects."""
    return load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

@st.cache_resource
def load_class_names(train_dir: str):
    """Scan training directory for class subfolders."""
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

# --- Default Paths ---
MODEL_PATH = "best_model_final.h5"
TRAIN_DIR = "10/Train"

# --- Streamlit App Setup ---
st.set_page_config(page_title="Bird Species Classifier", layout="centered")

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Classification"])

# --- Main Page Content ---
if app_mode == "Home":
    st.title("Welcome to Bird Species Classifier")
    st.markdown("This app allows you to classify bird species from images using a deep learning model.")
    st.markdown("Navigate to the 'Classification' page to upload an image and get predictions.")

elif app_mode == "About":
    st.title("About")
    st.markdown("This application uses a convolutional neural network model to classify bird species.")
    st.markdown("The model is trained on a dataset of 200 bird species.")
    st.markdown("The default model is used for predictions.")

elif app_mode == "Classification":
    st.title("🐦 Bird Species Classification")
    st.markdown("Upload a bird image below to predict the species.")

    # Load the model
    try:
        model = load_bird_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from '{MODEL_PATH}'. Error: {str(e)}")
        st.stop()

    # Load class labels
    try:
        class_names = load_class_names(TRAIN_DIR)
    except Exception:
        st.error(f"Could not load class names from '{TRAIN_DIR}'. Check that the directory exists.")
        st.stop()

    # Prediction function
    def predict_bird(img: Image.Image):
        """Predict the bird species from an image."""
        img = img.resize((224, 224))  # Resize to model input size
        arr = np.array(img)
        # Handle RGBA by dropping alpha channel
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype('float32') / 255.0  # Normalize to [0,1]
        input_arr = np.expand_dims(arr, axis=0)  # Add batch dimension
        preds = model.predict(input_arr)[0]  # Get predictions
        top_ind = np.argmax(preds)  # Index of highest probability
        label = class_names[top_ind].replace('_', ' ')  # Map to species name
        prob = preds[top_ind]  # Confidence score
        return label, prob

    # Image upload and prediction
    img_file = st.file_uploader("Upload an image of a bird (jpg/png)", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                label, prob = predict_bird(img)
            st.success("Done!")
            st.subheader("Predicted Bird Species:")
            st.write(f"**{label}** ")

if __name__ == "__main__":
    pass



# hello