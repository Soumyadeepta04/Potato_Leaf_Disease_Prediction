import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once and cache it to optimize performance
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to preprocess and predict image
def model_prediction(image_file):
    try:
        image = Image.open(image_file).convert('RGB')  # Convert image to RGB
        image = image.resize((128, 128))  # Resize to match model input size
        input_arr = np.array(image) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# Sidebar
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Banner Image
st.image("leafimg.jpg", use_column_width=True)

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Upload a Potato Leaf Image')

    # File Uploader
    test_image = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Prediction Button
    if st.button('Predict'):
        if test_image and model:
            with st.spinner('Processing...'):
                result_index = model_prediction(test_image)
                class_names = ['Potato Early Blight', 'Potato Late Blight', 'Healthy']
                
                if result_index is not None:
                    st.success(f'Model Prediction: {class_names[result_index]}')
                else:
                    st.error("Prediction failed. Please try again.")
        else:
            st.warning("Please upload an image before predicting.")
