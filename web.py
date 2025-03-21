import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Path to the model file (now tracked by Git LFS)
MODEL_PATH = "trained_plant_disease_model.keras"

# Debug: Print current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to make predictions
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to array
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Banner Image (Ensuring uniform width)
img = Image.open('leafimg.jpg')
st.image(img, use_container_width=True)

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection System</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection')

    # File Uploader
    test_image = st.file_uploader('Choose an Image:')

    if test_image:
        st.image(test_image, use_container_width=True)

    # Prediction Button
    if st.button('Predict'):
        if test_image:
            st.snow()
            st.write('Our Prediction:')
            result_index = model_prediction(test_image)
            class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
            st.success(f'Model predicts: {class_name[result_index]}')
        else:
            st.warning("Please upload an image before predicting.")
