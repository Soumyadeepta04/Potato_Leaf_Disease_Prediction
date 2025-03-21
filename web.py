import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive model link (replace 'YOUR_FILE_ID' with actual ID)
drive_url = "https://drive.google.com/uc?id=1aJLIC4LvSLpoM16dgmoHELCzP0QD2Ol2"
model_path = "trained_plant_disease_model.keras"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    st.info("Downloading model, please wait...")
    gdown.download(drive_url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Function to load model and predict
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize the image
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Banner Image
img = Image.open('leafimg.jpg')
st.image(img, use_container_width=True)  # Updated parameter

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection System</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection')

# File Uploader
test_image = st.file_uploader('Choose an Image:', type=['jpg', 'png', 'jpeg'])

if test_image:
    st.image(test_image, use_container_width=True)

# Prediction Button
if st.button('Predict'):
    if test_image:
        st.snow()
        st.write('Analyzing Image...')
        result_index = model_prediction(test_image)
        class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
        st.success(f'Model predicts: **{class_name[result_index]}**')
    else:
        st.warning("Please upload an image before predicting.")
