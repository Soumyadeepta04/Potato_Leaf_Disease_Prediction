import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load model and predict
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
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
st.image(img, use_column_width=True)  # Fixed width issue

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System for Sustainable Agriculture')

# File Uploader
test_image = st.file_uploader('Choose an Image: ')

if test_image:  # Only show image if uploaded
    st.image(test_image, use_column_width=True)

# Prediction Button
if st.button('Predict'):
    if test_image:  # Ensure an image is uploaded before predicting
        st.snow()
        st.write('Our Prediction:')
        result_index = model_prediction(test_image)
        class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
        st.success(f'Model predicts: {class_name[result_index]}')
    else:
        st.warning("Please upload an image before predicting.")
