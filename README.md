🥔 Potato Leaf Disease Detection
📌 Overview
This project is a Machine Learning-based Potato Leaf Disease Detection System designed to identify and classify diseases in potato plants based on leaf images. It utilizes Deep Learning models for disease prediction and features an interactive Streamlit-based web application for user-friendly access.

📁 Project Structure
sql
Copy
Edit
potato_disease_detection/  
│── datasets/  
│   ├── potato_disease_dataset/  
│── models/  
│   ├── potato_leaf_model.h5  
│── app.py  
│── requirements.txt  
│── README.md  
│── utils.py  
│── static/ (Contains images and UI assets)  
🛠️ Technologies Used
Python 🐍
TensorFlow/Keras (Deep Learning Model)
OpenCV (Image Processing)
Streamlit (Interactive Web Interface)
Matplotlib & NumPy (Data Processing & Visualization)
⚡ Features
✔ Detects different types of potato leaf diseases
✔ User-friendly Streamlit-based UI
✔ Model trained on real-world agricultural datasets
✔ Provides fast and accurate predictions

🚀 How to Run
1️⃣ Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/potato-disease-detection.git  
cd potato-disease-detection  
2️⃣ Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt  
3️⃣ Run the application:

bash
Copy
Edit
streamlit run app.py  
4️⃣ Upload an image of a potato leaf and get predictions!

📊 How It Works
User uploads a potato leaf image 🖼️
Pre-trained CNN model processes the image
System predicts whether the leaf is healthy or diseased
Results are displayed on the UI with insights
🏗 Future Improvements
Expand dataset for improved accuracy 📈
Integrate real-time leaf scanning using mobile cameras 📱
Implement a multi-crop disease detection system 🌱
Deploy on cloud platforms for global access ☁️
