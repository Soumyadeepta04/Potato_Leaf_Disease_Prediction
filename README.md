# 🥔 Potato Leaf Disease Detection System

## 📌 Overview
The **Potato Leaf Disease Detection System** is a deep learning-based application that detects various potato leaf diseases using image processing techniques. The system utilizes **TensorFlow**, **Keras**, and **OpenCV** to classify leaf images and provide insights into potential diseases.

## 🛠️ Technologies Used
- **Python** 🐍
- **TensorFlow & Keras** (for deep learning model training)
- **OpenCV** (for image processing)
- **Matplotlib** (for visualization)
- **streamlit** (for web application)

## ⚡ Features
✔ Detects different types of potato leaf diseases
✔ Image-based classification using CNN model
✔ User-friendly web interface
✔ Provides real-time disease prediction

## 🚀 How to Run
### 1️⃣ Clone the repository
```bash
git clone https://github.com/Soumyadeepta04/potato-leaf-detection.git
cd potato-leaf-detection
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model (Optional if model is pre-trained)
```bash
python src/train.py
```

### 4️⃣ Run the web application
```bash
python potato_leaf_detection/web.py
```

## 📊 How It Works
1️⃣ User uploads an image of a potato leaf.
2️⃣ The model processes the image and classifies the disease.
3️⃣ The result is displayed on the UI with the disease type and confidence score.

## 🏗 Future Improvements
- Improve model accuracy using a larger dataset
- Deploy the application on cloud platforms (AWS/GCP)
- Add more plant disease classifications



