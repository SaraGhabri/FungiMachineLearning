# 🍄 Fungal Leaf Disease Detection – Streamlit App

## 📌 Project Description

This web application allows users (farmers, agronomists, or general users) to upload images of plant leaves to detect fungal diseases using a trained deep learning model.

### 🔧 Built With:
- Python
- TensorFlow / Keras
- Streamlit
- MobileNetV2 for image classification

---

## ⚙️ Step-by-Step Setup

### 1️⃣ Prerequisites
Make sure the following are installed:
- Python ≥ 3.8  
- Anaconda or Miniconda (recommended)  
- Git  
- VS Code  

### 2️⃣ Create and Activate Virtual Environment
```bash
conda create -n streamlit_fungi python=3.8
conda activate streamlit_fungi
pip install streamlit tensorflow pillow numpy
```
### 3️⃣ Project Structure
Organize your project folder (e.g., `FungiFront`) like this:

```
FungiFront/
│
├── app.py                        # Streamlit app
├── mobilenetv2_fungal_disease.h5 # Trained model file
├── requirements.txt              # Dependency list
├── README.md                     # Project description
```

To generate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

### 4️⃣ Streamlit App (app.py)
```python
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("mobilenetv2_fungal_disease.h5")

class_names = ['Apple Scab', 'Apple Black Rot', 'Grape Black Rot', 'Healthy', 'Cedar Rust']

st.title("🍂 Fungal Disease Detector")
st.write("Upload a leaf image and get a prediction.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.success(f"Prediction: {class_names[class_idx]}")
```

---

## 🧠 Why MobileNetV2?

You absolutely worked with a **Convolutional Neural Network (CNN)** in this project.

Even though you used **MobileNetV2**, which is a pre-trained model from TensorFlow/Keras, it is fundamentally a deep CNN architecture. Specifically:

- **MobileNetV2** is a lightweight CNN designed for mobile and embedded vision applications.
- It uses **depthwise separable convolutions**, **inverted residual blocks**, and **linear bottlenecks** to reduce computational cost while maintaining performance.

In your case, you likely used **transfer learning**, meaning you took this CNN's pre-trained convolutional base and either:
- Fine-tuned its last layers on your fungal dataset, **or**
- Froze the base and added custom fully connected layers for classification.

### ❓Other Models Considered

| Model         | Why not chosen                             |
|---------------|---------------------------------------------|
| VGG16/VGG19   | Large size, slow, outdated                 |
| ResNet50      | Good accuracy, but slower and heavier      |
| DenseNet121   | Good accuracy, but bulkier than MobileNetV2 |
| EfficientNet  | More accurate but more tuning/time needed  |

> ⚠️ EfficientNetB0 or DenseNet are good options with more data/GPU.

---

---


## 🧪 Dataset Info

- Source: [Fungal leaf disease dataset (Mendeley)](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- Classes: 5  
- Image size: 224x224  
- Preprocessing: Normalization to [0, 1]  
- Training: Transfer learning (MobileNetV2 fine-tuned)

---

## 🚀 Run the App
```bash
conda activate streamlit_fungi
streamlit run app.py
```

---

## 🧑‍💻 GitHub Setup

### Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit - fungal disease detector"
```

### Push to GitHub
```bash
git remote add origin https://github.com/your-username/fungi-disease-streamlit.git
git branch -M main
git push -u origin main
```
