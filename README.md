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

## 👨‍🔬 Model

We used MobileNetV2 for its speed and efficiency. It's trained to recognize:

- Apple Scab
- Apple Black Rot
- Grape Black Rot
- Cedar Rust
- Healthy leaves

## 🚀 Usage

1. Clone the repo
2. Install dependencies from `requirements.txt` : pip install streamlit tensorflow pillow
3. Run the app with:

```bash
conda activate streamlite_fungi
streamlit run app.py
