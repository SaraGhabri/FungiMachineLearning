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
