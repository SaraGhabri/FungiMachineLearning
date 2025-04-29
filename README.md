# 🍄 Fungal Leaf Disease Detection – Streamlit App

## 📌 Project Description

This web application allows users (farmers, agronomists, or general users) to upload images of plant leaves to detect fungal diseases using a trained deep learning model.

### 🔧 Built With:
- Python
- TensorFlow / Keras
- Streamlit
- MobileNetV2 for image classification

This app allows users to detect fungal diseases in plant leaves by uploading images. It uses a MobileNetV2 deep learning model fine-tuned on a fungal leaf dataset.

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
