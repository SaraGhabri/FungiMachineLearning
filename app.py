import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the trained model
model = load_model("mobilenetv2_fungal_disease.h5")

st.title("Fungal Leaf Disease Classifier")
st.write("Upload a leaf image to detect the fungal disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image to match model input
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        if img_array.shape[-1] == 4:  # Convert RGBA to RGB
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Example labels (customize to your dataset)
        class_names = ["Apple Scab", "Cercospora Leaf Spot", "Healthy", "Powdery Mildew"]
        st.success(f"Prediction: {class_names[predicted_class]}")
    else:
        st.error("Invalid file type. Please upload a .jpg, .jpeg, or .png file.")