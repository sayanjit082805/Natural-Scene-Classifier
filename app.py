import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("classifier.keras")

st.title("Natural Scene Classifier")

st.subheader(
    "Classify natural scenes as buildings, forests, mountains, glaciers, seas, or streets.",
    divider="gray",
)

st.write(
    """
   A deep learning project to classify natural scenes into six categories: buildings, forests, mountains, glaciers, seas, and streets. The model is trained on the Intel Image Classification dataset and uses a Convolutional Neural Network (CNN) architecture.     
"""
)

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


uploaded_file = st.file_uploader("Select an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((256, 256))
    img_rgb = img.convert("RGB")  
    img_array = np.array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Classify"):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        st.success(f"Prediction: **{classes[predicted_class].title()}**")

st.divider()
