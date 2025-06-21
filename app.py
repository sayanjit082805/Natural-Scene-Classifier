import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Natural Scene Classifier",
    page_icon="🌄",
    initial_sidebar_state="expanded",
)

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

st.sidebar.header("Dataset Information", divider="gray")

st.sidebar.markdown(
    """ 
    The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). It contains around 25,000 images of natural scenes from around the world.

**Statistics** :

- **Total Images**: ~25,000
- **Training Set**: ~14,000 images
- **Test Set**: ~3,000 images
- **Validation Set**: ~7,000 images
- **Classes**: 6 (Buildings, Forest, Glacier, Mountain, Sea, Street)
"""
)


st.sidebar.header("Model Metrics", divider="gray")

st.sidebar.markdown(
    """ 
The model achieves an overall accuracy of ~83%.

Due to the use of Early fitting, the model convergences at 66 epochs.
"""
)


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
