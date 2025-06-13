import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Live Camera Anomaly Detection")

model = tf.keras.models.load_model("model/keras_model.h5")
camera = st.camera_input("Capture a photo")

if camera:
    img = Image.open(camera).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.error("Anomaly Detected from Live Camera!")
    else:
        st.success("Product is Normal.")
