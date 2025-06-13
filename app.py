# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# st.title("Anomaly Detection System")

# model = tf.keras.models.load_model("model/keras_model.h5")
# image = st.file_uploader("Upload Product Image", type=['jpg', 'png'])

# if image:
#     img = Image.open(image).convert("RGB").resize((224, 224))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     if prediction[0][0] > 0.5:
#         st.error("Anomaly Detected!")
#     else:
#         st.success("Product is Normal.")




import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Anomaly Detection System")

# Load model
model = tf.keras.models.load_model("model/keras_model.h5")

# Upload image
image = st.file_uploader("Upload Product Image", type=['jpg', 'png'])

if image:
    # Convert to RGB, resize and normalize
    img = Image.open(image).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]  # Assuming index 0 is "Anomaly"

    # Show prediction confidence
    st.write(f"📊 Confidence (Anomaly): **{confidence:.2f}**")

    # Apply threshold
    if confidence > 0.7:  # You can tweak this threshold if needed
        st.error("🚨 Anomaly Detected!")
    else:
        st.success("✅ Product is Normal.")
