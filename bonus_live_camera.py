# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# st.set_page_config(page_title="Anomaly Detection", layout="centered")
# st.title("🧪 Anomaly Detection: Live Camera or Image Upload")

# # Load the trained model
# model = tf.keras.models.load_model("model/keras_model.h5")

# # Choose input method
# option = st.radio("Choose input method:", ("Live Camera", "Upload Image"))

# image = None

# if option == "Live Camera":
#     camera = st.camera_input("📸 Capture an image")
#     if camera:
#         image = Image.open(camera)

# elif option == "Upload Image":
#     uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)

# # Prediction section
# if image:
#     # Preprocess the image
#     img = image.resize((224, 224)).convert('RGB')
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)
#     score = prediction[0][0]
#     st.write(f"🧪 Prediction Score: `{score:.4f}`")

#     # Set your threshold (adjust if needed)
#     THRESHOLD = 0.3

#     # Final decision
#     if score > THRESHOLD:
#         st.error("🔴 Product is **Defective**.")
#     else:
#         st.success("✅ Product is **Normal**.")

# ------------------------------------------------------------------------------------------




# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Page settings
# st.set_page_config(page_title="🧪 Anomaly Detection", layout="centered")
# st.markdown("""
#     <style>
#         .main { background-color: #f9f9f9; }
#         .title { font-size: 36px; font-weight: 700; color: #4B8BBE; }
#         .subtitle { font-size: 18px; color: #666; margin-top: -10px; }
#         .stButton>button { background-color: #4B8BBE; color: white; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("<div class='title'>🧪 Anomaly Detection</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Detect whether a product is Normal or Defective using image input</div>", unsafe_allow_html=True)

# # Load model
# model = tf.keras.models.load_model("model/keras_model.h5")

# # Choose input method
# st.markdown("### 🔍 Select Input Method")
# option = st.radio("", ("📸 Live Camera", "📂 Upload Image", "🗂️ Upload Multiple Images"))

# image = None

# # --- Live Camera
# if option == "📸 Live Camera":
#     camera = st.camera_input("Take a photo")
#     if camera:
#         image = Image.open(camera)
#         st.image(image, caption="Captured Image", use_container_width=True)

# # --- Single Image Upload
# elif option == "📂 Upload Image":
#     uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)

# # --- Predict Single Image
# if image:
#     st.markdown("---")
#     st.markdown("### 🔎 Prediction Result")

#     # Preprocess
#     img = image.resize((224, 224)).convert('RGB')
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)
#     score = prediction[0][0]
#     THRESHOLD = 0.3

#     # Show score
#     st.metric(label="📊 Prediction Score", value=f"{score:.4f}")

#     # Final result
#     if score > THRESHOLD:
#         st.error("🔴 Product Status: **Defective**")
#     else:
#         st.success("✅ Product Status: **Normal**")

# # --- Multiple Images Upload + Chart
# elif option == "🗂️ Upload Multiple Images":
#     uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

#     if uploaded_files:
#         st.markdown("### 📈 Score Chart")
#         image_names = []
#         scores = []

#         for file in uploaded_files:
#             img = Image.open(file).resize((224, 224)).convert('RGB')
#             img_array = np.array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             pred = model.predict(img_array)
#             score = pred[0][0]

#             image_names.append(file.name)
#             scores.append(score)

#         # Display bar chart
#         fig, ax = plt.subplots()
#         ax.barh(image_names, scores, color=["green" if s <= 0.3 else "red" for s in scores])
#         ax.set_xlabel("Prediction Score")
#         ax.set_title("Anomaly Detection Scores")

#         st.pyplot(fig)

#         # Show table with classification
#         for name, score in zip(image_names, scores):
#             status = "✅ Normal" if score <= 0.3 else "🔴 Defective"
#             st.write(f"**{name}** → Score: `{score:.4f}` → {status}")

# ------------------------------------------------------------------------------------


# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Load model
# model = tf.keras.models.load_model("model/keras_model.h5")

# # Page settings
# st.set_page_config(page_title="Anomaly Detector", layout="wide")

# # Custom CSS for styling
# st.markdown("""
# <style>
# body {
#     background-color: #f4f6f8;
# }
# h1, h2, h3 {
#     color: #1f2937;
#     font-family: 'Segoe UI', sans-serif;
# }
# .section {
#     background-color: white;
#     padding: 2rem;
#     border-radius: 12px;
#     box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
#     margin-bottom: 2rem;
# }
# .stButton>button {
#     background-color: #2563eb;
#     color: white;
#     border-radius: 6px;
#     font-weight: 600;
#     padding: 0.6em 1.4em;
#     margin-top: 1rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown("<h1 style='text-align:center;'>🔍 Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align:center; color:gray;'>Identify Defective vs Normal Products using Deep Learning</h3>", unsafe_allow_html=True)
# st.markdown("")

# # Layout
# left, right = st.columns(2)

# with left:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.subheader("📷 Choose Input Method")

#     option = st.radio("", ("Live Camera", "Upload Image"))

#     image = None
#     if option == "Live Camera":
#         camera = st.camera_input("Capture from Camera")
#         if camera:
#             image = Image.open(camera)
#     else:
#         uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#         if uploaded:
#             image = Image.open(uploaded)
#     st.markdown("</div>", unsafe_allow_html=True)

# with right:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.subheader("🖼️ Image Preview")
#     if image:
#         st.image(image, use_column_width=True)
#     else:
#         st.info("Upload or capture an image to preview here.")
#     st.markdown("</div>", unsafe_allow_html=True)

# # Prediction section
# if image:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.subheader("📊 Prediction Result")

#     # Preprocess image
#     img = image.resize((224, 224)).convert('RGB')
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     with st.spinner("Analyzing..."):
#         prediction = model.predict(img_array)
#         score = prediction[0][0]
#         threshold = 0.3

#     st.metric(label="Prediction Score", value=f"{score:.4f}")

#     if score > threshold:
#         st.error("🔴 Status: Defective Product")
#     else:
#         st.success("✅ Status: Normal Product")

#     st.markdown("</div>", unsafe_allow_html=True)

# # Footer
# st.markdown("""
# ---
# <center style='color:gray; font-size:0.9em;'>
#     Made with ❤️ using Streamlit and TensorFlow • 
#     <a href='https://linkedin.com' target='_blank'>Connect on LinkedIn</a>
# </center>
# """, unsafe_allow_html=True)










# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Page configuration
# st.set_page_config(page_title="AI Anomaly Detector", layout="centered")

# # Styling
# st.markdown("""
#     <style>
#         .title { font-size: 36px; font-weight: bold; color: #3E64FF; text-align:center; margin-bottom: 10px; }
#         .subtitle { font-size: 18px; color: #666; text-align:center; margin-bottom: 30px; }
#         .stButton>button { background-color: #3E64FF; color: white; font-weight: bold; border-radius: 8px; }
#         .metric-label { font-size: 16px; color: #666; }
#         .metric-value { font-size: 28px; font-weight: bold; color: #3E64FF; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("<div class='title'>🧪 AI Anomaly Detector</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload or Capture an Image to Detect Product Anomalies</div>", unsafe_allow_html=True)

# # Load model
# model = tf.keras.models.load_model("model/keras_model.h5")

# # Image Input
# image = None
# input_method = st.radio("Choose Input Method", ("📁 Upload Image", "📸 Use Camera"))

# if input_method == "📁 Upload Image":
#     uploaded_file = st.file_uploader("Upload a Product Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)

# elif input_method == "📸 Use Camera":
#     camera_image = st.camera_input("Capture Image with Camera")
#     if camera_image:
#         image = Image.open(camera_image)

# # Perform Prediction
# if image:
#     st.image(image, caption="Product Preview", use_column_width=False, width=250)

#     # Preprocessing
#     resized_img = image.resize((224, 224)).convert("RGB")
#     img_array = np.array(resized_img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Prediction
#     prediction = model.predict(img_array)
#     score = float(prediction[0][0])
#     THRESHOLD = 0.3

#     st.markdown("---")
#     st.markdown("### 🔍 Prediction Result")
#     st.metric(label="Prediction Score", value=f"{score:.4f}")

#     if score > THRESHOLD:
#         st.error("🔴 Product Status: **Defective**")
#     else:
#         st.success("✅ Product Status: **Normal**")

#     # Score Chart
#     st.markdown("### 📈 Visual Score Chart")
#     fig, ax = plt.subplots(figsize=(5, 3))
#     ax.scatter(["Input Image"], [score], color='red' if score > THRESHOLD else 'green', s=150)
#     ax.axhline(THRESHOLD, color='blue', linestyle='--', label='Threshold (0.3)')
#     ax.set_ylim(0, 1)
#     ax.set_ylabel("Score")
#     ax.set_title("Prediction Score Visualization")
#     ax.legend()
#     ax.grid(True)
#     st.pyplot(fig)

# else:
#     st.warning("📌 Please upload or capture an image to start detection.")








import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="AI Anomaly Detector", layout="centered", page_icon="🧪")

# CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .main {
            padding: 20px;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #3E64FF;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #444;
            text-align: center;
            margin-bottom: 40px;
        }
        .stButton>button {
            background-color: #3E64FF;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        .upload-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 18px;
            color: #555;
        }
        .metric-value {
            font-size: 30px;
            font-weight: bold;
            color: #3E64FF;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='title'>🧪 AI Anomaly Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload or Capture a Product Image to Check if it's Defective</div>", unsafe_allow_html=True)

# Load the model
model = tf.keras.models.load_model("model/keras_model.h5")

# Image input section
with st.container():
    st.markdown("#### 📷 Image Input", unsafe_allow_html=True)
    input_method = st.radio("Select Method", ("📁 Upload Image", "📸 Use Camera"))

    image = None
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload a Product Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif input_method == "📸 Use Camera":
        camera_image = st.camera_input("Take a Photo")
        if camera_image:
            image = Image.open(camera_image)

# Prediction and Display
if image:
    st.markdown("#### 🖼️ Product Preview")
    st.image(image, caption="Uploaded Product Image", width=200)

    # Preprocess
    resized_img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    score = float(prediction[0][0])
    THRESHOLD = 0.3

    st.markdown("#### 📊 Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric(label="Prediction Score", value=f"{score:.4f}")

    if score > THRESHOLD:
        col2.error("🔴 Product Status: Defective")
    else:
        col2.success("✅ Product Status: Normal")

    # Score Chart
    st.markdown("#### 📈 Score Scatter Chart")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(["Uploaded Image"], [score], color='red' if score > THRESHOLD else 'green', s=200)
    ax.axhline(THRESHOLD, color='blue', linestyle='--', label='Threshold (0.3)')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prediction Score")
    ax.set_title("Anomaly Score Visualization")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Optional: Future Multi-Image Scores
    st.info("📌 Want to compare multiple images in a score chart? You can add batch mode soon!")

else:
    st.warning("📌 Please upload or capture an image to begin prediction.")
