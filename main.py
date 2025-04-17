import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# --- Google Drive Model URL ---
model_file = "plant_model.h5"
file_id = "1KnQ0U6y-nX4t428Yd0wMuq3y7qIe44Di"
url = f"https://drive.google.com/uc?id={file_id}"

# --- Download model if not present ---
if not os.path.exists(model_file):
    with st.spinner("🔄 Downloading model from Google Drive..."):
        try:
            gdown.download(url, model_file, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.stop()

# --- Load model (cached) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_file)

model = load_model()

# --- Class labels (38 classes) ---
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- UI ---
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a plant leaf image to classify the disease using a deep learning model.")

# --- Upload image ---
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

  
# Handle prediction logic

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🧪 Classify", use_container_width=True):
        prediction = model.predict(img_array)
        if prediction.shape[1] == len(class_names):
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = np.max(prediction)
            st.success(f"🧠 Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: {confidence * 100:.2f}%")
        else:
            st.error("❌ Invalid model output. Check number of classes or input shape.")

'''
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page config
st.set_page_config(page_title="🌿 Plant Disease Detector", page_icon="🧪", layout="centered")

# --- Google Drive Model URL ---
model_file = "plant_model.h5"
file_id = "1KnQ0U6y-nX4t428Yd0wMuq3y7qIe44Di"
url = f"https://drive.google.com/uc?id={file_id}"

# --- Download model if not present ---
if not os.path.exists(model_file):
    with st.spinner("🔄 Downloading model from Google Drive..."):
        try:
            gdown.download(url, model_file, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.stop()

# --- Load model (cached) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_file)

model = load_model()

# --- Class labels (38 classes) ---
class_names = [ ... ]  # same list as before

# --- UI Header ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a plant leaf image to detect disease using our deep learning model</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload Image Section ---
uploaded_file = st.file_uploader("📷 **Upload Image**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.markdown("---")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("🧪 **Classify Leaf Disease**", use_container_width=True):
            with st.spinner("🔍 Analyzing image..."):
                prediction = model.predict(img_array)
                if prediction.shape[1] == len(class_names):
                    predicted_index = np.argmax(prediction)
                    predicted_class = class_names[predicted_index]
                    confidence = np.max(prediction)

                    st.markdown(f"""
                    <div style='background-color: #e6ffe6; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: #2E8B57;'>🧠 Prediction: <b>{predicted_class}</b></h2>
                        <p style='font-size: 18px;'>📊 Confidence: <b>{confidence * 100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Invalid model output. Please check model class count or input shape.")
else:
    st.info("⬆️ Please upload an image to get started.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>🌱 Built with TensorFlow & Streamlit</p>", unsafe_allow_html=True)
'''
