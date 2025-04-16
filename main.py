'''
import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
'''

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

    # Button to classify
   st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .center-button button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }
    .center-button button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create centered button and trigger prediction
button_clicked = st.markdown(
    """
    <div class="center-button">
        <form action="" method="post">
            <button type="submit">🧪 Classify</button>
        </form>
    </div>
    """,
    unsafe_allow_html=True
)

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

