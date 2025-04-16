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

# Set page config (must be first command)
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# --- Model file and Google Drive URL ---
model_file = "plant_model.h5"
file_id = "1KnQ0U6y-nX4t428Yd0wMuq3y7qIe44Di"
url = f"https://drive.google.com/uc?id={file_id}"

# --- Download model if not already present ---
if not os.path.exists(model_file):
    with st.spinner("🔄 Downloading model file from Google Drive..."):
        try:
            gdown.download(url, model_file, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.stop()

# --- Load the model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_file)
    return model

model = load_model()

# --- Class labels (update these to your actual classes, ensure you have 38 classes here) ---
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 
               'Class 11', 'Class 12', 'Class 13', 'Class 14', 'Class 15', 'Class 16', 'Class 17', 'Class 18', 'Class 19', 'Class 20', 
               'Class 21', 'Class 22', 'Class 23', 'Class 24', 'Class 25', 'Class 26', 'Class 27', 'Class 28', 'Class 29', 'Class 30',
               'Class 31', 'Class 32', 'Class 33', 'Class 34', 'Class 35', 'Class 36', 'Class 37', 'Class 38']  # Ensure you have 38 labels

# --- Streamlit UI ---
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a plant leaf image to detect possible diseases using a deep learning model.")

# --- File uploader ---
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Preprocessing ---
    img = image.resize((224, 224))  # Update this size if your model uses something else
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Prediction ---
    prediction = model.predict(img_array)
    
    # Debugging: Print the raw prediction output
    st.write(f"Prediction array: {prediction}")  # Print raw prediction
    
    # Check the prediction shape to ensure it's correct
    st.write(f"Prediction shape: {prediction.shape}")

    # Ensure that the prediction array has the expected dimensions
    if prediction.shape[0] > 0 and prediction.shape[1] == len(class_names):
        predicted_index = np.argmax(prediction)
        st.write(f"Predicted index: {predicted_index}")  # Debugging: Print predicted index
        confidence = np.max(prediction)
        st.success(f"🧠 Prediction: **{class_names[predicted_index]}**")
        st.info(f"📊 Confidence: {confidence * 100:.2f}%")
    else:
        st.error("❌ Invalid prediction output. Please check the model input and output.")

