import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('waste_classification_mobilenetv2_pro.keras')

# Define the class names
class_names = ['metal waste', 'organic waste', 'paper waste', 'plastic waste']

def classify_image(image):
    """
    Takes an image, preprocesses it, and returns the model's prediction.
    """
    st.image(image, caption='Classifying Image...', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resize the image to the model's expected input size
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write(f"**Prediction:** {predicted_class_name}")
    st.write(f"**Confidence:** {confidence:.2f}")

st.title("Waste Classification App")
st.write("Upload an image of waste, or use your camera to snap a picture.")

input_method = st.radio("Choose input method:", ("Upload a file", "Use camera"))

if input_method == "Upload a file":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        classify_image(image)
else: # Use camera
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        classify_image(image)
