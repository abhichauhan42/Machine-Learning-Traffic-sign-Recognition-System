import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("C:/Users/hp/Desktop/Project/my_model1.h5")

st.title('Your Machine Learning App')

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = image.resize((224, 224))  # Resize the image to the size your model expects
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])
    st.write(f'The model predicts this image as class {class_idx}.')

