import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('fine_tuned_mobilenetv2.h5')

# Class names (change these based on your classes)
class_names = ['class_1', 'class_2', 'class_3']

def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    return class_names[predicted_class[0]]

def main():
    st.title('CNN Image Classifier with Streamlit')
    st.write('Upload an image and the model will predict the class.')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        prediction = predict(image)

        st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
