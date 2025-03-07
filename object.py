#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D

# Custom layer definition for DepthwiseConv2D that ignores the 'groups' argument
def custom_depthwise_conv2d(kernel_size, strides, padding, depth_multiplier, activation, use_bias, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Remove 'groups' argument if it exists
    return DepthwiseConv2D(
        kernel_size=kernel_size, 
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        **kwargs
    )

# Load the trained model, ensuring custom layers (like DepthwiseConv2D) are included
try:
    model = load_model("keras_model.h5", custom_objects={
        'DepthwiseConv2D': custom_depthwise_conv2d,
        'Conv2D': Conv2D
    }, compile=False)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Define labels (Make sure this matches the model output shape)
labels = ["Background", "Watch", "Earbuds"]

# Preprocess image before passing to the model
def preprocess_image(image):
    if image.mode == 'RGBA':  # Convert RGBA to RGB
        image = image.convert('RGB')
    
    image = image.resize((224, 224))  # Resize image to expected input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    image = preprocess_image(image)
    predictions = model.predict(image)

    st.write("📊 Model Output:", predictions)  # Debugging output

    if predictions.ndim == 1:  # If model outputs a flat array
        predicted_class = np.argmax(predictions)
    else:
        predicted_class = np.argmax(predictions, axis=1)[0]

    # Handle out-of-bounds index errors
    if predicted_class >= len(labels):
        return "Unknown", 0.0

    return labels[predicted_class], float(predictions[0][predicted_class])

# Streamlit app layout
st.title("🖼️ Object Classification App")
st.write("Upload an image for classification.")

# File uploader widget
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="🖼️ Uploaded Image", use_column_width=True)

    # When user clicks the "Predict" button
    if st.button("🔍 Predict"):
        label, confidence = predict(uploaded_image)
        st.write(f"🎯 **Prediction:** {label} (Confidence: {confidence:.2f})")
