import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="wide")

st.title("🌾 Rice Leaf Disease Detection using Deep Learning")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("rice_disease_model.h5")
    return model

model = load_trained_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Healthy"
]

# -----------------------------
# Disease Solutions
# -----------------------------
solutions = {
    "Bacterial Leaf Blight":
        "Use resistant rice varieties and apply balanced fertilizers. Avoid excessive nitrogen.",
    
    "Brown Spot":
        "Apply fungicides and improve soil fertility. Maintain proper irrigation.",
    
    "Leaf Smut":
        "Use certified seeds and apply appropriate fungicides.",
    
    "Healthy":
        "The rice leaf is healthy. No treatment required."
}

# -----------------------------
# GradCAM Function
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1.image(img, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Preprocess Image
    # -----------------------------
    img_resized = cv2.resize(img, (224,224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = float(np.max(prediction))*100

    st.subheader("Prediction Result")

    st.write("Disease :", pred_class)
    st.write("Confidence :", f"{confidence:.2f}%")

    st.subheader("Recommended Solution")

    st.write(solutions[pred_class])

    # -----------------------------
    # GradCAM
    # -----------------------------
    last_conv_layer_name = model.layers[-5].name

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        pred_index
    )

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    col2.image(heatmap, caption="GradCAM Heatmap", use_container_width=True)

    st.subheader("GradCAM Overlay")

    st.image(superimposed_img.astype("uint8"), use_container_width=True)

# -----------------------------
# Accuracy Section
# -----------------------------
st.sidebar.title("Model Information")

st.sidebar.write("Model: Transfer Learning CNN")

st.sidebar.write("Training Accuracy: 96%")

st.sidebar.write("Input Size: 224 x 224")

st.sidebar.write("Dataset: Rice Leaf Disease Dataset")

st.sidebar.write("Author: Yuvraj Sharma")
