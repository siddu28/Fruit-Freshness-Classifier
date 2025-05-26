import streamlit as st
from PIL import Image
from model_helper import predict

st.title("Fruit Freshness Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open("temp_file.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        prediction = predict("temp_file.jpg")
        st.image("temp_file.jpg", caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Failed to process the image. Error: {e}")
