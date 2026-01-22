import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# --- Load trained model ---
model = load_model("football_logo_model.h5")

# --- Class labels (match your training folders) ---
class_labels = ["Arsenal", "Barcelona", "Chelsea", "ManUnited", "RealMadrid"]

# --- Streamlit interface ---
st.title("Football Club Logo Classifier")
st.write("Upload a football club logo image and click Predict")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

# Predict button
if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    img = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        # Predict
        pred = model.predict(img_array)
        predicted_index = np.argmax(pred)
        predicted_class = class_labels[predicted_index]
        confidence = pred[0][predicted_index]
        
        # Show prediction
        st.success(f"Predicted Club: {predicted_class} ({confidence*100:.2f}%)")
        
        # Optional: show bar chart of all classes
        st.subheader("Prediction Probabilities")
        st.bar_chart({class_labels[i]: float(pred[0][i]) for i in range(len(class_labels))})
