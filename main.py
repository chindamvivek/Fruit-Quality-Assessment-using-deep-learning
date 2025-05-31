import tensorflow as tf
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('inception_resnetv2_fruit_quality2ndtime.keras')  # Update with your model path
    return model

model = load_model()

# Load treatment suggestions from JSON file
@st.cache_resource
def load_treatment_suggestions():
    with open("treatment_suggestions.json", "r") as file:
        return json.load(file)
    
treatment_data = load_treatment_suggestions()

# Define class labels
class_names = ['Apple_Bad',
 'Apple_Good',
 'Apple_mixed',
 'Banana_Bad',
 'Banana_Good',
 'Banana_mixed',
 'Guava_Bad',
 'Guava_Good',
 'Guava_mixed',
 'Lemon_mixed',
 'Lime_Bad',
 'Lime_Good',
 'Orange_Bad',
 'Orange_Good',
 'Orange_mixed',
 'Pomegranate_Bad',
 'Pomegranate_Good',
 'Pomegranate_mixed']


# Function to get detailed treatment suggestions
def get_treatment_suggestions(fruit, quality):
    fruit = fruit.capitalize()
    quality = quality.capitalize()

    if fruit in treatment_data and quality in treatment_data[fruit]:
        return treatment_data[fruit][quality]
    
    return ["No specific suggestions available for this fruit."]

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# Home Page
if page == "Home":
    st.title("🍇 Fruit Quality Assessment System")
    st.markdown("""
Welcome to the **Fruit Quality Assessment System**!  
This application uses deep learning to assess the quality of various fruits from an uploaded image.  
You will receive a quality prediction and AI-powered treatment suggestions.

---

### 🔍 Model Used:
- **InceptionResNetV2 (Pretrained)**

### ✅ Features:
- Confidence Score  
- Rule-based Treatment Suggestions  

---

### 🔧 How to Use This App:
1. Go to the **Prediction** page using the sidebar.
2. Upload an image of a fruit (supported formats: **JPG, PNG, JPEG**).
3. Click **"Show Image"** to confirm the uploaded image.
4. Click **"Predict"** to get:
   - Fruit type
   - Quality assessment
   - Confidence score
5. Receive **AI-powered treatment suggestions** to improve or manage the fruit quality (based on rule-based logic).

---

### 🌟 Advantages of This System:
- ✅ **Easy to Use**: No technical knowledge required — just upload and click!
- ✅ **Fast Results**: Instant classification with confidence percentage.
- ✅ **Wide Fruit Support**: Supports **18 fruit categories** with multiple quality classes.
- ✅ **Useful Suggestions**: Offers practical treatment advice for bad or mixed quality fruits.
- ✅ **Accessible Anywhere**: Works on any device with a browser using Streamlit.
- ✅ **Offline Prediction Ready**: All logic is handled on-device (post-deployment), suitable for remote or rural users.
- ✅ **Built with AI**: Uses **transfer learning** for accurate and efficient predictions.

---
    """)


# About Page
elif page == "About":
    st.title("📄 About This Project")
    st.markdown("""
This project is developed to help **farmers and traders** quickly assess the quality of fruits.  
It classifies fruit images into different quality categories (**Good, Bad, Mixed**) and provides treatment advice accordingly.

---

### 🧠 Technologies Used:
- **TensorFlow & Keras** – for model building and training  
- **Transfer Learning** – using **InceptionResNetV2** for improved accuracy  
- **Streamlit** – to build an interactive and user-friendly web interface  
- **JSON-based Rule System** – to provide smart treatment suggestions

---

### 📊 Dataset Information:
- Total Images: **19,526**  
- Fruit Categories: **18 classes** including Apple, Banana, Guava, Lime, Orange, Pomegranate, etc.  
- Quality Labels: **Bad**, **Good**, and **Mixed**  
- Data Split: **Training (80%)**, **Validation (10%)**, **Test (10%)**  
- Each image is labeled with both **fruit type** and **quality** (e.g., `Banana_Good`, `Apple_Bad`, etc.)

---

### 🎓 Project Purpose:
- Aimed to provide a helpful AI-based tool that can potentially assist in **agriculture and food quality control**

---
    """)


# Prediction Page
elif page == "Prediction":
    st.title("🍏🍌 Fruit Quality Prediction 🍊🍓")
    st.write("Upload a fruit image to assess its quality.")

    uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        if st.button("Show Image"):
            st.image(image_pil, caption='Uploaded Image', width=300)

        if st.button("Predict"):
            # Preprocess the image
            img = image_pil.resize((256, 256))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)

            # Make prediction
            pred = model.predict(img_array)
            pred_class_index = np.argmax(pred, axis=1)[0]
            confidence_score = np.max(pred) * 100  # Convert to percentage
            pred_class = class_names[pred_class_index]

            # Extract fruit and quality
            if "_" in pred_class:
                fruit, quality = pred_class.split("_")
            else:
                fruit, quality = pred_class, "Unknown"

            # Show the predicted result
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(image_pil)
            ax.set_title(f"Predicted: Fruit: {fruit}, Quality: {quality}", fontsize=10)
            ax.axis('off')
            st.pyplot(fig)

            st.write(f"### ✅ Fruit: {fruit}, Quality: {quality}")
            st.write(f"### 🔍 Confidence Score: {confidence_score:.2f}%")

            # Flag uncertain predictions
            threshold = 50  # Confidence threshold
            if confidence_score < threshold:
                st.warning("⚠️ This prediction has low confidence. The result may be incorrect.")

            # Fetch treatment suggestions from JSON
            treatment_suggestions = get_treatment_suggestions(fruit, quality)

            st.subheader("💡 Edibility and Storage Recommendations:")
            for suggestion in treatment_suggestions:
                st.write(f"- {suggestion}")