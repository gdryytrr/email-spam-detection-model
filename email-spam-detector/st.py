import streamlit as st
import joblib
import os

# Set the absolute path for the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "test_classifier_pipeline.pkl")

# Set page configuration
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# App title
st.markdown("<h1 style='text-align: center;'>📧 Email Spam Detector</h1>", unsafe_allow_html=True)

# Load model safely
model = joblib.load(MODEL_PATH)

# Text input
st.subheader("Enter a Message:")
message = st.text_area("Put a message here...")

# Predict button
if st.button("🔍 Detect"):
    if message:
        prediction = model.predict([message])[0]
        confidence = model.predict_proba([message]).max() * 100

        if prediction == "spam":
            st.error(f"⚠️ This message is classified as **Spam** with {confidence:.2f}% confidence.")
        else:
            st.success(f"✅ This message is classified as **Not Spam** with {confidence:.2f}% confidence.")
    else:
        st.warning("Please enter a message to analyze.")

