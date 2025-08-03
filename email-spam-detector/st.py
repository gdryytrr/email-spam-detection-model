import streamlit as st
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# App title
st.markdown("<h1 style='text-align: center;'>📧 Email Spam Detector</h1>", unsafe_allow_html=True)

# Load model
model = joblib.load("test_classifier_pipeline.pkl")

# Text input
st.subheader("Enter a Message:")
message = st.text_area("Put a message here...")

# Predict button
if st.button("🔍 Detect"):
    if message:
        # Predict
        prediction = model.predict([message])[0]
        confidence = model.predict_proba([message]).max() * 100

        # Show result with color and confidence
        if prediction == "spam":
            st.error(f"⚠️ This message is classified as **Spam** with {confidence:.2f}% confidence.")
        else:
            st.success(f"✅ This message is classified as **Not Spam** with {confidence:.2f}% confidence.")
    else:
        st.warning("Please enter a message to analyze.")

