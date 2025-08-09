import streamlit as st
import joblib
import numpy as np
import os

# Preprocess function (used during unpickling if needed)
def preprocess_text(text):
    return text.lower()

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "test_classifier_pipeline.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Prediction function
def predict_spam(message):
    transformed_text = vectorizer.transform([message])
    prediction = model.predict(transformed_text)[0]
    confidence = np.max(model.predict_proba(transformed_text))
    return prediction, confidence

# Streamlit page settings
st.set_page_config(page_title="Email Spam Detector", page_icon="📩", layout="centered")
st.title("📩 Email Spam Detector")
st.markdown("#### Enter a Message:")

# User input
user_input = st.text_area("Put a message here...")

if st.button("🔍 Detect"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to check.")
    else:
        prediction, confidence = predict_spam(user_input)

        if prediction == 1:
            st.error(f"🚨 This message is classified as **Spam** with {confidence * 100:.2f}% confidence.")
        else:
            st.success(f"✅ This message is classified as **Legit** with {confidence * 100:.2f}% confidence.")








