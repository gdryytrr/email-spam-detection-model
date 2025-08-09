import os
import pickle
import streamlit as st

# Ensure we are in the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Debug: show where we're running
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir(BASE_DIR))

# Load the model
model_path = os.path.join(BASE_DIR, "test_classifier_pipeline.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
vectorizer_path = os.path.join(BASE_DIR, "vectorizer (1).pkl")  # Exact name
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit app
st.title("Email Spam Detection")

user_input = st.text_area("Enter your email/message here:")

if st.button("Check"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        prediction_proba = model.predict_proba(transformed_input)[0]

        confidence = round(max(prediction_proba) * 100, 2)

        if prediction == 1:  # Spam
            st.error(f"🚨 This message is SPAM! (Confidence: {confidence}%)")
        else:
            st.success(f"✅ This message is NOT spam. (Confidence: {confidence}%)")
    else:
        st.warning("Please enter a message to check.")














