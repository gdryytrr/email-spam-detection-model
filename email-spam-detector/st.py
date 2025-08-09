import streamlit as st
import pickle

# ------------------------
# Load the model & vectorizer
# ------------------------
with open("test_classifier_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer (1).pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------
# Prediction function
# ------------------------
def predict_spam(message):
    # Vectorize the input text
    message_transformed = vectorizer.transform([message])
    
    # Predict
    prediction = model.predict(message_transformed)[0]
    confidence = model.predict_proba(message_transformed).max() * 100  # percentage
    
    return prediction, confidence

# ------------------------
# Streamlit UI
# ------------------------
st.title("📧 Email Spam Detection App")

user_input = st.text_area("Enter your email message:")

if st.button("Check"):
    if user_input.strip():
        prediction, confidence = predict_spam(user_input)

        if prediction == 1:  # Assuming 1 = Spam
            st.error(f"🚨 This message is SPAM! Confidence: {confidence:.2f}%")
        else:  # Assuming 0 = Not Spam
            st.success(f"✅ This message is NOT spam. Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a message first.")












