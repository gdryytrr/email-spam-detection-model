import streamlit as st
import joblib
import os

# Load model and vectorizer
model = joblib.load("test_classifier_pipeline.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title and style
st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 40px;
        color: #4B0082;
        text-align: center;
        margin-bottom: 30px;
    }
    .spam-box {
        background-color: #FFCCCC;
        padding: 20px;
        border-radius: 10px;
        color: #8B0000;
        font-weight: bold;
    }
    .ham-box {
        background-color: #CCFFCC;
        padding: 20px;
        border-radius: 10px;
        color: #006400;
        font-weight: bold;
    }
    .footer {
        margin-top: 50px;
        font-size: 13px;
        color: gray;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📧 Email Spam Detection</div>', unsafe_allow_html=True)

message = st.text_area("Enter your message here:")

if st.button("Check if Spam"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and predict
        vectorized_msg = vectorizer.transform([message])
        prediction = model.predict(vectorized_msg)[0]
        confidence = model.predict_proba(vectorized_msg).max() * 100

        if prediction == 1:
            st.markdown(
                f'<div class="spam-box">🚨 This message is likely <b>SPAM</b> with <b>{confidence:.2f}%</b> confidence.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ham-box">✅ This message is <b>NOT SPAM</b> with <b>{confidence:.2f}%</b> confidence.</div>',
                unsafe_allow_html=True,
            )

st.markdown('<div class="footer">Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)



