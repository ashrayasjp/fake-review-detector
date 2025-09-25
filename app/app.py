import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_review

st.set_page_config(page_title="Fake Review Detector", page_icon="🤖", layout="centered")
st.title("Fake Review Detector 🕵️‍♂️")

st.markdown(
    "Enter a review below and the model will predict whether it is **Fake** or **Genuine**."
)

review_input = st.text_area("", height=150)

if st.button("Check Review"):
    if not review_input.strip():
        st.warning("Please enter a review to check.")
    else:
        result = predict_review(review_input)
        if result == "fake":
            st.error("⚠️ This review is likely **Fake**.")
        else:
            st.success("✅ This review appears **Genuine**.")

st.markdown("---")

