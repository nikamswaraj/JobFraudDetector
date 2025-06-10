# app.py

import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
vectorizer, model = joblib.load("logistic_job_fraud_model.pkl")

# Sidebar - Data & Correlation
st.sidebar.title("📊 Correlation Matrix")
try:
    raw_data = pd.read_csv("data/raw_data.csv")
    st.sidebar.subheader("Raw Data Sample")
    st.sidebar.dataframe(raw_data.head())

    corr = raw_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.sidebar.pyplot(fig)
except:
    st.sidebar.warning("Couldn't load data or correlation matrix.")

# Load model accuracy
try:
    with open("data/model_accuracy.txt", "r") as f:
        model_accuracy = float(f.read().strip())
except:
    model_accuracy = None

# Main App
st.title("🛡️ Job Fraud Detector")
st.write("Paste a job ad and click **Analyze** to check for fraud.")

text_input = st.text_area("Job Post Text", height=300)

if st.button("Analyze"):
    if text_input.strip():
        tf_input = vectorizer.transform([text_input])
        prediction = model.predict(tf_input)[0]
        proba = model.predict_proba(tf_input)[0]

        if prediction == 1:
            st.error("🚨 This job post appears to be **FRAUDULENT**.")
        else:
            st.success("✅ This job post appears to be **LEGITIMATE**.")

        st.markdown(f"**Confidence → Legit: {proba[0]:.2%} | Fraudulent: {proba[1]:.2%}**")
    else:
        st.warning("Please paste a job post to analyze.")

if model_accuracy is not None:
    st.markdown(f"**Model Accuracy:** {model_accuracy:.2%}")
