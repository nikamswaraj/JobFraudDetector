import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
vectorizer, model = joblib.load("logistic_job_fraud_model.pkl")

# Sidebar
st.sidebar.title("📊 Correlation Matrix")

try:
    raw_data = pd.read_csv("data/raw_data.csv")
    st.sidebar.write("Raw data preview:", raw_data.head())
    corr = raw_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.sidebar.pyplot(fig)
except Exception as e:
    st.sidebar.warning("Couldn't load correlation matrix.")

# Main Page
st.title("🛡️ Job Fraud Detector")
st.write("Paste a job ad to check if it may be fraudulent.")

text_input = st.text_area("Job Post Text", height=300)

if st.button("Analyze"):
    if text_input.strip():
        tf_input = vectorizer.transform([text_input])
        pred = model.predict(tf_input)[0]
# Load model
vectorizer, model = joblib.load("logistic_job_fraud_model.pkl")

# Load model accuracy
try:
    with open("data/model_accuracy.txt", "r") as f:
        model_accuracy = float(f.read().strip())
except:
    model_accuracy = None

# Main Page

if model_accuracy is not None:
    st.markdown(f"**Model Accuracy:** {model_accuracy:.2%}")
else:
    st.warning("Model accuracy not available.")



