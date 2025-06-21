# app.py

import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Fraud Detector", page_icon="🛡️")

# --- Load Model ---
try:
    vectorizer, model = joblib.load("logistic_job_fraud_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Sidebar: Raw Data + Correlation ---
st.sidebar.title("📊 Data Insights")

try:
    raw_data = pd.read_csv("data/raw_data.csv", encoding="ISO-8859-1")
    st.sidebar.subheader("🔍 Sample of Raw Data")
    st.sidebar.dataframe(raw_data.head())

    st.sidebar.subheader("📈 Correlation Matrix")
    corr = raw_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.sidebar.pyplot(fig)
except Exception as e:
    st.sidebar.warning("⚠️ Couldn’t load raw data or correlation matrix.")

# --- Load Model Accuracy ---
try:
    with open("data/model_accuracy.txt", "r") as f:
        model_accuracy = float(f.read().strip())
except:
    model_accuracy = None

# --- Main Interface ---
st.title("🛡️ Job Fraud Detection App")
st.markdown("Analyze job ads for potential fraud. Just paste the text below and click **Analyze**.")

description = st.text_area("💼 Job Ad Content", height=300, placeholder="Paste the job description here...")

if st.button("🔎 Analyze"):
    try:
        if not description or not description.strip():
            raise ValueError("Job description cannot be empty.")

        clean_text = description.strip().lower()
        word_count = len(clean_text.split())

        if len(clean_text) < 30 or word_count < 5:
            raise ValueError("The job description is too short or uninformative. Please provide more detail.")

        # --- Prediction ---
        tf_input = vectorizer.transform([description])
        prediction = model.predict(tf_input)[0]
        proba = model.predict_proba(tf_input)[0]
        legit_score = proba[0]
        fraud_score = proba[1]
        confidence_gap = abs(legit_score - fraud_score)

        # --- Threshold logic ---
        if confidence_gap < 0.1:
            st.warning("🤔 This job description is **ambiguous**. The model isn't confident enough to decide.")
            st.markdown(f"**Confidence → Legit: {legit_score:.2%} | Fraudulent: {fraud_score:.2%}**")
        elif prediction == 1:
            st.error("🚨 This job post appears to be **FRAUDULENT**.")
            st.markdown(f"**Confidence → Legit: {legit_score:.2%} | Fraudulent: {fraud_score:.2%}**")
        else:
            st.success("✅ This job post appears to be **LEGITIMATE**.")
            st.markdown(f"**Confidence → Legit: {legit_score:.2%} | Fraudulent: {fraud_score:.2%}**")

    except ValueError as ve:
        st.warning(f"⚠️ {ve}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --- Footer Accuracy Info ---
if model_accuracy is not None:
    st.markdown(f"📊 **Model Accuracy:** `{model_accuracy:.2%}`")
else:
    st.markdown("⚠️ *Model accuracy unavailable.*")
