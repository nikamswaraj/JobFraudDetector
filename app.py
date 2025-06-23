import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Live Job Fraud Detector", layout="wide")

st.title("🕵️‍♂️ Job Posting Fraud Detection")
st.markdown("Paste any job description to predict if it's **fraudulent** or **legitimate**. Data insights update as you type!")

# Load and train model
@st.cache_data
def load_model():
    df = pd.read_csv("fake_job_postings.csv", encoding='ISO-8859-1')
    df['text'] = df['description'].fillna('') + ' ' + df['requirements'].fillna('') + ' ' + df['benefits'].fillna('')
    df = df[['text', 'fraudulent']].dropna()
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['text'])
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    return model, tfidf, X_test, y_test, model.predict(X_test)

model, tfidf, X_test, y_test, y_pred = load_model()

# ---- Sidebar: Data Insights ----
st.sidebar.title("📊 Data Insights")

# ---- User input ----
user_input = st.text_area("Paste Job Posting Text Below", height=300)

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9 ]", "", text.lower())
    return text

def extract_keywords(text, top_n=5):
    words = clean_text(text).split()
    stopwords = set(tfidf.get_stop_words() or [])
    words = [word for word in words if word not in stopwords]
    return Counter(words).most_common(top_n)

if user_input.strip():
    # Sidebar insights
    word_count = len(user_input.split())
    st.sidebar.markdown(f"**📝 Word Count:** {word_count}")

    # Extract top words
    top_words = extract_keywords(user_input)
    st.sidebar.markdown("**🔥 Top Keywords:**")
    for word, count in top_words:
        st.sidebar.write(f"• {word} ({count}x)")

    # Spammy keywords (simple examples)
    spammy_words = ['congratulations', 'earn', 'click', 'limited', 'urgent', 'guaranteed', 'fee']
    found_spam = [word for word in spammy_words if word in clean_text(user_input)]
    if found_spam:
        st.sidebar.markdown("**⚠️ Trigger Words:**")
        for word in found_spam:
            st.sidebar.write(f"• `{word}`")
    else:
        st.sidebar.markdown("**✅ No spammy words detected.**")

    # ---- Predict fraud ----
    input_vec = tfidf.transform([user_input])
    pred = model.predict(input_vec)[0]
    prob = model.predict_proba(input_vec)[0][pred]
    label = "🚨 Fraudulent" if pred == 1 else "✅ Legitimate"

    st.subheader("🔍 Prediction")
    st.markdown(f"**Result**: {label}")
    st.markdown(f"**Confidence**: {prob:.2f}")

else:
    st.sidebar.markdown("Enter a job posting above to see insights.")

# ---- Model performance ----
with st.expander("📈 Show Model Evaluation"):
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

