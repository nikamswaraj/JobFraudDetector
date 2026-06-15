# 🕵️‍♂️ Job Posting Fraud Detection System

## 📌 Overview

The **Job Posting Fraud Detection System** is a machine learning-based web application that helps users identify whether a job posting is **legitimate** or **fraudulent**. The application uses **Natural Language Processing (NLP)** and **Logistic Regression** to analyze job descriptions and predict potential fraud risks.

Built with **Streamlit**, the system provides an interactive user interface where users can paste job descriptions and receive real-time predictions along with useful insights.

---

## 🚀 Features

* ✅ Detects fraudulent job postings using Machine Learning
* ✅ Real-time job description analysis
* ✅ Interactive Streamlit dashboard
* ✅ Displays fraud confidence score
* ✅ Extracts top keywords from job descriptions
* ✅ Identifies suspicious trigger words
* ✅ Shows model evaluation metrics
* ✅ Confusion Matrix visualization
* ✅ User-friendly interface

---

## 🛠️ Technologies Used

| Technology                  | Purpose                        |
| --------------------------- | ------------------------------ |
| Python                      | Programming Language           |
| Streamlit                   | Web Application Framework      |
| Pandas                      | Data Processing                |
| NumPy                       | Numerical Computation          |
| Scikit-learn                | Machine Learning               |
| TF-IDF Vectorizer           | Text Feature Extraction        |
| Logistic Regression         | Fraud Classification           |
| Matplotlib                  | Data Visualization             |
| Seaborn                     | Confusion Matrix Visualization |
| Regular Expressions (Regex) | Text Cleaning                  |

---

## 📂 Project Structure

```text
Job-Fraud-Detection/
│
├── app.py
├── fake_job_postings.csv
├── requirements.txt
├── README.md
└── screenshots/
    ├── homepage.png
    ├── prediction.png
    └── evaluation.png
```

---

## ⚙️ Working Principle

### 1. Dataset Loading

The system loads the `fake_job_postings.csv` dataset containing legitimate and fraudulent job postings.

### 2. Data Preprocessing

* Removes unnecessary columns
* Handles missing values
* Combines:

  * Description
  * Requirements
  * Benefits

into a single text feature.

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts text data into numerical vectors.

### 4. Model Training

A Logistic Regression classifier is trained using:

* 80% Training Data
* 20% Testing Data

### 5. Prediction

When a user enters a job description:

* Text is transformed using TF-IDF
* Logistic Regression predicts:

  * Legitimate Job
  * Fraudulent Job

### 6. Results Visualization

The application displays:

* Prediction Result
* Confidence Score
* Top Keywords
* Trigger Words
* Accuracy
* Classification Report
* Confusion Matrix

---

## 📊 Machine Learning Model

### Algorithm Used

**Logistic Regression**

Reasons for choosing Logistic Regression:

* Fast training and prediction
* Performs well on text classification tasks
* Interpretable results
* Suitable for binary classification

---

## 📈 Evaluation Metrics

The system evaluates model performance using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example:

```text
Accuracy: 95%

Precision: 92%
Recall: 90%
F1-Score: 91%
```

---

## 🔥 Suspicious Keywords Detection

The system checks for commonly used scam-related terms such as:

* congratulations
* earn
* click
* limited
* urgent
* guaranteed
* fee

These words help users identify potentially suspicious job advertisements.

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/job-fraud-detection.git
cd job-fraud-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📋 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install all packages:

```bash
pip install -r requirements.txt
```

---

## 📸 Screenshots

### Home Page

(Add screenshot here)

### Prediction Result

(Add screenshot here)

### Model Evaluation

(Add screenshot here)

---

## 🎯 Future Enhancements

* Deep Learning Models (LSTM/BERT)
* Fake Company Detection
* Resume-Job Matching
* Real-time Job Scraping
* Explainable AI (XAI)
* Multi-Language Support
* Email Scam Detection

---

## 👨‍💻 Author

**Swaraj Nikam**

B.Tech Student
AI & Data Science Enthusiast

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful:

* Star the repository ⭐
* Fork the project 🍴
* Contribute improvements 🚀

---

### Conclusion

The Job Posting Fraud Detection System provides an efficient and intelligent approach to identifying fraudulent job advertisements. By leveraging Machine Learning and NLP techniques, the system helps job seekers avoid scams and make informed career decisions.
