import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\spam.csv")

# Preprocess data
bow = CountVectorizer(stop_words='english')
x = df['Message']
y = df['Category']
X_bow = bow.fit_transform(x)  # Keep as sparse matrix

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)


# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Multinomial NB": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# Streamlit UI
st.title("📩 Spam Detector")
st.markdown("## Choose a model to classify emails as Spam or Ham")

# Select model
selected_model = st.selectbox("Select a Machine Learning Model", list(models.keys()))

# Train model with train-test split for accuracy
model = models[selected_model]
if selected_model == "SVM":
    model.fit(x_train, y_train)  # Train with sparse matrix
    y_pred = model.predict(x_test)  # Predict with sparse matrix
else:
    model.fit(x_train.toarray(), y_train)  # Convert to dense
    y_pred = model.predict(x_test.toarray())  # Convert to dense

accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.markdown(f"### 🔍 {selected_model} Accuracy: **{accuracy:.4f}**")

# Train model with full data for prediction
if selected_model == "SVM":
    model.fit(X_bow, y)  # Train with sparse matrix
else:
    model.fit(X_bow.toarray(), y)  # Convert to dense

# Email input
email_text = st.text_area("📧 Enter an email to check if it's spam or ham:")

# Predict button
if st.button("🚀 Predict"):
    if email_text:
        email_vector = bow.transform([email_text])  # Keep as sparse
        email_vector_nb = email_vector.toarray()  # Convert for Naive Bayes
        
        prediction = model.predict(email_vector_nb if selected_model == "Naive Bayes" else email_vector)[0]
        if prediction == "spam":
            st.error("🚨 This email is **Spam**!")
        else:
            st.success("✅ This email is **Ham** (Not Spam).")
    else:
        st.warning("⚠️ Please enter an email message.")
