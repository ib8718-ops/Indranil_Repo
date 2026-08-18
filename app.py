import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.title("🍷 Wine Quality Classification App")

# ✅ Dataset upload option
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.subheader("Dataset Preview")
    st.write(df.head())

    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    # ✅ Model selection dropdown
    model_choice = st.selectbox("Select a model", list(models.keys()))

    if st.button("Run Model"):
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        os.makedirs("model", exist_ok=True)
        filename = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)

        # Metrics
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(label_binarize(y_test, classes=sorted(y.unique())), y_prob, multi_class="ovr")
        else:
            auc = "N/A"

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        # ✅ Display evaluation metrics
        st.subheader("Evaluation Metrics")
        st.write(pd.DataFrame(metrics, index=[model_choice]))

        # ✅ Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())))

        # ✅ Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())
