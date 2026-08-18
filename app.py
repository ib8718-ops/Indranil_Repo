import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import joblib
import os

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #  Save trained model as .pkl
    filename = f"/home/cloud/Desktop/model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")   # Debug message to confirm saving

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
    else:
        auc = "N/A"

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# Save results table
results_df = pd.DataFrame(results).T
print(results_df)
results_df.to_csv("/home/cloud/Desktop/test_data.csv", index=True)
