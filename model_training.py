import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def get_performance_label(percentage):
    if percentage >= 75:
        return "Good"
    elif percentage >= 60:
        return "Average"
    else:
        return "Poor"

def train_model():
    df = pd.read_csv("dataset.csv")

    df["Performance"] = df["Percentage"].apply(get_performance_label)

    features = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "Total", "Percentage"]
    X = df[features]
    y = df["Performance"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    with open("model/student_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nModel saved to model/student_model.pkl")
    return model, le

if __name__ == "__main__":
    train_model()
