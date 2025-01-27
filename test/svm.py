import json
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
metadata_path = r"E:\Project_2\metadata\reduced_metadata2.json"
log_path = r"E:\Project_2\logs\svm_rm1_log.txt"
model_path = r"E:\Project_2\models\svm_rm1.pkl"

# Load Metadata
def load_metadata(metadata_path):
    with open(metadata_path, "r") as file:
        data = json.load(file)
    return data

# Extract Features and Labels
def prepare_data(metadata):
    features, labels = [], []
    for video_id, details in metadata.items():
        for chunk in details["chunks"]:
            # Load features
            feature_path = chunk["video_feature_path"]
            features.append(np.load(feature_path))
            # Label: Assuming label is "real" (0) or "fake" (1) based on filename
            labels.append(1 if "Fake" in video_id else 0)
    return np.array(features), np.array(labels)

# Logging Function
def log_metrics(log_path, metrics):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        for key, value in metrics.items():
            log_file.write(f"{key}: {value}\n")

# Main Training Pipeline
if __name__ == "__main__":
    # Load metadata
    metadata = load_metadata(metadata_path)

    # Prepare features and labels
    X, y = prepare_data(metadata)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train SVM classifier
    svm_model = SVC(kernel="linear", probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba)
    }

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(svm_model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Log metrics
    log_metrics(log_path, metrics)
    print(f"\nMetrics logged to: {log_path}")
