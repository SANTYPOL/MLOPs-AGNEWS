"""
Evaluation Module for Fake News Detection (AG News Dataset)
=============================================================
This module evaluates a trained machine learning model on a test dataset.
It computes accuracy and weighted F1 score, generates a confusion matrix,
saves the plot as an artifact, logs metrics to MLflow, and prints a
classification report.
"""

import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate(model, X, y):
    """
    Evaluate a trained model and log results.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model with a predict() method.
    X : array-like or pandas DataFrame
        Feature matrix (e.g., TF-IDF vectors, embeddings).
    y : array-like or pandas Series
        True labels (0-3 for AG News classes).

    Returns
    -------
    float
        Accuracy of the model on the provided data.
    """
    # Generate predictions on the test set
    preds = model.predict(X)

    # Calculate evaluation metrics
    # Accuracy: overall correctness
    acc = accuracy_score(y, preds)
    # Weighted F1: handles class imbalance by averaging per-class F1 scores
    # weighted by the number of true instances in each class
    f1 = f1_score(y, preds, average="weighted")

    # Print summary metrics to console
    print(f"\nFinal Accuracy: {acc:.4f}, F1: {f1:.4f}\n")

    # Compute confusion matrix to visualize misclassifications
    cm = confusion_matrix(y, preds)

    # Create directory for saving artifacts (if it doesn't exist)
    os.makedirs("artifacts", exist_ok=True)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    # Add title with metrics for quick reference
    plt.title(f"Confusion Matrix\nAccuracy: {acc:.3f} | F1: {f1:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the plot to disk
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()  # Close the figure to free memory

    print("📊 Confusion Matrix saved at:", cm_path)
    print("\n📄 Classification Report:\n")
    print(classification_report(y, preds))

    # Log metrics and artifact to MLflow for experiment tracking
    mlflow.log_metric("final_accuracy", acc)
    mlflow.log_metric("final_f1", f1)
    mlflow.log_artifact(cm_path)

    return acc

if __name__ == "__main__":
    print("Evaluation handled in pipeline")