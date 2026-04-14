"""
Training Module (Simplified) for Fake News Detection
======================================================
This module trains two classical ML models (Multinomial Naive Bayes and Linear SVM)
on TF-IDF features, compares their accuracy, logs experiments with MLflow,
saves the best model, and generates a confusion matrix.
"""

import os
import pickle
import time
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm  # Added for progress bar

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def train(df, config=None):
    """
    Train Naive Bayes and SVM models on the preprocessed text data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'text' and 'label' columns.
    config : dict, optional
        Configuration dictionary (unused in this simplified version).

    Returns
    -------
    tuple
        (best_model, X_test, y_test) where best_model is the trained pipeline
        of the model with higher accuracy, and X_test, y_test are the test data.
    """
    print("🚀 Training Models (NaiveBayes + SVM)...")

    # Extract features and labels
    X = df['text']
    y = df['label']

    # Split into train (80%) and test (20%) with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Dictionary of models to train
    models = {
        "NaiveBayes": MultinomialNB(),   # Suitable for discrete features like TF-IDF counts
        "SVM": LinearSVC()                # Linear SVM works well with high-dimensional text data
    }

    results = {}  # Store accuracy for each model

    # Create directories for saved models and artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Set MLflow experiment (all runs will be grouped under 'agnews-exp')
    mlflow.set_experiment("agnews-exp")

    # Train each model with a progress bar
    for name, model in tqdm(models.items(), desc="Training Models", unit="model"):
        print(f"\n🔹 Training {name}...")

        # Build pipeline: TF-IDF vectorizer + classifier
        # max_features=5000 limits vocabulary size for faster training
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("model", model)
        ])

        start = time.time()

        # Start an MLflow run for this model
        with mlflow.start_run(run_name=name):
            # Train the pipeline
            pipe.fit(X_train, y_train)

            # Predict on test set and compute accuracy
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results[name] = acc

            # Log parameters and metrics to MLflow
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)

            # Save the trained pipeline as a pickle file
            model_path = f"models/{name}.pkl"
            pickle.dump(pipe, open(model_path, "wb"))
            mlflow.log_artifact(model_path)  # Upload to MLflow

            print(f"✅ {name} Accuracy: {acc:.4f}")

    # ------------------ Model Comparison Plot ------------------
    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), results.values())

    # Add accuracy values on top of bars
    for i, v in enumerate(results.values()):
        plt.text(i, v, f"{v:.3f}", ha='center')

    plt.title("Model Comparison (Accuracy)")
    plt.ylim(0.8, 1.0)  # Set y-axis limits for better visualization
    plt.tight_layout()

    comp_path = "artifacts/model_comparison.png"
    plt.savefig(comp_path)
    mlflow.log_artifact(comp_path)  # Log the plot
    plt.close()

    # ------------------ Confusion Matrix for Best Model ------------------
    # Select the model with highest accuracy
    best_model_name = max(results, key=results.get)
    # Load the best model from disk
    best_model = pickle.load(open(f"models/{best_model_name}.pkl", "rb"))

    # Generate predictions on test set using the best model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix ({best_model_name})")

    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    print("\n🏆 Best Model:", best_model_name)

    # Return the best model along with test data for further evaluation
    return best_model, X_test, y_test