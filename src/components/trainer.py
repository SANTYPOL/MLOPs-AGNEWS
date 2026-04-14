"""
Training Module for Fake News Detection (AG News Dataset)
===========================================================
This module trains multiple models (classical ML + BERT) on the AG News dataset.
It performs:
- Train-test split with stratification
- TF-IDF vectorization + classical classifiers (LogReg, NaiveBayes, SVM, RandomForest)
- BERT fine-tuning using Hugging Face Transformers
- Model comparison and selection of the best classical model
- Logging metrics and artifacts to MLflow
- Saving the best classical model as a pickle file
"""

import os
import pickle
import time
import mlflow
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# ------------------ Helper: Model Comparison Plot ------------------
def plot_model_comparison(results):
    """
    Generate a bar chart comparing model accuracies and save it as an image.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names (str) to accuracy scores (float).

    Returns
    -------
    str
        File path where the plot image is saved.
    """
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    models = list(results.keys())
    scores = list(results.values())

    # Create bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, scores)

    # Add accuracy values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 yval + 0.005,
                 f"{yval:.3f}",
                 ha='center')

    plt.title("Model Comparison (Accuracy)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # Save and close plot
    path = "artifacts/model_comparison.png"
    plt.savefig(path)
    plt.close()

    return path


# ------------------ BERT Fine-Tuning ------------------
def train_bert(X_train, X_test, y_train, y_test):
    """
    Fine-tune a BERT-base-uncased model for sequence classification (4 classes).

    Parameters
    ----------
    X_train : array-like
        Training texts.
    X_test : array-like
        Test texts.
    y_train : array-like
        Training labels.
    y_test : array-like
        Test labels.

    Returns
    -------
    float
        Accuracy of the fine-tuned BERT model on the test set.
    """
    # Detect available hardware (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔥 BERT using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Convert pandas Series/arrays to Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({
        "text": list(X_train),
        "label": list(y_train)
    })

    test_dataset = Dataset.from_dict({
        "text": list(X_test),
        "label": list(y_test)
    })

    # Tokenization function: truncates/pads to max_length=128
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Apply tokenization to both datasets (batched for speed)
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Set format to PyTorch tensors with required columns
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load pre-trained BERT model with 4 output classes (AG News classes)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )
    model.to(device)

    # Configure training hyperparameters
    training_args = TrainingArguments(
        output_dir="./bert_results",      # Directory for checkpoints (not saved)
        num_train_epochs=2,               # Small number for demo; increase for production
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",      # Evaluate at the end of each epoch
        logging_steps=50,
        save_strategy="no",               # Do not save intermediate checkpoints
        fp16=True,                        # Mixed precision training (faster on GPU)
        report_to="none"                  # Disable external logging (we use MLflow separately)
    )

    # Metric function for evaluation
    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    # Create Trainer and start fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()          # Evaluate on test set

    return metrics["eval_accuracy"]


# ------------------ Main Training Function ------------------
def train(df, config):
    """
    Orchestrate the complete training process:
    1. Split data into train/test (stratified, 80/20).
    2. Train classical ML models with TF-IDF.
    3. Fine-tune BERT.
    4. Compare models and select the best classical model.
    5. Log metrics, plots, and save the best model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'text' and 'label' columns (preprocessed).
    config : dict
        Configuration dictionary loaded from config/config.yaml.
        Expected to contain:
            - mlflow.experiment : str (MLflow experiment name)

    Returns
    -------
    tuple
        (best_model, X_test, y_test) where best_model is the best classical
        sklearn pipeline (TF-IDF + classifier), and X_test, y_test are the
        test features and labels.
    """
    print("\n⚙️ Splitting Data...\n")

    # Extract features and labels
    X = df["text"]
    y = df["label"]

    # Stratified split ensures class proportions are preserved in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Dictionary of classical models to evaluate
    models = {
        "LogReg": LogisticRegression(max_iter=500),
        "NaiveBayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "RandomForest": RandomForestClassifier(n_estimators=200)
    }

    # Store results and track the best classical model
    results = {}
    best_model = None
    best_score = 0

    # Set MLflow experiment (all runs will be grouped under this experiment)
    mlflow.set_experiment(config["mlflow"]["experiment"])

    print("\n🤖 Training Classical Models...\n")

    # Train each classical model with a progress bar
    for name, clf in tqdm(models.items(), desc="Classical Models"):

        start = time.time()

        # Start an MLflow run for this model
        with mlflow.start_run(run_name=name):

            # Build pipeline: TF-IDF vectorizer + classifier
            # TF-IDF parameters: max 10k features, unigrams + bigrams, remove English stopwords
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ("clf", clf)
            ])

            # Train pipeline
            pipe.fit(X_train, y_train)
            # Evaluate on test set
            score = pipe.score(X_test, y_test)

            # Store results
            results[name] = score

            # Log parameters and metrics to MLflow
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", score)
            mlflow.log_metric("train_time", time.time() - start)

            print(f"{name} → Accuracy: {score:.4f}")

            # Update best classical model if this one is better
            if score > best_score:
                best_score = score
                best_model = pipe

    # Train BERT (deep learning model)
    print("\n🚀 Training BERT (GPU)...\n")
    bert_acc = train_bert(X_train, X_test, y_train, y_test)

    # Add BERT to results dictionary
    results["BERT"] = bert_acc
    print(f"BERT → Accuracy: {bert_acc:.4f}")

    # Generate and save model comparison plot
    plot_path = plot_model_comparison(results)
    # Log the plot as an artifact in the parent MLflow run (if active)
    # Note: This assumes an active run exists; in a full pipeline you may start a parent run.
    # For simplicity, we log it without starting a new run (may be picked up by outer context).
    mlflow.log_artifact(plot_path)

    # Save the best classical model to disk as a pickle file
    os.makedirs("models", exist_ok=True)
    pickle.dump(best_model, open("models/model.pkl", "wb"))

    print(f"\n🏆 Best Classical Model Accuracy: {best_score:.4f}")

    # Return best classical model and test data (for evaluation)
    return best_model, X_test, y_test