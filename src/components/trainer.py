"""
FINAL Trainer (MLflow + DVC + Production Ready)
"""

import os
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


def train(df, config=None):

    # 🔥 CRITICAL FIX (DO NOT REMOVE)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("AGNEWS_PIPELINE")

    print("🚀 Training Models (NaiveBayes + SVM)...")

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "NaiveBayes": MultinomialNB(),
        "SVM": LinearSVC()
    }

    results = {}

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 🔥 MAIN RUN
    with mlflow.start_run(run_name="full_pipeline"):

        for name, model in tqdm(models.items(), desc="Training Models"):

            print(f"\n🔹 Training {name}...")

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000)),
                ("model", model)
            ])

            # 🔥 NESTED RUN (important)
            with mlflow.start_run(run_name=name, nested=True):

                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                results[name] = acc

                # ✅ Params + metrics
                mlflow.log_param("model", name)
                mlflow.log_metric("accuracy", acc)

                # ✅ Save model locally
                model_path = f"models/{name}.pkl"
                pickle.dump(pipe, open(model_path, "wb"))

                # ✅ Log model (VERY IMPORTANT)
                mlflow.sklearn.log_model(pipe, name)

                print(f"✅ {name} Accuracy: {acc:.4f}")

        # ------------------ Comparison Plot ------------------
        plt.figure(figsize=(6, 4))
        plt.bar(results.keys(), results.values())

        for i, v in enumerate(results.values()):
            plt.text(i, v, f"{v:.3f}", ha='center')

        plt.title("Model Comparison (Accuracy)")
        plt.ylim(0.8, 1.0)
        plt.tight_layout()

        comp_path = "artifacts/model_comparison.png"
        plt.savefig(comp_path)
        plt.close()

        mlflow.log_artifact(comp_path)

        # ------------------ Best Model ------------------
        best_model_name = max(results, key=results.get)
        best_model = pickle.load(open(f"models/{best_model_name}.pkl", "rb"))

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(f"Confusion Matrix ({best_model_name})")

        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        print("\n🏆 Best Model:", best_model_name)

    return best_model, X_test, y_test
