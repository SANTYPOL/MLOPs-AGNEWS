"""
Fake News Detection using Machine Learning - AG News Dataset
=============================================================
This script implements an end-to-end MLOps pipeline for detecting fake news.
It uses the AG News dataset (loaded via configuration) and performs:
- Data ingestion
- Preprocessing
- Model training
- Evaluation
"""

# Configuration file & progress bars
import yaml
from tqdm import tqdm

# Import custom pipeline components
from src.components.data_ingestion import load_data
from src.components.preprocessing import preprocess
from src.components.trainer import train
from src.components.evaluator import evaluate


def run():
    """
    Execute the complete MLOps pipeline for fake news detection.
    
    Steps:
    1. Load configuration from YAML file.
    2. Sequentially run data ingestion, preprocessing, training, and evaluation.
    3. Display progress using tqdm progress bars.
    """
    print("\n🚀 Starting MLOps Pipeline...\n")

    # Load pipeline configuration (e.g., paths, hyperparameters, model settings)
    config = yaml.safe_load(open("config/config.yaml"))

    # Define pipeline stages in order
    stages = [
        "Data Ingestion",
        "Preprocessing",
        "Training",
        "Evaluation"
    ]

    # Iterate through stages with a progress bar
    for stage in tqdm(stages, desc="Pipeline Progress"):

        if stage == "Data Ingestion":
            print("\n📥 Loading Data...")
            # Load raw data from source (e.g., AG News dataset)
            df = load_data(config)

        elif stage == "Preprocessing":
            print("🧹 Preprocessing...")
            # Clean, tokenize, and transform text data for ML models
            df = preprocess(df)

        elif stage == "Training":
            print("🤖 Training Models...")
            # Train multiple models, select the best, and return it along with test data
            model, X_test, y_test = train(df, config)

        elif stage == "Evaluation":
            print("📊 Evaluating Best Model...")
            # Evaluate the best model on the test set (accuracy, precision, recall, etc.)
            evaluate(model, X_test, y_test)

    print("\n✅ Pipeline Completed Successfully!\n")


if __name__ == "__main__":
    # Entry point: run the pipeline when the script is executed directly
    run()