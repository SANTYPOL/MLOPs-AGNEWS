"""
Data Ingestion Module for Fake News Detection (AG News Dataset)
================================================================
This module loads the AG News dataset from Hugging Face Datasets,
converts it to a pandas DataFrame, saves a local CSV snapshot for DVC,
and optionally subsamples the data for testing mode.
"""

from datasets import load_dataset
import pandas as pd
import os
from src.utils.logger import get_logger

# Initialize logger for tracking pipeline events
logger = get_logger()


def load_data(config):
    """
    Load and prepare the AG News dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config/config.yaml.
        Expected to contain:
            - training.test_mode : bool
            - training.max_samples : int (used if test_mode is True)

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded (and optionally subsampled) dataset
        with columns 'label' (int: 0-3) and 'text' (str).
    """
    # Load the AG News dataset from Hugging Face Hub
    # AG News has 4 classes: World, Sports, Business, Sci/Tech
    ds = load_dataset("ag_news")

    # Extract the training split as a pandas DataFrame
    df = pd.DataFrame(ds["train"])

    # Rename columns to standard names (label, text) - ensures consistency
    # Note: The original dataset already uses 'label' and 'text', but we keep
    # this step for clarity and potential future dataset changes.
    df = df.rename(columns={"label": "label", "text": "text"})

    # Save a local snapshot of the full dataset for DVC (Data Version Control)
    # This ensures reproducibility and allows DVC to track changes.
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/agnews.csv"

    if not os.path.exists(file_path):
        # Save only if the file does not already exist (avoid redundant writes)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved dataset snapshot to {file_path}")
    else:
        logger.info(f"Dataset already exists at {file_path}")

    # Apply test mode subsampling if enabled in configuration
    # This is useful for rapid prototyping and debugging on a smaller dataset.
    if config["training"]["test_mode"]:
        df = df.sample(config["training"]["max_samples"], random_state=42)
        logger.info(f"Test mode active: subsampled to {len(df)} rows")

    logger.info(f"Loaded {len(df)} samples from AG News")
    return df

if __name__ == "__main__":
    import yaml
    config = yaml.safe_load(open("config/config.yaml"))
    load_data(config)