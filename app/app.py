"""
FastAPI Inference Server with Multiple Model Support (SVM / NaiveBayes)
========================================================================
This module serves two pre-trained scikit-learn pipelines (SVM and NaiveBayes)
via a REST API. Clients can specify which model to use for prediction.
Includes error handling for invalid model names and missing files.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

# -----------------------------
# App Initialization
# -----------------------------
# Create FastAPI instance with metadata for auto-generated documentation
app = FastAPI(
    title="AG News Classification API",
    description="Predict news category using trained ML models (SVM / NaiveBayes)",
    version="1.0.0"
)

# -----------------------------
# Input Schema (Request Body)
# -----------------------------
class InputText(BaseModel):
    """
    Pydantic model for the prediction request.

    Attributes
    ----------
    text : str
        The news article text to classify.
    model_name : str, optional
        Which model to use: "svm" (default) or "nb" (NaiveBayes).
    """
    text: str
    model_name: str = "svm"   # default to SVM if not specified

# -----------------------------
# Label Mapping (Numeric -> Category)
# -----------------------------
# AG News dataset has 4 classes:
# 0: World, 1: Sports, 2: Business, 3: Sci/Tech
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# -----------------------------
# Model Loader Helper Function
# -----------------------------
def load_model(name: str):
    """
    Load a trained model from disk based on the requested name.

    Parameters
    ----------
    name : str
        Model identifier: "svm" or "nb".

    Returns
    -------
    model : sklearn Pipeline
        The loaded pipeline (TF-IDF + classifier).

    Raises
    ------
    HTTPException
        - 400 if model name is invalid.
        - 500 if the model file does not exist on disk.
    """
    # Map logical names to actual file paths
    model_paths = {
        "svm": "models/SVM.pkl",
        "nb": "models/NaiveBayes.pkl"
    }

    # Validate model name
    if name not in model_paths:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Use 'svm' or 'nb'."
        )

    path = model_paths[name]

    # Check if file exists to avoid cryptic pickle errors
    if not os.path.exists(path):
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {path}"
        )

    # Load and return the model
    return pickle.load(open(path, "rb"))

# -----------------------------
# API Routes (Endpoints)
# -----------------------------

@app.get("/")
def home():
    """
    Health check / root endpoint.
    Returns a simple message to confirm the API is running.
    """
    return {"message": "🚀 MLOps AG News API is running!"}


@app.post("/predict")
def predict(input: InputText):
    """
    Predict the category of a news article.

    Request Body (JSON):
    {
        "text": "Your news article here...",
        "model_name": "svm"   # optional, defaults to "svm"
    }

    Response (JSON):
    {
        "prediction": 3,
        "category": "Sci/Tech",
        "model_used": "svm"
    }

    Error Responses:
    - 400: Invalid model name.
    - 500: Model file missing or other internal error.
    """
    try:
        # Load the requested model (may raise HTTPException)
        model = load_model(input.model_name)

        # Perform prediction
        # model.predict() expects a list of strings; wrap input.text in a list
        pred = int(model.predict([input.text])[0])

        # Return prediction along with metadata
        return {
            "prediction": pred,
            "category": label_map[pred],
            "model_used": input.model_name
        }

    except HTTPException:
        # Re-raise HTTP exceptions (they already have appropriate status codes)
        raise
    except Exception as e:
        # Catch any unexpected errors and return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))