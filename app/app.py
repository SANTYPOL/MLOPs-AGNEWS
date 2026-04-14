"""
FastAPI Inference Server for Fake News Detection (AG News Dataset)
===================================================================
This module serves a pre-trained machine learning model (saved as 'model.pkl')
via a REST API endpoint. The model is a scikit-learn pipeline that expects
raw text input and returns a predicted class label (0-3 corresponding to
AG News categories: World, Sports, Business, Sci/Tech).
"""

from fastapi import FastAPI
import pickle

# Initialize the FastAPI application instance
app = FastAPI(
    title="Fake News Detection API",
    description="Predict the category of a news article using a trained ML model",
    version="1.0.0"
)

# Load the pre-trained model from disk
# The model is a scikit-learn Pipeline (TF-IDF + classifier) saved during training
# Path: models/model.pkl (created by trainer.py)
model = pickle.load(open("models/model.pkl", "rb"))


@app.post("/predict")
def predict(text: str):
    """
    Predict the category of a given news article text.

    Parameters
    ----------
    text : str
        The raw news article text to classify.

    Returns
    -------
    dict
        A dictionary containing the predicted class label as an integer.
        Example: {"prediction": 0}  (0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech)

    Notes
    -----
    - The input text is automatically preprocessed by the pipeline's TF-IDF vectorizer.
    - The model expects raw text (no manual cleaning required, as the pipeline
      includes the same preprocessing steps used during training).
    - The endpoint uses HTTP POST method; send JSON with key "text".
    """
    # Predict using the loaded model
    # model.predict() expects a list of strings; [text] makes it a single-element list
    # Returns an array of predictions; [0] extracts the first (only) prediction
    prediction = int(model.predict([text])[0])

    # Return the prediction as a JSON response
    return {"prediction": prediction}