"""
Preprocessing Module for Fake News Detection (AG News Dataset)
===============================================================
This module cleans and normalizes the text data from the AG News dataset.
Preprocessing steps include:
- Converting text to lowercase
- Removing non-alphabetic characters (numbers, punctuation, symbols)
- Handling missing or non-string values
"""

import re


def preprocess(df):
    """
    Clean the text column of a DataFrame for NLP tasks.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'text' column with raw news articles.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with the 'text' column cleaned and transformed.
    """
    def clean_text(text):
        """
        Apply text cleaning to a single string.

        Steps:
        1. Convert to lowercase for case-insensitive processing.
        2. Remove any character that is NOT a letter (a-z) or whitespace.
           This eliminates numbers, punctuation, and special symbols,
           leaving only alphabetic tokens.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text containing only lowercase letters and spaces.
        """
        # Normalize case: all lowercase to reduce vocabulary size
        text = text.lower()
        # Remove non-alphabetic characters: keep only a-z and spaces
        # The pattern [^a-zA-Z\s] matches anything not a letter or whitespace
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    # Ensure the 'text' column exists and convert to string (handling potential NaNs)
    # Apply the cleaning function to every entry in the column
    df["text"] = df["text"].astype(str).apply(clean_text)

    return df

if __name__ == "__main__":
    print("Preprocessing handled in pipeline")