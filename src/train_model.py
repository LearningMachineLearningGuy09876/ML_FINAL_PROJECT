"""
Train and save the FlikPik hybrid recommender model.

Run from the project root:
    python src/train_model.py

This creates a compressed model file:
    models/hybrid_recommender.pkl.gz

The compressed file is smaller and easier to manage than a regular .pkl file.
You usually should not commit large model files to GitHub. Instead, keep this
script in the repository and regenerate the model when needed.
"""

import gzip
import os
import pickle
import sys
from pathlib import Path

# Make sure src imports work whether this file is run from the root or src folder.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import MODEL_PATH, RECOMMENDER_PARAMS
from data_loader import load_processed_data
from recommender import HybridRecommender


COMPRESSED_MODEL_PATH = f"{MODEL_PATH}.gz"


def save_compressed_model(model, path=COMPRESSED_MODEL_PATH):
    """Save the trained recommender as a compressed pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with gzip.open(path, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def train_and_save_model():
    """Load processed data, train the recommender, and save it as .pkl.gz."""
    print("Loading processed data...")
    train, val, test, movies, genres = load_processed_data()

    print(f"Training rows: {len(train):,}")
    print(f"Movies: {len(movies):,}")
    print(f"Saving compressed model to: {COMPRESSED_MODEL_PATH}")

    model = HybridRecommender(
        train_df=train,
        movies_df=movies,
        genre_df=genres,
        **RECOMMENDER_PARAMS,
    )

    print("Training hybrid recommender...")
    model.fit()

    saved_path = save_compressed_model(model)

    print("Model training complete.")
    print(f"Saved model: {saved_path}")
    return model


if __name__ == "__main__":
    train_and_save_model()
