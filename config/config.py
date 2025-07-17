"""
Configuration settings for the Mental Health Text Classifier.

This module contains all configuration parameters used throughout the application.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 3,
    "scoring_metric": "f1_weighted",
    "tuned_model_name": "tuned_lightgbm_model.pkl",
    "vectorizer_name": "tfidf_vectorizer.pkl",
    "label_encoder_name": "label_encoder.pkl"
}

# TF-IDF vectorizer parameters
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "lowercase": True,
    "min_df": 2,
    "max_df": 0.95
}

# Model hyperparameters
LIGHTGBM_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [15, 31]
}

# Data file mappings
DATA_FILES = {
    "Depression_Anxiety": "LD DA 1.csv",
    "Emotional_Loneliness": "LD EL1.csv", 
    "Panic_Family": "LD PF1.csv",
    "Trauma_Stress": "LD TS 1.csv"
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "🧠 Mental Health Text Classifier",
    "page_icon": "🧠",
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "app.log"
}
