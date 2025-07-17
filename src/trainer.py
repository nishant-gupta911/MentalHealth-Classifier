"""
Model training and evaluation utilities for the Mental Health Text Classifier.

This module provides comprehensive training pipeline including model selection,
hyperparameter tuning, evaluation, and performance comparison.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
from pathlib import Path

from src.models import get_models
from src.evaluator import evaluate, plot_confusion
from lightgbm import LGBMClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate(X, y):
    """
    Main training and evaluation function with enhanced logging and error handling.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        
    Returns:
        Dict: Information about the best performing model
    """
    logger.info("🚀 Starting model training and evaluation")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Testing set: {X_test.shape[0]} samples")
    
    # Tune LightGBM first
    tuned_model = tune_lightgbm(X_train, y_train)

    models = get_models()
    # Add the tuned model to comparison
    models["LightGBM (Tuned)"] = tuned_model
    
    results = []
    best_f1 = 0.0
    best_model_name = None

    for name, model in models.items():
        try:
            logger.info(f"🔄 Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name

            logger.info(f"📊 {name} - F1 Score: {f1:.4f}")
            
            evaluate(y_test, y_pred, name)
            plot_confusion(y_test, y_pred, np.unique(y), name)

            # Save model with proper path
            model_filename = f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, f"models/{model_filename}")
            logger.info(f"💾 {name} model saved")
            
        except Exception as e:
            logger.error(f"❌ Error training {name}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values(by="F1-score", ascending=False)
    logger.info("\n🏆 Model Performance Comparison:")
    logger.info("\n" + df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    df.to_csv("models/model_comparison_results.csv", index=False)
    
    return {
        'name': best_model_name,
        'f1_score': best_f1,
        'results_df': df
    }


def tune_lightgbm(X_train, y_train):
    """
    Enhanced LightGBM hyperparameter tuning with better logging.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained LightGBM model
    """
    logger.info("🎯 Tuning LightGBM with GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31]
    }

    lgbm = LGBMClassifier(force_col_wise=True, random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        lgbm, param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    try:
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✅ Best Parameters: {grid_search.best_params_}")
        logger.info(f"🏆 Best CV F1 Score: {grid_search.best_score_:.4f}")

        # Save the best model
        joblib.dump(grid_search.best_estimator_, "models/tuned_lightgbm_model.pkl")
        logger.info("💾 Best LightGBM model saved as 'models/tuned_lightgbm_model.pkl'")
        
        return grid_search.best_estimator_
        
    except Exception as e:
        logger.error(f"❌ Error during LightGBM tuning: {str(e)}")
        # Return a default LightGBM model if tuning fails
        default_model = LGBMClassifier(force_col_wise=True, random_state=42, verbose=-1)
        default_model.fit(X_train, y_train)
        joblib.dump(default_model, "models/tuned_lightgbm_model.pkl")
        return default_model
