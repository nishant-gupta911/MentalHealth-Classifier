"""
Text vectorization utilities for the Mental Health Text Classifier.

This module provides various text vectorization methods including TF-IDF,
with optimized parameters for mental health text classification.
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from pathlib import Path
from config.config import TFIDF_CONFIG, MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextVectorizer:
    """
    Advanced text vectorization class with multiple vectorization options.
    """
    
    def __init__(self, vectorizer_type: str = 'tfidf', **kwargs):
        """
        Initialize the TextVectorizer.
        
        Args:
            vectorizer_type (str): Type of vectorizer ('tfidf', 'count', 'lda')
            **kwargs: Additional parameters for the vectorizer
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None
        self.is_fitted = False
        
        if vectorizer_type == 'tfidf':
            config = {**TFIDF_CONFIG, **kwargs}
            self.vectorizer = TfidfVectorizer(**config)
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**kwargs)
        elif vectorizer_type == 'lda':
            # For LDA, we need a count vectorizer first
            self.count_vectorizer = CountVectorizer(**kwargs)
            n_components = kwargs.get('n_components', 10)
            self.vectorizer = LatentDirichletAllocation(
                n_components=n_components,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: Vectorized text features
        """
        logger.info(f"Fitting {self.vectorizer_type} vectorizer on {len(texts)} documents...")
        
        if self.vectorizer_type == 'lda':
            # For LDA, first apply count vectorization
            count_matrix = self.count_vectorizer.fit_transform(texts)
            features = self.vectorizer.fit_transform(count_matrix)
        else:
            features = self.vectorizer.fit_transform(texts)
        
        self.is_fitted = True
        logger.info(f"Vectorization completed. Feature shape: {features.shape}")
        
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: Vectorized text features
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        if self.vectorizer_type == 'lda':
            count_matrix = self.count_vectorizer.transform(texts)
            features = self.vectorizer.transform(count_matrix)
        else:
            features = self.vectorizer.transform(texts)
        
        return features
    
    def save(self, filepath: str):
        """Save the fitted vectorizer to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        save_data = {
            'vectorizer_type': self.vectorizer_type,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }
        
        if self.vectorizer_type == 'lda':
            save_data['count_vectorizer'] = self.count_vectorizer
        
        joblib.dump(save_data, filepath)
        logger.info(f"Vectorizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted vectorizer from disk."""
        save_data = joblib.load(filepath)
        
        instance = cls(save_data['vectorizer_type'])
        instance.vectorizer = save_data['vectorizer']
        instance.is_fitted = save_data['is_fitted']
        
        if save_data['vectorizer_type'] == 'lda':
            instance.count_vectorizer = save_data['count_vectorizer']
        
        logger.info(f"Vectorizer loaded from {filepath}")
        return instance


def get_tfidf_features(texts: List[str], 
                      max_features: Optional[int] = None,
                      ngram_range: Tuple[int, int] = (1, 2),
                      save: bool = True,
                      save_path: Optional[str] = None) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF features from text data with optimized parameters.
    
    Args:
        texts (List[str]): List of text documents
        max_features (int, optional): Maximum number of features
        ngram_range (Tuple[int, int]): Range of n-grams to extract
        save (bool): Whether to save the fitted vectorizer
        save_path (str, optional): Path to save the vectorizer
        
    Returns:
        Tuple[np.ndarray, TfidfVectorizer]: Feature matrix and fitted vectorizer
    """
    logger.info("Creating TF-IDF features...")
    
    # Use config defaults if not specified
    if max_features is None:
        max_features = TFIDF_CONFIG['max_features']
    
    # Initialize TF-IDF vectorizer with optimized parameters
    tfidf_params = {
        'max_features': max_features,
        'ngram_range': ngram_range,
        'stop_words': 'english',
        'lowercase': True,
        'min_df': 2,  # Ignore terms that appear in less than 2 documents
        'max_df': 0.95,  # Ignore terms that appear in more than 95% of documents
        'sublinear_tf': True,  # Apply sublinear tf scaling
        'use_idf': True,
        'smooth_idf': True
    }
    
    tfidf = TfidfVectorizer(**tfidf_params)
    X = tfidf.fit_transform(texts)
    
    logger.info(f"TF-IDF vectorization completed:")
    logger.info(f"  - Feature matrix shape: {X.shape}")
    logger.info(f"  - Vocabulary size: {len(tfidf.vocabulary_)}")
    logger.info(f"  - Feature density: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
    
    # Save the fitted vectorizer
    if save:
        if save_path is None:
            save_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        else:
            save_path = Path(save_path)
        
        joblib.dump(tfidf, save_path)
        logger.info(f"TF-IDF vectorizer saved to {save_path}")
    
    return X, tfidf


def analyze_feature_importance(vectorizer: TfidfVectorizer, 
                             feature_matrix: np.ndarray,
                             top_n: int = 20) -> dict:
    """
    Analyze the most important features in the TF-IDF matrix.
    
    Args:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        feature_matrix (np.ndarray): TF-IDF feature matrix
        top_n (int): Number of top features to return
        
    Returns:
        dict: Dictionary with feature importance analysis
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate mean TF-IDF scores for each feature
    mean_scores = np.array(feature_matrix.mean(axis=0)).flatten()
    
    # Get top features by mean score
    top_indices = np.argsort(mean_scores)[-top_n:][::-1]
    top_features = [(feature_names[i], mean_scores[i]) for i in top_indices]
    
    analysis = {
        'total_features': len(feature_names),
        'top_features': top_features,
        'vocabulary_size': len(vectorizer.vocabulary_),
        'mean_tfidf_score': mean_scores.mean(),
        'max_tfidf_score': mean_scores.max()
    }
    
    return analysis
