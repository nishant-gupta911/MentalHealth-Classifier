"""
Text preprocessing utilities for the Mental Health Text Classifier.

This module provides comprehensive text cleaning and preprocessing functions
for mental health text data, including tokenization, stopword removal,
and various text normalization techniques.
"""

import re
import string
import logging
from typing import List, Optional, Union
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic stopwords list (avoiding NLTK dependency issues)
stop_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'would', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'feel', 'feeling', 'felt', 'think', 'thinking', 'thought',
    'like', 'really', 'would', 'could', 'should', 'might',
    'get', 'getting', 'got', 'make', 'making', 'made'
}


def clean_text(text: Union[str, None], 
               remove_stopwords: bool = True,
               apply_stemming: bool = False,
               min_word_length: int = 2) -> str:
    """
    Clean and preprocess a single text string.
    
    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove stopwords
        apply_stemming (bool): Whether to apply stemming
        min_word_length (int): Minimum word length to keep
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, and social media handles
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation except apostrophes (for contractions)
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Handle contractions (basic expansion)
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Simple tokenization (split on whitespace)
    tokens = text.split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        # Skip if too short
        if len(token) < min_word_length:
            continue
        # Skip if it's a stopword (if enabled)
        if remove_stopwords and token in stop_words:
            continue
        # Skip if it's purely numeric
        if token.isdigit():
            continue
        # Skip if it contains only punctuation
        if all(c in string.punctuation for c in token):
            continue
            
        # Simple stemming (remove common suffixes)
        if apply_stemming:
            # Basic suffix removal
            for suffix in ['ing', 'ed', 'er', 'est', 's']:
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    token = token[:-len(suffix)]
                    break
            
        filtered_tokens.append(token)
    
    return ' '.join(filtered_tokens)


def preprocess_column(data: pd.DataFrame, 
                     text_column: Optional[str] = None,
                     **kwargs) -> pd.DataFrame:
    """
    Preprocess text data in a pandas DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        text_column (str, optional): Name of the text column to process
        **kwargs: Additional arguments passed to clean_text function
        
    Returns:
        pd.DataFrame: DataFrame with added 'clean_text' column
        
    Raises:
        ValueError: If text column cannot be identified
    """
    logger.info("Starting text preprocessing...")
    
    # Auto-detect text column if not specified
    if text_column is None:
        potential_columns = ['text', 'body', 'post', 'content', 'message']
        text_column = None
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in potential_columns):
                text_column = col
                break
    
    if text_column is None or text_column not in data.columns:
        raise ValueError(
            f"Could not identify text column. Available columns: {list(data.columns)}"
        )
    
    logger.info(f"Processing text column: '{text_column}'")
    
    # Apply text cleaning
    data['clean_text'] = data[text_column].apply(
        lambda x: clean_text(x, **kwargs)
    )
    
    # Remove rows with empty cleaned text
    initial_length = len(data)
    data = data[data['clean_text'].str.strip() != ''].copy()
    final_length = len(data)
    
    if initial_length != final_length:
        logger.info(f"Removed {initial_length - final_length} rows with empty text")
    
    # Log preprocessing statistics
    avg_length_original = data[text_column].str.len().mean()
    avg_length_cleaned = data['clean_text'].str.len().mean()
    
    logger.info(f"Preprocessing completed:")
    logger.info(f"  - Average original text length: {avg_length_original:.1f}")
    logger.info(f"  - Average cleaned text length: {avg_length_cleaned:.1f}")
    logger.info(f"  - Final dataset size: {len(data)} samples")
    
    return data


def get_text_statistics(texts: List[str]) -> dict:
    """
    Calculate comprehensive statistics for a list of texts.
    
    Args:
        texts (List[str]): List of text strings
        
    Returns:
        dict: Dictionary containing text statistics
    """
    if not texts:
        return {}
    
    lengths = [len(text.split()) for text in texts if text]
    
    stats = {
        'total_texts': len(texts),
        'empty_texts': sum(1 for text in texts if not text.strip()),
        'word_count': {
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'total': sum(lengths)
        },
        'character_count': {
            'mean': sum(len(text) for text in texts) / len(texts),
            'total': sum(len(text) for text in texts)
        }
    }
    
    return stats


def preprocess_text(text: str) -> str:
    """
    Legacy function for backward compatibility.
    Simple text preprocessing for quick use.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    return clean_text(text, remove_stopwords=True, apply_stemming=False)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Re-join
    return " ".join(tokens)
