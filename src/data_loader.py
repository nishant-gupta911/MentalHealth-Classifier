"""
Data loading and preprocessing utilities for the Mental Health Text Classifier.

This module handles loading, combining, and basic preprocessing of mental health
text data from multiple CSV files.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
from config.config import DATA_DIR, DATA_FILES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_combine(save_combined: bool = True) -> pd.DataFrame:
    """
    Load and combine mental health text data from multiple CSV files.
    
    Args:
        save_combined (bool): Whether to save the combined dataset to CSV
        
    Returns:
        pd.DataFrame: Combined dataset with text and labels
        
    Raises:
        FileNotFoundError: If any of the required data files are missing
        ValueError: If the loaded data is empty or invalid
    """
    logger.info("Starting data loading and combination process...")
    
    dfs = []
    missing_files = []
    
    for label, filename in DATA_FILES.items():
        file_path = DATA_DIR / filename
        
        try:
            if not file_path.exists():
                missing_files.append(str(file_path))
                continue
                
            logger.info(f"Loading {filename} for label: {label}")
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Warning: {filename} is empty")
                continue
                
            df['label'] = label
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records for {label}")
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            continue
    
    if missing_files:
        raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    if not dfs:
        raise ValueError("No valid data files could be loaded")
    
    # Combine all dataframes
    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset shape: {data.shape}")
    
    # Basic data validation
    if data.empty:
        raise ValueError("Combined dataset is empty")
    
    # Check for required columns
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing expected columns: {missing_columns}")
    
    # Save combined data if requested
    if save_combined:
        output_path = DATA_DIR / "combined.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"Combined data saved to: {output_path}")
    
    # Display label distribution
    label_counts = data['label'].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    
    return data


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate the loaded dataset for completeness and quality.
    
    Args:
        data (pd.DataFrame): Dataset to validate
        
    Returns:
        bool: True if data passes validation, False otherwise
    """
    logger.info("Validating dataset...")
    
    # Check for empty dataset
    if data.empty:
        logger.error("Dataset is empty")
        return False
    
    # Check for required columns
    required_columns = ['text', 'label']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Missing required columns. Expected: {required_columns}")
        return False
    
    # Check for null values
    null_counts = data.isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
    
    # Check text column
    empty_text = data['text'].isna() | (data['text'].str.strip() == '')
    if empty_text.any():
        logger.warning(f"Found {empty_text.sum()} empty text entries")
    
    # Check label distribution
    label_counts = data['label'].value_counts()
    min_samples = label_counts.min()
    if min_samples < 10:
        logger.warning(f"Some labels have very few samples (min: {min_samples})")
    
    logger.info("Dataset validation completed")
    return True


def get_data_statistics(data: pd.DataFrame) -> Dict:
    """
    Generate comprehensive statistics about the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
        
    Returns:
        Dict: Dictionary containing various dataset statistics
    """
    stats = {
        'total_samples': len(data),
        'unique_labels': data['label'].nunique(),
        'label_distribution': data['label'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': data['text'].str.len().mean(),
            'median': data['text'].str.len().median(),
            'min': data['text'].str.len().min(),
            'max': data['text'].str.len().max()
        },
        'missing_values': data.isnull().sum().to_dict()
    }
    
    return stats
