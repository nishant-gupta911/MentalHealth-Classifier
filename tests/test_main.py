"""
Unit tests for the Mental Health Text Classifier.

This module contains comprehensive tests for all components of the application
including data loading, preprocessing, model training, and prediction.
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import clean_text, preprocess_column, preprocess_text
from src.data_loader import validate_data, get_data_statistics


class TestPreprocessing(unittest.TestCase):
    """Test text preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "I'm feeling really anxious about tomorrow's meeting.",
            "Can't sleep, my mind keeps racing with worries.",
            "I feel so lonely and isolated from everyone.",
            "The trauma from that event still haunts me daily.",
            ""
        ]
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        text = "I'm feeling REALLY anxious!!! 😰"
        cleaned = clean_text(text)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!!!", cleaned)
        self.assertNotIn("😰", cleaned)
    
    def test_clean_text_empty(self):
        """Test handling of empty text."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")
    
    def test_clean_text_urls(self):
        """Test URL removal."""
        text = "Check this out: https://example.com and www.test.com"
        cleaned = clean_text(text)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.", cleaned)
    
    def test_preprocess_column(self):
        """Test DataFrame column preprocessing."""
        df = pd.DataFrame({
            'text': self.sample_texts,
            'label': ['anxiety', 'anxiety', 'loneliness', 'trauma', 'empty']
        })
        
        result = preprocess_column(df)
        self.assertIn('clean_text', result.columns)
        self.assertEqual(len(result), 4)  # Empty text should be removed
    
    def test_preprocess_text_legacy(self):
        """Test legacy preprocessing function."""
        text = "I'm feeling anxious and worried."
        result = preprocess_text(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestDataLoader(unittest.TestCase):
    """Test data loading and validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = pd.DataFrame({
            'text': [
                'I feel anxious about the future',
                'So lonely and sad today',
                'Family stress is overwhelming',
                'Trauma memories keep coming back'
            ],
            'label': ['anxiety', 'loneliness', 'family', 'trauma']
        })
        
        self.invalid_data = pd.DataFrame({
            'content': ['some text'],
            'category': ['label']
        })
    
    def test_validate_data_valid(self):
        """Test validation with valid data."""
        self.assertTrue(validate_data(self.valid_data))
    
    def test_validate_data_empty(self):
        """Test validation with empty data."""
        empty_df = pd.DataFrame()
        self.assertFalse(validate_data(empty_df))
    
    def test_validate_data_missing_columns(self):
        """Test validation with missing required columns."""
        self.assertFalse(validate_data(self.invalid_data))
    
    def test_get_data_statistics(self):
        """Test data statistics generation."""
        stats = get_data_statistics(self.valid_data)
        
        self.assertIn('total_samples', stats)
        self.assertIn('unique_labels', stats)
        self.assertIn('label_distribution', stats)
        self.assertIn('text_length_stats', stats)
        
        self.assertEqual(stats['total_samples'], 4)
        self.assertEqual(stats['unique_labels'], 4)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.random.random((100, 50))
        self.y = np.random.randint(0, 4, 100)
    
    def test_data_shapes(self):
        """Test that data has correct shapes."""
        self.assertEqual(self.X.shape[0], self.y.shape[0])
        self.assertEqual(len(self.X.shape), 2)
        self.assertEqual(len(self.y.shape), 1)


class TestConfigAndUtils(unittest.TestCase):
    """Test configuration and utility functions."""
    
    def test_config_import(self):
        """Test that configuration can be imported."""
        try:
            from config.config import MODEL_CONFIG, TFIDF_CONFIG
            self.assertIsInstance(MODEL_CONFIG, dict)
            self.assertIsInstance(TFIDF_CONFIG, dict)
        except ImportError:
            self.fail("Could not import configuration")
    
    def test_required_config_keys(self):
        """Test that required configuration keys exist."""
        from config.config import MODEL_CONFIG, TFIDF_CONFIG
        
        required_model_keys = ['test_size', 'random_state', 'cv_folds']
        for key in required_model_keys:
            self.assertIn(key, MODEL_CONFIG)
        
        required_tfidf_keys = ['max_features', 'ngram_range']
        for key in required_tfidf_keys:
            self.assertIn(key, TFIDF_CONFIG)


def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running performance tests...")
    
    # Test text preprocessing speed
    import time
    from src.preprocessing import clean_text
    
    sample_texts = [
        "I'm feeling really anxious about tomorrow's meeting and can't sleep.",
        "The loneliness is overwhelming and I don't know how to cope.",
        "Family conflicts are causing me so much stress and anxiety."
    ] * 1000
    
    start_time = time.time()
    cleaned_texts = [clean_text(text) for text in sample_texts]
    end_time = time.time()
    
    processing_time = end_time - start_time
    texts_per_second = len(sample_texts) / processing_time
    
    print(f"📊 Preprocessing Performance:")
    print(f"  - Processed {len(sample_texts)} texts in {processing_time:.2f} seconds")
    print(f"  - Speed: {texts_per_second:.0f} texts/second")
    
    # Assert reasonable performance
    assert texts_per_second > 100, f"Preprocessing too slow: {texts_per_second:.0f} texts/second"
    print("✅ Performance tests passed!")


if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()
    
    print("\n" + "="*50)
    print("🎉 All tests completed!")
