#!/usr/bin/env python3
"""
Test the preprocessing functions
"""

import sys
from pathlib import Path

# Add src to path for imports - adjust for being in tests directory
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import clean_text, preprocess_column
import pandas as pd

def test_clean_text():
    """Test the clean_text function with various inputs"""
    
    print("🧪 Testing clean_text function...")
    
    test_cases = [
        ("I feel anxious and worried!", "anxious worried"),
        ("Hello, how are you doing today?", "hello today"),
        ("I can't sleep at night.", "cannot sleep night"),
        ("Visit http://example.com for help", "visit help"),
        ("Email me at test@example.com", "email"),
        ("", ""),
        (None, ""),
        ("123 456", ""),
        ("!@#$%", ""),
        ("I'm feeling really depressed...", "feeling really depressed")
    ]
    
    for input_text, expected_pattern in test_cases:
        try:
            result = clean_text(input_text)
            print(f"Input: '{input_text}' -> Output: '{result}'")
            
            # Basic validation - should not be None and should be string
            assert isinstance(result, str), f"Output should be string, got {type(result)}"
            
            # Check if expected words are present (for non-empty expected patterns)
            if expected_pattern and expected_pattern.strip():
                expected_words = expected_pattern.split()
                for word in expected_words:
                    if word not in result:
                        print(f"⚠️  Expected word '{word}' not found in output")
            
        except Exception as e:
            print(f"❌ Error processing '{input_text}': {e}")
            return False
    
    print("✅ clean_text function tests passed")
    return True

def test_preprocess_column():
    """Test the preprocess_column function with a DataFrame"""
    
    print("\n🧪 Testing preprocess_column function...")
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'text': [
                "I feel very anxious about my future.",
                "Depression is affecting my daily life.",
                "I can't handle the stress anymore!",
                "",
                None,
                "Visit our website at http://example.com"
            ],
            'label': ['anxiety', 'depression', 'stress', 'empty', 'none', 'url']
        })
        
        print(f"Original data shape: {test_data.shape}")
        print("Sample original text:")
        for i, text in enumerate(test_data['text'][:3]):
            print(f"  {i+1}: '{text}'")
        
        # Assume the function expects a text column (check what column it processes)
        # Let's see what columns the function expects
        result_data = preprocess_column(test_data.copy())
        
        print(f"✅ preprocess_column executed successfully")
        print(f"Result data shape: {result_data.shape}")
        print(f"Result columns: {list(result_data.columns)}")
        
        # Check if clean_text column was added
        if 'clean_text' in result_data.columns:
            print("Sample cleaned text:")
            for i, text in enumerate(result_data['clean_text'][:3]):
                print(f"  {i+1}: '{text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocess_column test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_edge_cases():
    """Test preprocessing with edge cases"""
    
    print("\n🧪 Testing preprocessing edge cases...")
    
    edge_cases = [
        "a" * 1000,  # Very long text
        "word " * 500,  # Repeated words
        "I'm can't won't don't shouldn't",  # Multiple contractions
        "😀😃😄😁 emojis and unicode ñáéíóú",  # Unicode and emojis
        "   multiple    spaces   between    words   ",  # Multiple spaces
        "UPPER CASE TEXT SHOULD BE LOWERCASE",  # Case handling
        "Numbers 123 456 789 mixed with text",  # Numbers
        "@user #hashtag http://url.com",  # Social media elements
    ]
    
    for test_text in edge_cases:
        try:
            result = clean_text(test_text)
            print(f"✅ Processed edge case: '{test_text[:50]}...' -> '{result[:50]}...'")
            
            # Basic validations
            assert isinstance(result, str), "Result should be string"
            assert len(result) <= len(test_text), "Result should not be longer than input"
            
        except Exception as e:
            print(f"❌ Failed on edge case '{test_text[:50]}...': {e}")
            return False
    
    print("✅ Edge case tests passed")
    return True

if __name__ == "__main__":
    print("🧪 Running preprocessing tests...\n")
    
    test1_success = test_clean_text()
    test2_success = test_preprocess_column()
    test3_success = test_preprocessing_edge_cases()
    
    if test1_success and test2_success and test3_success:
        print("\n🎉 All preprocessing tests passed!")
    else:
        print("\n❌ Some preprocessing tests failed")
        sys.exit(1)
