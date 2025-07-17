"""
Mental Health Text Classifier - Standalone Prediction Module

A command-line interface for making predictions using the trained mental health
text classification model. Supports both single predictions and batch processing.

Author: [Your Name]
Date: July 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd

from src.preprocessing import clean_text
from config.config import MODELS_DIR, MODEL_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class MentalHealthPredictor:
    """
    Advanced mental health text classifier with confidence scoring and error handling.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor with trained models.
        
        Args:
            model_path (str, optional): Path to model directory
        """
        self.model_path = Path(model_path) if model_path else MODELS_DIR
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> bool:
        """
        Load trained model artifacts from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_file = self.model_path / MODEL_CONFIG['tuned_model_name']
            vectorizer_file = self.model_path / MODEL_CONFIG['vectorizer_name']
            encoder_file = self.model_path / MODEL_CONFIG['label_encoder_name']
            
            # Check if files exist
            missing_files = []
            for file_path, name in [(model_file, 'model'), (vectorizer_file, 'vectorizer'), (encoder_file, 'encoder')]:
                if not file_path.exists():
                    missing_files.append(f"{name}: {file_path}")
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                return False
            
            # Load artifacts
            self.model = joblib.load(model_file)
            self.vectorizer = joblib.load(vectorizer_file)
            self.label_encoder = joblib.load(encoder_file)
            
            self.is_loaded = True
            logger.info("✅ Successfully loaded all model artifacts")
            
            # Log model info
            logger.info(f"📊 Model Info:")
            logger.info(f"  - Labels: {list(self.label_encoder.classes_)}")
            logger.info(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model artifacts: {str(e)}")
            return False
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Make a prediction on a single text input.
        
        Args:
            text (str): Input text to classify
            return_probabilities (bool): Whether to return probability scores
            
        Returns:
            Dict: Prediction results with confidence scores
        """
        if not self.is_loaded:
            return {"error": "Model not loaded properly"}
        
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            # Preprocess text
            clean_text_input = clean_text(text)
            
            if not clean_text_input.strip():
                return {"error": "Text is empty after preprocessing"}
            
            # Vectorize
            X = self.vectorizer.transform([clean_text_input])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            result = {
                "predicted_label": predicted_label,
                "original_text": text,
                "cleaned_text": clean_text_input,
                "prediction_index": int(prediction)
            }
            
            # Add probability scores if requested
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence_scores = dict(zip(self.label_encoder.classes_, probabilities))
                result["confidence_scores"] = confidence_scores
                result["max_confidence"] = float(max(probabilities))
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = True) -> List[Dict]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts (List[str]): List of texts to classify
            return_probabilities (bool): Whether to return probability scores
            
        Returns:
            List[Dict]: List of prediction results
        """
        if not self.is_loaded:
            return [{"error": "Model not loaded properly"}] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.predict_single(text, return_probabilities)
            result["index"] = i
            results.append(result)
        
        return results
    
    def analyze_text(self, text: str) -> Dict:
        """
        Provide comprehensive analysis of the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        prediction_result = self.predict_single(text, return_probabilities=True)
        
        if "error" in prediction_result:
            return prediction_result
        
        # Add text analysis
        words = text.split()
        analysis = {
            **prediction_result,
            "text_analysis": {
                "word_count": len(words),
                "character_count": len(text),
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
            }
        }
        
        return analysis


def preprocess_text(text: str) -> str:
    """
    Legacy preprocessing function for backward compatibility.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    return clean_text(text)


def load_artifacts() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """
    Legacy function to load model artifacts.
    
    Returns:
        Tuple: model, vectorizer, label_encoder
    """
    predictor = MentalHealthPredictor()
    if predictor.is_loaded:
        return predictor.model, predictor.vectorizer, predictor.label_encoder
    return None, None, None


def predict_text(text: str) -> None:
    """
    Legacy function for command-line prediction.
    
    Args:
        text (str): Input text to classify
    """
    predictor = MentalHealthPredictor()
    if not predictor.is_loaded:
        print("❌ Error: Could not load model artifacts")
        return
    
    result = predictor.analyze_text(text)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n🔮 Predicted Mental Health Category: {result['predicted_label']}")
    print(f"🎯 Confidence: {result['max_confidence']:.1%}")
    
    if "confidence_scores" in result:
        print("\n📊 Detailed Confidence Scores:")
        for label, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {label}: {score:.1%}")
    
    print(f"\n📝 Text Analysis:")
    analysis = result['text_analysis']
    print(f"  - Words: {analysis['word_count']}")
    print(f"  - Characters: {analysis['character_count']}")
    print(f"  - Avg word length: {analysis['average_word_length']:.1f}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Mental Health Text Classifier - Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --text "I'm feeling anxious about tomorrow"
  python predict.py --file input_texts.txt
  python predict.py --interactive
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Text to classify'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing texts to classify (one per line)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode for multiple predictions'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (CSV format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize predictor
    predictor = MentalHealthPredictor()
    if not predictor.is_loaded:
        print("❌ Error: Could not load model artifacts")
        print("Please ensure you have run the training pipeline: python main.py")
        sys.exit(1)
    
    # Handle different input modes
    if args.text:
        # Single text prediction
        result = predictor.analyze_text(args.text)
        print_prediction_result(result)
        
    elif args.file:
        # Batch prediction from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"📂 Processing {len(texts)} texts from {args.file}")
            results = predictor.predict_batch(texts)
            
            # Print results
            for i, result in enumerate(results):
                print(f"\n--- Text {i+1} ---")
                print_prediction_result(result)
            
            # Save to CSV if requested
            if args.output:
                save_results_to_csv(results, args.output)
                
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            sys.exit(1)
    
    elif args.interactive:
        # Interactive mode
        print("🎯 Interactive Mental Health Text Classifier")
        print("Type your text and press Enter. Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n📝 Enter your text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("⚠️ Please enter some text.")
                    continue
                
                result = predictor.analyze_text(text)
                print_prediction_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    else:
        # Default: ask for input
        text = input("📝 Enter your text: ")
        predict_text(text)


def print_prediction_result(result: Dict) -> None:
    """Print formatted prediction result."""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🔮 Predicted Category: {result['predicted_label']}")
    
    if "max_confidence" in result:
        print(f"🎯 Confidence: {result['max_confidence']:.1%}")
    
    if "confidence_scores" in result:
        print("📊 All Scores:")
        for label, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {label}: {score:.1%}")


def save_results_to_csv(results: List[Dict], output_file: str) -> None:
    """Save prediction results to CSV file."""
    try:
        # Prepare data for CSV
        csv_data = []
        for result in results:
            if "error" not in result:
                row = {
                    'original_text': result.get('original_text', ''),
                    'predicted_label': result.get('predicted_label', ''),
                    'max_confidence': result.get('max_confidence', 0),
                }
                
                # Add confidence scores as separate columns
                if 'confidence_scores' in result:
                    for label, score in result['confidence_scores'].items():
                        row[f'confidence_{label}'] = score
                
                csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"💾 Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")


if __name__ == "__main__":
    main()
