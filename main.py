"""
Mental Health Text Classifier - Main Training Pipeline

This module orchestrates the complete machine learning pipeline for mental health
text classification, including data loading, preprocessing, feature extraction,
model training, and evaluation.

Author: [Your Name]
Date: July 2025
"""

import logging
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import load_and_combine, validate_data, get_data_statistics
from src.preprocessing import preprocess_column
from src.vectorizer import get_tfidf_features
from src.trainer import train_and_evaluate
from sklearn.preprocessing import LabelEncoder
import joblib
from config.config import MODELS_DIR, MODEL_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    MODELS_DIR.mkdir(exist_ok=True)
    logger.info(f"Models directory: {MODELS_DIR}")


def main(args=None):
    """
    Main training pipeline for the mental health text classifier.
    
    Args:
        args: Command line arguments (optional)
    """
    logger.info("🚀 Starting Mental Health Text Classifier Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Setup
        setup_directories()
        
        # Step 1: Data Loading
        logger.info("📥 Step 1: Loading and combining data...")
        data = load_and_combine()
        
        # Data validation
        logger.info("🔍 Validating dataset...")
        if not validate_data(data):
            logger.error("Data validation failed. Exiting.")
            return False
        
        # Display data statistics
        stats = get_data_statistics(data)
        logger.info("📊 Dataset Statistics:")
        logger.info(f"  - Total samples: {stats['total_samples']:,}")
        logger.info(f"  - Unique labels: {stats['unique_labels']}")
        logger.info(f"  - Label distribution: {stats['label_distribution']}")
        logger.info(f"  - Average text length: {stats['text_length_stats']['mean']:.1f} characters")
        
        # Step 2: Text Preprocessing
        logger.info("\n🧹 Step 2: Preprocessing text data...")
        data = preprocess_column(data)
        
        # Step 3: Feature Extraction
        logger.info("\n📈 Step 3: Creating TF-IDF features...")
        X, vectorizer = get_tfidf_features(
            data['clean_text'].tolist(),
            save_path=MODELS_DIR / MODEL_CONFIG['vectorizer_name']
        )
        
        # Step 4: Label Encoding
        logger.info("\n🏷️  Step 4: Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['label'])
        
        # Save label encoder
        label_encoder_path = MODELS_DIR / MODEL_CONFIG['label_encoder_name']
        joblib.dump(label_encoder, label_encoder_path)
        logger.info(f"✅ Label encoder saved to {label_encoder_path}")
        
        # Display label mapping
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        logger.info(f"📝 Label mapping: {label_mapping}")
        
        # Step 5: Model Training and Evaluation
        logger.info("\n🤖 Step 5: Training and evaluating models...")
        best_model_info = train_and_evaluate(X, y)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Training Pipeline Completed Successfully!")
        logger.info(f"📁 Models saved in: {MODELS_DIR}")
        logger.info(f"🏆 Best performing model: {best_model_info.get('name', 'N/A')}")
        logger.info(f"📊 Best F1 Score: {best_model_info.get('f1_score', 'N/A'):.4f}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        logger.exception("Full error traceback:")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mental Health Text Classifier Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --verbose         # Run with verbose logging
  python main.py --quick           # Quick training with reduced parameters
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode with reduced parameters'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(MODELS_DIR),
        help='Directory to save trained models'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run main pipeline
    success = main(args)
    sys.exit(0 if success else 1)
