# 🧠 Mental Health Text Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/nishant-gupta911/MentalHealth-Classifier)](https://github.com/nishant-gupta911/MentalHealth-Classifier/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/nishant-gupta911/MentalHealth-Classifier)](https://github.com/nishant-gupta911/MentalHealth-Classifier/issues)
[![Last commit](https://img.shields.io/github/last-commit/nishant-gupta911/MentalHealth-Classifier)](https://github.com/nishant-gupta911/MentalHealth-Classifier/commits/main)

> An AI-powered mental health text classifier that analyzes text inputs to identify mental health conditions with ethical considerations and professional accuracy.

## 🚀 Project Overview

The Mental Health Text Classifier is a machine learning application designed to analyze text inputs and classify them into mental health categories. Built with ethical AI principles and responsible deployment practices, this project serves as a research and educational tool for understanding mental health patterns in text data.

**⚠️ Important Disclaimer**: This tool is for educational and research purposes only and should not be used as a substitute for professional mental health diagnosis or treatment.

### Why This Project Matters
- **Mental Health Awareness**: Contributes to the growing field of AI-assisted mental health research
- **Early Detection**: Helps identify potential mental health concerns in text communications
- **Research Tool**: Provides a foundation for academic and clinical research in computational psychiatry
- **Educational Value**: Demonstrates modern ML techniques applied to healthcare text analysis

## 🧠 Features

### 🎯 Core Functionality
- **Multi-class Classification**: Identifies 4 distinct mental health categories
- **Real-time Prediction**: Instant text analysis through web interface
- **Confidence Scoring**: Provides prediction confidence levels for transparency
- **Batch Processing**: Supports multiple text inputs simultaneously

### 🔬 Technical Highlights
- **Advanced NLP Pipeline**: TF-IDF vectorization with optimized preprocessing
- **Ensemble Learning**: LightGBM classifier with hyperparameter tuning
- **Feature Engineering**: N-gram analysis and text normalization
- **Model Interpretability**: Feature importance analysis and prediction explanations

### 💻 User Interface
- **Streamlit Web App**: Beautiful, responsive interface with dark theme
- **Interactive Visualizations**: Real-time charts and analytics
- **User-Friendly Design**: Intuitive interface for both technical and non-technical users
- **Comprehensive Analytics**: Detailed prediction breakdowns and statistics

## 🧰 Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Scikit-learn** - Machine learning framework
- **LightGBM** - Gradient boosting classifier
- **Pandas & NumPy** - Data manipulation and analysis
- **NLTK & TextBlob** - Natural language processing

### ML Pipeline
- **TF-IDF Vectorization** - Text feature extraction
- **Feature Selection** - Optimized feature engineering
- **Cross-validation** - Robust model evaluation
- **Hyperparameter Tuning** - Automated optimization

### Web Framework & Visualization
- **Streamlit** - Interactive web application
- **Plotly** - Dynamic data visualizations
- **Matplotlib & Seaborn** - Statistical plotting

### Development Tools
- **Joblib** - Model serialization
- **Pytest** - Testing framework
- **Black & Flake8** - Code formatting and linting
- **Docker** - Containerization support

## 📂 Directory Structure

```
MentalHealth-Classifier/
├── 📁 src/                          # Core source code
│   ├── data_loader.py               # Data loading and validation
│   ├── preprocessing.py             # Text preprocessing pipeline
│   ├── vectorizer.py                # Feature extraction
│   ├── trainer.py                   # Model training and evaluation
│   ├── models.py                    # Model architecture definitions
│   └── visuals.py                   # Visualization utilities
├── 📁 config/                       # Configuration files
│   └── config.py                    # Application settings
├── 📁 data/                         # Dataset files
│   ├── combined.csv                 # Main dataset
│   ├── LD_DA_1.csv                 # Depression & Anxiety data
│   ├── LD_EL1.csv                  # Emotional Loneliness data
│   ├── LD_PF1.csv                  # Panic & Family data
│   └── LD_TS_1.csv                 # Trauma & Stress data
├── 📁 models/                       # Trained model artifacts
│   ├── tuned_lightgbm_model.pkl    # Trained classifier
│   ├── tfidf_vectorizer.pkl        # Text vectorizer
│   └── label_encoder.pkl           # Label encoder
├── 📁 tests/                        # Test suite
│   ├── test_main.py                # Main pipeline tests
│   ├── test_preprocessing.py       # Preprocessing tests
│   └── test_runner.py              # Test runner
├── 📁 backup_unused/                # Utility scripts
│   └── create_fresh_models.py      # Model creation scripts
├── 📱 app.py                        # Streamlit web application
├── 🚀 main.py                       # Main training pipeline
├── 📊 predict.py                    # Prediction interface
├── 📋 requirements.txt              # Dependencies
└── 📄 README.md                     # Project documentation
```

## 📊 Dataset

### Data Overview
The classifier is trained on a comprehensive dataset of mental health-related text samples across four primary categories:

| Category | Description | Sample Size |
|----------|-------------|-------------|
| **Depression_Anxiety** | Text indicating depressive or anxious states | ~25% of dataset |
| **Emotional_Loneliness** | Content expressing loneliness and isolation | ~25% of dataset |
| **Panic_Family** | Family-related panic and stress indicators | ~25% of dataset |
| **Trauma_Stress** | Trauma-related and stress-induced content | ~25% of dataset |

### Data Characteristics
- **Format**: CSV files with text and label columns
- **Language**: English text samples
- **Preprocessing**: Cleaned and normalized text data
- **Validation**: Balanced dataset with quality assurance checks

### Ethical Considerations
- **Privacy**: All data is anonymized and de-identified
- **Consent**: Data collection follows ethical guidelines
- **Bias Mitigation**: Balanced representation across categories
- **Responsible Use**: Clear disclaimers about clinical limitations

## 🧪 Model Training

### Training Pipeline
The model follows a comprehensive training pipeline:

1. **Data Loading & Validation**
   - Load datasets from CSV files
   - Validate data integrity and format
   - Generate descriptive statistics

2. **Text Preprocessing**
   - Lowercase normalization
   - Punctuation and special character removal
   - Stopword elimination
   - Text tokenization and cleaning

3. **Feature Engineering**
   - TF-IDF vectorization (1000-5000 features)
   - N-gram analysis (1-2 grams)
   - Feature selection and optimization

4. **Model Training**
   - LightGBM classifier with hyperparameter tuning
   - Cross-validation for robust evaluation
   - Grid search for optimal parameters

5. **Evaluation & Validation**
   - Accuracy, precision, recall, and F1-score metrics
   - Confusion matrix analysis
   - Feature importance visualization

### Training Command
```bash
python main.py --train --evaluate --save-model
```

### Model Performance
- **Accuracy**: ~85-90% on validation set
- **Precision**: Balanced across all categories
- **Recall**: Optimized for clinical relevance
- **F1-Score**: Consistent performance metrics

## 💻 How to Run

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/nishant-gupta911/MentalHealth-Classifier.git
cd MentalHealth-Classifier
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Model Artifacts
```bash
# Option 1: Train new model
python main.py

# Option 2: Use pre-trained model (if available)
python backup_unused/create_fresh_models.py
```

### 5. Run the Application
```bash
# Launch Streamlit web app
streamlit run app.py

# Or use the task runner
python run_app.py
```

### 6. Access the Application
Open your browser and navigate to `http://localhost:8501`

## 🎯 Example Predictions

### Input Examples

**Example 1: Depression/Anxiety**
```
Input: "I've been feeling really down lately and can't seem to find motivation for anything. Everything feels overwhelming."
Output: {
  "category": "Depression_Anxiety",
  "confidence": 0.87,
  "probability_distribution": {
    "Depression_Anxiety": 0.87,
    "Emotional_Loneliness": 0.08,
    "Panic_Family": 0.03,
    "Trauma_Stress": 0.02
  }
}
```

**Example 2: Emotional Loneliness**
```
Input: "I feel so isolated and disconnected from everyone around me. Nobody really understands what I'm going through."
Output: {
  "category": "Emotional_Loneliness",
  "confidence": 0.92,
  "probability_distribution": {
    "Emotional_Loneliness": 0.92,
    "Depression_Anxiety": 0.05,
    "Panic_Family": 0.02,
    "Trauma_Stress": 0.01
  }
}
```

### Prediction Interface
```python
from predict import predict_text

# Single prediction
result = predict_text("Your text here")
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
results = predict_batch(texts)
```

## 📦 Output Artifacts

### Model Files
The trained model generates three key artifacts:

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| `tuned_lightgbm_model.pkl` | Trained LightGBM classifier | ~12-400KB | Main prediction model |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer | ~4-40KB | Text feature extraction |
| `label_encoder.pkl` | Label encoder | ~600-650 bytes | Category encoding/decoding |

### Artifact Validation
```python
import joblib

# Load and validate artifacts
model = joblib.load('models/tuned_lightgbm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
encoder = joblib.load('models/label_encoder.pkl')

print("✅ All artifacts loaded successfully")
print(f"Model type: {type(model)}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Categories: {list(encoder.classes_)}")
```

## 📈 Future Improvements

### Planned Enhancements
- **🤖 Deep Learning Models**: Integration with BERT, RoBERTa, or custom transformers
- **🌐 Multi-language Support**: Extend to support multiple languages
- **📊 Advanced Analytics**: Real-time dashboard with detailed insights
- **🔗 API Development**: RESTful API for external integrations
- **📱 Mobile App**: React Native or Flutter mobile application

### Research Opportunities
- **📚 Academic Collaboration**: Partner with mental health research institutions
- **🧬 Biomarker Integration**: Combine with physiological data
- **📈 Longitudinal Analysis**: Track mental health patterns over time
- **🎯 Personalization**: User-specific model adaptation

### Technical Improvements
- **⚡ Performance Optimization**: Model compression and inference speed
- **🔐 Security Enhancements**: Advanced data privacy and encryption
- **🧪 A/B Testing**: Continuous model improvement framework
- **📊 MLOps Pipeline**: Automated training, deployment, and monitoring

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
- **🐛 Bug Reports**: Report issues or unexpected behavior
- **💡 Feature Requests**: Suggest new features or improvements
- **📖 Documentation**: Improve documentation and tutorials
- **🧪 Testing**: Add tests and improve test coverage
- **🔧 Code Contributions**: Submit pull requests with enhancements

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/MentalHealth-Classifier.git

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

### Contribution Guidelines
1. Follow the existing code style and conventions
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting
5. Create clear, descriptive commit messages

### Code of Conduct
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand our community standards.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Important Notice
This software is designed for educational and research purposes in the field of mental health text analysis. It should not be used as a substitute for professional mental health diagnosis, treatment, or advice.

---

## 🙏 Acknowledgments

- **Mental Health Community**: For highlighting the importance of AI in mental health
- **Open Source Contributors**: For their valuable contributions and feedback
- **Research Community**: For advancing the field of computational psychiatry
- **Ethical AI Advocates**: For promoting responsible AI development

## 📞 Contact

- **GitHub**: [@nishant-gupta911](https://github.com/nishant-gupta911)
- **Project Repository**: [MentalHealth-Classifier](https://github.com/nishant-gupta911/MentalHealth-Classifier)
- **Issues**: [Report Issues](https://github.com/nishant-gupta911/MentalHealth-Classifier/issues)

---

**⭐ If you find this project helpful, please consider giving it a star!**
