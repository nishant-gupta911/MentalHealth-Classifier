# 🔧 Create Scripts - Model Artifact Generation

This directory contains utility scripts for creating and regenerating model artifacts for the Mental Health Text Classifier project. These scripts are helpful for setup, debugging, and recreating models when needed.

## 📁 Scripts Overview

### Core Model Creation Scripts
- **`create_fresh_models.py`** - Creates complete fresh model artifacts from sample data
- **`create_proper_artifacts.py`** - Creates sklearn-compatible artifacts from actual data
- **`quick_train.py`** - Comprehensive training script with full pipeline

### Simple Creation Scripts  
- **`create_artifacts.py`** - Simple script for basic model artifacts
- **`simple_create.py`** - Minimal script for basic placeholder artifacts

## 🚀 Usage

### Quick Setup (Recommended)
If you need working models immediately:
```bash
cd create
python create_fresh_models.py
```

### Full Training Pipeline
For comprehensive model training with real data:
```bash
cd create
python quick_train.py
```

### Basic Artifacts Only
For simple placeholder artifacts:
```bash
cd create  
python simple_create.py
```

### sklearn-Compatible Artifacts
For proper sklearn artifacts using project data:
```bash
cd create
python create_proper_artifacts.py
```

## 🎯 When to Use Each Script

### `create_fresh_models.py` ✅ **Recommended**
- **Use when**: You need working models quickly
- **Creates**: Complete set of working artifacts
- **Data**: Uses built-in sample data
- **Dependencies**: pandas, scikit-learn, joblib
- **Output**: 
  - RandomForest model (12KB)
  - TF-IDF vectorizer (4KB) 
  - Label encoder (610 bytes)

### `quick_train.py` 🚀 **Full Pipeline**
- **Use when**: You want complete training with real data
- **Creates**: Full ML pipeline with evaluation
- **Data**: Uses project's actual data files
- **Dependencies**: pandas, scikit-learn, lightgbm, joblib
- **Features**: 
  - Data loading and validation
  - Text preprocessing
  - Model training and evaluation
  - Artifact verification

### `create_proper_artifacts.py` 🔬 **Data-Driven**
- **Use when**: You need artifacts from real project data
- **Creates**: Vectorizer and encoder from actual data
- **Data**: Uses project's CSV files
- **Note**: Requires existing model file

### `create_artifacts.py` 🛠️ **Basic Setup**
- **Use when**: Simple setup needed
- **Creates**: Mock artifacts for testing
- **Data**: Uses predefined vocabulary
- **Dependencies**: Basic Python only

### `simple_create.py` ⚡ **Minimal**
- **Use when**: Quick placeholder artifacts needed
- **Creates**: Minimal working artifacts
- **Dependencies**: Python standard library only

## 📊 Output Artifacts

All scripts create files in the `../models/` directory:

```
models/
├── tuned_lightgbm_model.pkl    # Trained ML model
├── tfidf_vectorizer.pkl        # Text vectorizer
└── label_encoder.pkl           # Label encoder
```

### Expected File Sizes
- **Model**: ~12-400KB (depending on complexity)
- **Vectorizer**: ~4-40KB (depending on vocabulary size)
- **Label Encoder**: ~600-650 bytes

## 🔧 Configuration

### Mental Health Categories
All scripts use these 4 categories:
- `Depression_Anxiety`
- `Emotional_Loneliness` 
- `Panic_Family`
- `Trauma_Stress`

### Model Parameters
- **TF-IDF**: 1000-5000 features, 1-2 gram range
- **Model**: RandomForest or LightGBM
- **Text Processing**: Lowercase, stopword removal

## 🛠️ Troubleshooting

### Common Issues

1. **Module Not Found**
   ```bash
   # Install dependencies
   pip install pandas scikit-learn joblib
   # For full pipeline:
   pip install lightgbm
   ```

2. **Data File Not Found**
   - Check that `../data/` directory exists
   - Ensure CSV files are present
   - Use `create_fresh_models.py` if no data available

3. **Permission Errors**
   - Ensure write permissions to `../models/` directory
   - Check if model files are in use by another process

4. **Empty Model Files**
   - Delete existing model files and re-run script
   - Check for Python environment issues
   - Try `simple_create.py` first

### Validation
After running any script, verify artifacts:
```bash
cd ..
python -c "
import joblib
model = joblib.load('models/tuned_lightgbm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
encoder = joblib.load('models/label_encoder.pkl')
print('✅ All artifacts loaded successfully')
print(f'Model: {type(model)}')
print(f'Vectorizer vocab: {len(vectorizer.vocabulary_)}')
print(f'Encoder classes: {list(encoder.classes_)}')
"
```

## 📝 Development Notes

### Path Handling
All scripts are adjusted to work from the `create/` directory:
- Data paths: `../data/`
- Model paths: `../models/`
- Project root: `..`

### Dependencies
- **Minimal**: Python standard library only
- **Basic**: + pandas, scikit-learn, joblib
- **Full**: + lightgbm, numpy

### Error Handling
All scripts include:
- Exception handling and error reporting
- File existence checks
- Size validation
- Test predictions

## 🎯 Best Practices

1. **Start Simple**: Use `create_fresh_models.py` first
2. **Verify Output**: Always check file sizes and test loading
3. **Clean Setup**: Delete old artifacts before creating new ones
4. **Check Dependencies**: Ensure required packages are installed
5. **Test Pipeline**: Run a test prediction after creation

---

**These scripts ensure you always have working model artifacts for development and testing!** 🚀
