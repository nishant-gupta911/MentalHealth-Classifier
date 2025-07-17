# 🧪 Test Suite - Mental Health Text Classifier

This directory contains comprehensive tests for the Mental Health Text Classifier project. All test files have been organized here for better project structure and maintainability.

## 📁 Test Files Overview

### Core Tests
- **`test_main.py`** - Original comprehensive unit tests with unittest framework
- **`test_runner.py`** - Test runner that discovers and executes all tests

### Component Tests  
- **`test_simple_loading.py`** - Tests model loading with simplified preprocessing
- **`test_model_loading.py`** - Detailed model artifact loading diagnostics
- **`test_app_loading.py`** - Application-specific model loading tests
- **`test_diagnostics.py`** - System diagnostics and import testing
- **`test_artifacts.py`** - Model artifact creation and validation tests
- **`test_preprocessing.py`** - Text preprocessing function tests

## 🚀 Running Tests

### Run All Tests
```bash
# From the project root directory
cd tests
python test_runner.py
```

### Run Individual Tests
```bash
# Run specific test files
python test_simple_loading.py
python test_model_loading.py
python test_preprocessing.py
python test_artifacts.py
```

### Run Original Unit Tests
```bash
# Run the comprehensive unittest suite
python -m unittest test_main.py -v
```

## 🧪 Test Categories

### 1. Model Loading Tests
- **Purpose**: Verify that trained models can be loaded correctly
- **Files**: `test_simple_loading.py`, `test_model_loading.py`, `test_app_loading.py`
- **What they test**:
  - Model file existence and size validation
  - Joblib loading functionality
  - Model artifact compatibility
  - Prediction pipeline integrity

### 2. Preprocessing Tests  
- **Purpose**: Validate text preprocessing functions
- **Files**: `test_preprocessing.py`, `test_main.py`
- **What they test**:
  - Text cleaning functions
  - Edge case handling (empty text, unicode, long text)
  - Preprocessing pipeline
  - DataFrame column processing

### 3. System Diagnostics
- **Purpose**: Ensure system dependencies and imports work
- **Files**: `test_diagnostics.py`
- **What they test**:
  - Module imports
  - Dependency availability
  - Data loading functionality
  - Configuration access

### 4. Artifact Validation
- **Purpose**: Test model artifact creation and validation
- **Files**: `test_artifacts.py`
- **What they test**:
  - Model artifact creation process
  - Artifact file integrity
  - Prediction workflow with fresh artifacts
  - Temporary artifact creation

## 📊 Test Coverage

The tests cover:
- ✅ Model loading and validation
- ✅ Text preprocessing functions
- ✅ Data loading and validation
- ✅ Configuration management
- ✅ Error handling and edge cases
- ✅ Prediction pipeline
- ✅ Artifact creation and persistence

## 🔧 Test Configuration

### Path Management
All test files are configured to work from the `tests/` directory with proper import paths:
```python
# Standard path setup for tests
sys.path.append(str(Path(__file__).parent.parent / "src"))
```

### Dependencies
Tests require the same dependencies as the main application:
- pandas, numpy, scikit-learn
- joblib for model persistence
- pytest (optional, for advanced testing)

## 📈 Test Results

Expected test outcomes:
- **Model Loading**: Should pass if model artifacts exist and are valid
- **Preprocessing**: Should pass with proper text cleaning
- **Diagnostics**: Should pass if all dependencies are installed
- **Artifacts**: Should pass if model creation pipeline works

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from the correct directory
2. **Model Not Found**: Run `python create_fresh_models.py` from project root
3. **Missing Dependencies**: Install requirements with `pip install -r requirements.txt`

### Debug Mode
For verbose output during testing:
```python
# Add to any test file for debug output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Adding New Tests

To add new test files:
1. Create `test_[component].py` in this directory
2. Follow the path setup pattern from existing tests
3. The test runner will automatically discover and run new tests

Example test file structure:
```python
#!/usr/bin/env python3
"""
Test description
"""
import sys
from pathlib import Path

# Path setup
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_function():
    # Your test logic here
    pass

if __name__ == "__main__":
    success = test_function()
    if not success:
        sys.exit(1)
```

## 📝 Test Maintenance

- Tests are updated whenever core functionality changes
- All tests should pass before committing changes
- New features should include corresponding tests
- Edge cases and error conditions should be tested

---

**Run the complete test suite regularly to ensure system integrity!** 🚀
