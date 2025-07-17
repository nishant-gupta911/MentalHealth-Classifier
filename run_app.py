#!/usr/bin/env python3
"""
Simple launcher for the Streamlit app
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting Mental Health Text Classifier App...")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Make sure you're in the project directory.")
        return
    
    # Check if models exist
    models_dir = Path("models")
    required_files = [
        "tuned_lightgbm_model.pkl",
        "tfidf_vectorizer.pkl", 
        "label_encoder.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("Please run the model creation script first:")
        print("   python fix_models_comprehensive.py")
        return
    
    # Start Streamlit app
    python_path = Path(sys.executable)
    
    try:
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            str(python_path), "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
