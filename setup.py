"""
Setup script for Mental Health Text Classifier

This script automates the installation and setup process for the
Mental Health Text Classifier application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_banner():
    """Print a welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🧠 Mental Health Text Classifier             ║
    ║                        Setup Script                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")


def create_virtual_environment():
    """Create a virtual environment."""
    print("\n📦 Creating virtual environment...")
    
    try:
        if platform.system() == "Windows":
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            activate_script = "venv\\Scripts\\activate.bat"
        else:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            activate_script = "source venv/bin/activate"
        
        print("✅ Virtual environment created successfully")
        print(f"📝 To activate it, run: {activate_script}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False
    
    return True


def install_requirements():
    """Install required packages."""
    print("\n📥 Installing required packages...")
    
    try:
        # Determine pip path
        if platform.system() == "Windows":
            pip_path = "venv\\Scripts\\pip.exe"
        else:
            pip_path = "venv/bin/pip"
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ All packages installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("💡 Try installing manually: pip install -r requirements.txt")
        return False
    
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    try:
        # Determine python path
        if platform.system() == "Windows":
            python_path = "venv\\Scripts\\python.exe"
        else:
            python_path = "venv/bin/python"
        
        # Download NLTK data
        subprocess.run([
            python_path, "-c",
            "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
        ], check=True)
        
        print("✅ NLTK data downloaded successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "models",
        "logs", 
        "docs/images",
        "static/css",
        "static/js"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    return True


def check_data_files():
    """Check if data files are present."""
    print("\n📊 Checking for data files...")
    
    data_files = [
        "data/LD DA 1.csv",
        "data/LD EL1.csv",
        "data/LD PF1.csv", 
        "data/LD TS 1.csv"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please add your CSV data files to the data/ directory")
        return False
    
    print("✅ All data files found")
    return True


def run_tests():
    """Run basic tests to verify installation."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Determine python path
        if platform.system() == "Windows":
            python_path = "venv\\Scripts\\python.exe"
        else:
            python_path = "venv/bin/python"
        
        # Test imports
        test_script = """
import sys
sys.path.append('src')
from src.preprocessing import clean_text
from config.config import MODEL_CONFIG
print('✅ All imports successful')
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    activation_cmd = "venv\\Scripts\\activate" if platform.system() == "Windows" else "source venv/bin/activate"
    
    next_steps = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                        🎉 Setup Complete!                   ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📋 Next Steps:
    
    1. Activate virtual environment:
       {activation_cmd}
    
    2. Train the models:
       python main.py
    
    3. Launch the web app:
       streamlit run app.py
    
    4. Or make predictions:
       python predict.py --interactive
    
    📚 Documentation:
       - README.md for detailed instructions
       - docs/ directory for additional documentation
    
    🐛 Issues?
       - Check the logs/ directory for error details
       - Run tests: python -m pytest tests/
       - Review requirements.txt for dependencies
    
    🎯 Ready to classify mental health text!
    """
    
    print(next_steps)


def main():
    """Main setup function."""
    print_banner()
    
    # Check system requirements
    check_python_version()
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Creating directories", create_directories),
        ("Checking data files", check_data_files),
        ("Running tests", run_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n{'='*60}")
        print(f"🔄 {step_name}...")
        
        if step_function():
            print(f"✅ {step_name} completed")
        else:
            print(f"❌ {step_name} failed")
            failed_steps.append(step_name)
    
    print(f"\n{'='*60}")
    
    if failed_steps:
        print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n💡 Please resolve these issues before proceeding")
    else:
        print("🎉 Setup completed successfully!")
        print_next_steps()


if __name__ == "__main__":
    main()
