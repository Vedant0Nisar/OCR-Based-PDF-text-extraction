#!/usr/bin/env python3
"""
Setup script for Legal Document IDP System
This script helps set up the system dependencies and validates the installation
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

def print_step(step, message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print(f"{'='*60}")

def run_command(command, check=True):
    """Run command and return result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please install Python 3.8 or higher")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print_step(2, "Installing Python Dependencies")
    
    # Check if pip is available
    success, _, _ = run_command("pip --version", check=False)
    if not success:
        print("❌ pip not found. Please install pip first.")
        return False
    
    print("📦 Installing Python packages...")
    success, stdout, stderr = run_command("pip install -r requirements.txt", check=False)
    
    if success:
        print("✅ Python dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install Python dependencies")
        print(f"Error: {stderr}")
        return False

def setup_windows_dependencies():
    """Setup Windows-specific dependencies"""
    print_step(3, "Setting up Windows Dependencies")
    
    # Check if Tesseract is installed
    success, _, _ = run_command("tesseract --version", check=False)
    if not success:
        print("⚠️  Tesseract OCR not found")
        print("   Please download and install from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Make sure to add it to your PATH")
    else:
        print("✅ Tesseract OCR found")
    
    # Check if Poppler is available
    success, _, _ = run_command("pdftoppm -v", check=False)
    if not success:
        print("⚠️  Poppler not found")
        print("   Please download from:")
        print("   https://github.com/oschwartz10612/poppler-windows/releases")
        print("   Extract and add bin folder to PATH")
    else:
        print("✅ Poppler found")

def setup_linux_dependencies():
    """Setup Linux-specific dependencies"""
    print_step(3, "Setting up Linux Dependencies")
    
    # Detect package manager
    if shutil.which("apt-get"):
        pkg_manager = "apt-get"
    elif shutil.which("yum"):
        pkg_manager = "yum"
    elif shutil.which("dnf"):
        pkg_manager = "dnf"
    else:
        print("⚠️  Could not detect package manager")
        print("   Please install tesseract-ocr and poppler-utils manually")
        return
    
    print(f"📦 Using {pkg_manager} package manager")
    
    if pkg_manager == "apt-get":
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-deu tesseract-ocr-ori",
            "sudo apt-get install -y poppler-utils"
        ]
    elif pkg_manager in ["yum", "dnf"]:
        commands = [
            f"sudo {pkg_manager} install -y tesseract tesseract-langpack-hin tesseract-langpack-deu",
            f"sudo {pkg_manager} install -y poppler-utils"
        ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        success, _, stderr = run_command(cmd, check=False)
        if not success:
            print(f"⚠️  Command failed: {stderr}")

def setup_macos_dependencies():
    """Setup macOS-specific dependencies"""
    print_step(3, "Setting up macOS Dependencies")
    
    # Check if Homebrew is installed
    success, _, _ = run_command("brew --version", check=False)
    if not success:
        print("⚠️  Homebrew not found")
        print("   Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return
    
    print("📦 Installing dependencies via Homebrew...")
    commands = [
        "brew install tesseract",
        "brew install poppler"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        success, _, stderr = run_command(cmd, check=False)
        if not success:
            print(f"⚠️  Command failed: {stderr}")

def download_models():
    """Pre-download AI models to cache"""
    print_step(4, "Pre-downloading AI Models")
    
    try:
        from transformers import (
            DonutProcessor, 
            VisionEncoderDecoderModel,
            MBart50TokenizerFast,
            MBartForConditionalGeneration,
            pipeline
        )
        
        print("📥 Downloading Donut model...")
        DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        
        print("📥 Downloading translation model...")
        MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        print("📥 Downloading NER model...")
        pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        print("📥 Downloading QA model...")
        pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        print("✅ All models downloaded successfully")
        
    except Exception as e:
        print(f"⚠️  Failed to download models: {e}")
        print("   Models will be downloaded on first use")

def validate_installation():
    """Validate the installation"""
    print_step(5, "Validating Installation")
    
    # Test imports
    try:
        import streamlit
        import torch
        import transformers
        import easyocr
        import PIL
        import pdf2image
        print("✅ All Python packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test system dependencies
    deps_ok = True
    
    # Test Tesseract
    success, stdout, _ = run_command("tesseract --version", check=False)
    if success:
        version = stdout.split('\n')[0]
        print(f"✅ Tesseract: {version}")
    else:
        print("❌ Tesseract not accessible")
        deps_ok = False
    
    # Test Poppler
    success, stdout, _ = run_command("pdftoppm -v", check=False)
    if success:
        print("✅ Poppler: Available")
    else:
        print("❌ Poppler not accessible")
        deps_ok = False
    
    # Test GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  GPU: Not available (will use CPU)")
    
    return deps_ok

def create_test_script():
    """Create a test script to verify functionality"""
    print_step(6, "Creating Test Script")
    
    test_script = '''
import streamlit as st
import torch
from transformers import pipeline

st.title("🧪 Legal IDP System Test")

st.write("### System Information")
st.write(f"Python version: {sys.version}")
st.write(f"PyTorch version: {torch.__version__}")
st.write(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    st.write(f"GPU: {torch.cuda.get_device_name()}")

st.write("### Quick Test")
if st.button("Test AI Pipeline"):
    with st.spinner("Testing..."):
        try:
            # Test simple pipeline
            classifier = pipeline("sentiment-analysis")
            result = classifier("This is a test document.")
            st.success("✅ AI pipeline working!")
            st.json(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.write("If you see this page, the basic setup is working!")
'''
    
    with open("test_system.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_system.py")
    print("   Run: streamlit run test_system.py")

def main():
    """Main setup function"""
    print("🚀 Legal Document IDP System Setup")
    print("=====================================")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("⚠️  Continuing with system dependencies...")
    
    # Setup system dependencies based on OS
    system = platform.system()
    if system == "Windows":
        setup_windows_dependencies()
    elif system == "Linux":
        setup_linux_dependencies()
    elif system == "Darwin":  # macOS
        setup_macos_dependencies()
    else:
        print(f"⚠️  Unsupported operating system: {system}")
    
    # Download models
    download_models()
    
    # Validate installation
    if validate_installation():
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues")
        print("   Please check the errors above and fix them")
    
    # Create test script
    create_test_script()
    
    print("\n📋 Next Steps:")
    print("1. Run: streamlit run test_system.py (to test basic functionality)")
    print("2. Run: streamlit run legal_idp_system.py (to start the main application)")
    print("3. Upload a legal document and test the extraction")
    
    print("\n🔧 If you encounter issues:")
    print("1. Check that all system dependencies are in PATH")
    print("2. Restart your terminal/command prompt")
    print("3. Ensure you have sufficient disk space (>5GB for models)")
    print("4. Check your internet connection for model downloads")

if __name__ == "__main__":
    main()
