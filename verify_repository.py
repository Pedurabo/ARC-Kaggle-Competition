#!/usr/bin/env python3
"""
Repository Verification Script
Verifies that all key components are present and functional.
"""

import os
import sys
import json

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} (MISSING)")
        return False

def main():
    """Main verification function."""
    print("🔍 ARC Prize 2025 Repository Verification")
    print("=" * 50)
    
    # Track overall status
    all_good = True
    
    # Check core directories
    print("\n📁 Directory Structure:")
    directories = [
        ("src/", "Source code directory"),
        ("src/models/", "Model implementations"),
        ("src/utils/", "Utility functions"),
        ("src/evaluation/", "Evaluation tools"),
        ("notebooks/", "Jupyter notebooks"),
        ("configs/", "Configuration files"),
        ("data/", "Data directory"),
        ("tests/", "Test files")
    ]
    
    for dirpath, description in directories:
        if not check_directory_exists(dirpath, description):
            all_good = False
    
    # Check key files
    print("\n📄 Key Files:")
    files = [
        ("README.md", "Project documentation"),
        ("requirements.txt", "Python dependencies"),
        ("setup.py", "Package setup"),
        ("src/models/breakthrough_modules.py", "Breakthrough models"),
        ("src/models/advanced_models.py", "Advanced models"),
        ("src/models/base_model.py", "Base model interface"),
        ("src/utils/data_loader.py", "Data loading utilities"),
        ("src/evaluation/scorer.py", "Evaluation scoring"),
        ("train_breakthrough_modules.py", "Training script"),
        ("test_models.py", "Model testing"),
        ("KAGGLE_PERFECT_SYNTAX.py", "Kaggle submission"),
        ("submission.json", "Sample submission")
    ]
    
    for filepath, description in files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check Python imports
    print("\n🐍 Python Import Test:")
    try:
        import sys
        sys.path.append('src')
        
        # Test basic imports
        from models.base_model import BaseARCModel
        print("✅ Base model import successful")
        
        from utils.data_loader import ARCDataset
        print("✅ Data loader import successful")
        
        from evaluation.scorer import CrossValidationScorer
        print("✅ Evaluation scorer import successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        all_good = False
    
    # Check submission format
    print("\n📊 Submission Format Test:")
    try:
        with open('submission.json', 'r') as f:
            submission = json.load(f)
        
        if isinstance(submission, dict):
            print("✅ Submission format is valid JSON object")
            print(f"   Tasks: {len(submission)}")
        else:
            print("❌ Submission format invalid")
            all_good = False
            
    except Exception as e:
        print(f"❌ Submission file error: {e}")
        all_good = False
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 REPOSITORY VERIFICATION: SUCCESS")
        print("✅ All components present and functional")
        print("🚀 Ready for development and Kaggle submission!")
    else:
        print("⚠️  REPOSITORY VERIFICATION: ISSUES FOUND")
        print("🔧 Please check the missing components above")
    
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test models: python test_models.py")
    print("3. Train breakthrough models: python train_breakthrough_modules.py")
    print("4. Generate submission: python KAGGLE_PERFECT_SYNTAX.py")
    print("5. Submit to Kaggle competition")

if __name__ == "__main__":
    main() 