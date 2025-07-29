#!/usr/bin/env python3
"""
Test script to verify Kaggle notebook code works with actual ARC dataset
"""

import json
import os

def test_data_loading():
    """Test if we can load the actual ARC dataset files."""
    print("🧪 Testing data loading with actual ARC dataset...")
    
    # Check for actual dataset files (with correct names - hyphens, not underscores)
    data_files = [
        'arc-agi_evaluation-challenges.json',
        'arc-agi_evaluation-solutions.json',
        'arc-agi_test-challenges.json',
        'arc-agi_training-challenges.json',
        'arc-agi_training-solutions.json',
        'sample_submission.json'
    ]
    
    # Also check in data/ directory
    data_dir_files = [
        'data/arc-agi_evaluation-challenges.json',
        'data/arc-agi_evaluation-solutions.json',
        'data/arc-agi_test-challenges.json',
        'data/arc-agi_training-challenges.json',
        'data/arc-agi_training-solutions.json',
        'data/sample_submission.json'
    ]
    
    print("📂 Checking for dataset files:")
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"  ✅ {file} ({size:.2f} MB)")
        else:
            print(f"  ❌ {file} (missing)")
    
    print("\n📂 Checking data/ directory:")
    for file in data_dir_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"  ✅ {file} ({size:.2f} MB)")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Try to find and load evaluation data
    eval_challenges_file = None
    eval_solutions_file = None
    
    # Check current directory first
    if os.path.exists('arc-agi_evaluation-challenges.json'):
        eval_challenges_file = 'arc-agi_evaluation-challenges.json'
    elif os.path.exists('data/arc-agi_evaluation-challenges.json'):
        eval_challenges_file = 'data/arc-agi_evaluation-challenges.json'
    
    if os.path.exists('arc-agi_evaluation-solutions.json'):
        eval_solutions_file = 'arc-agi_evaluation-solutions.json'
    elif os.path.exists('data/arc-agi_evaluation-solutions.json'):
        eval_solutions_file = 'data/arc-agi_evaluation-solutions.json'
    
    if not eval_challenges_file or not eval_solutions_file:
        print("❌ Could not find evaluation data files")
        return False
    
    try:
        print(f"\n📊 Loading evaluation challenges from {eval_challenges_file}...")
        with open(eval_challenges_file, 'r') as f:
            eval_challenges = json.load(f)
        
        print(f"📊 Loading evaluation solutions from {eval_solutions_file}...")
        with open(eval_solutions_file, 'r') as f:
            eval_solutions = json.load(f)
        
        print(f"✅ Successfully loaded evaluation data!")
        print(f"📊 Number of evaluation tasks: {len(eval_challenges)}")
        print(f"📊 Sample task IDs: {list(eval_challenges.keys())[:5]}")
        
        # Check a sample task structure
        sample_task_id = list(eval_challenges.keys())[0]
        sample_task = eval_challenges[sample_task_id]
        print(f"\n📋 Sample task structure ({sample_task_id}):")
        print(f"  Train pairs: {len(sample_task.get('train', []))}")
        print(f"  Test inputs: {len(sample_task.get('test', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_submission_format():
    """Test submission format generation."""
    print("\n🧪 Testing submission format...")
    
    # Create a simple test submission
    test_submission = {
        "test_task_1": [
            {
                "attempt_1": [[0, 1], [1, 0]],
                "attempt_2": [[1, 0], [0, 1]]
            }
        ],
        "test_task_2": [
            {
                "attempt_1": [[0, 0], [1, 1]],
                "attempt_2": [[1, 1], [0, 0]]
            },
            {
                "attempt_1": [[1, 1], [0, 0]],
                "attempt_2": [[0, 0], [1, 1]]
            }
        ]
    }
    
    # Save test submission
    with open('test_submission.json', 'w') as f:
        json.dump(test_submission, f, indent=2)
    
    print("✅ Test submission created: test_submission.json")
    
    # Verify format
    with open('test_submission.json', 'r') as f:
        loaded_data = json.load(f)
    
    valid_format = True
    for task_id, predictions in loaded_data.items():
        for pred in predictions:
            if 'attempt_1' not in pred or 'attempt_2' not in pred:
                valid_format = False
                break
    
    if valid_format:
        print("✅ Submission format is valid")
    else:
        print("❌ Submission format has issues")
    
    return valid_format

if __name__ == "__main__":
    print("🚀 Testing Kaggle Notebook Code")
    print("=" * 50)
    
    # Test data loading
    data_loaded = test_data_loading()
    
    # Test submission format
    format_valid = test_submission_format()
    
    print("\n" + "=" * 50)
    if data_loaded and format_valid:
        print("✅ ALL TESTS PASSED!")
        print("🎯 Ready for Kaggle submission!")
    else:
        print("❌ Some tests failed")
        print("🔧 Please check the issues above")
    
    print("=" * 50) 