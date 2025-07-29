#!/usr/bin/env python3
"""
Script to download and set up the ARC Prize 2025 dataset.
This script helps users get the dataset files in the correct format.
"""

import os
import sys
import json
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


def create_sample_data():
    """Create sample data files for testing if no real data is available."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Creating sample data files for testing...")
    
    # Sample training challenges
    sample_training_challenges = {
        "sample_task_1": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                },
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            "test": [
                {
                    "input": [[0, 0], [1, 1]]
                }
            ]
        },
        "sample_task_2": {
            "train": [
                {
                    "input": [[0, 0], [0, 0]],
                    "output": [[1, 1], [1, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 1], [1, 1]]
                }
            ]
        }
    }
    
    # Sample training solutions
    sample_training_solutions = {
        "sample_task_1": [[[1, 1], [0, 0]]],
        "sample_task_2": [[[0, 0], [0, 0]]]
    }
    
    # Sample evaluation challenges
    sample_evaluation_challenges = {
        "eval_task_1": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 0], [0, 1]]
                }
            ]
        }
    }
    
    # Sample evaluation solutions
    sample_evaluation_solutions = {
        "eval_task_1": [[[0, 1], [1, 0]]]
    }
    
    # Sample test challenges
    sample_test_challenges = {
        "test_task_1": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 0], [0, 1]]
                }
            ]
        }
    }
    
    # Sample submission
    sample_submission = {
        "test_task_1": [
            {
                "attempt_1": [[0, 1], [1, 0]],
                "attempt_2": [[1, 0], [0, 1]]
            }
        ]
    }
    
    # Save sample files
    files_to_create = [
        ("arc-agi_training-challenges.json", sample_training_challenges),
        ("arc-agi_training-solutions.json", sample_training_solutions),
        ("arc-agi_evaluation-challenges.json", sample_evaluation_challenges),
        ("arc-agi_evaluation-solutions.json", sample_evaluation_solutions),
        ("arc-agi_test-challenges.json", sample_test_challenges),
        ("sample_submission.json", sample_submission)
    ]
    
    for filename, data in files_to_create:
        filepath = data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Created {filename}")
    
    print("\nSample data created successfully!")
    print("You can now test the pipeline with:")
    print("  python src/main.py --evaluate --model baseline")


def check_dataset_files():
    """Check if the required dataset files exist."""
    data_dir = Path("data")
    required_files = [
        "arc-agi_training-challenges.json",
        "arc-agi_training-solutions.json",
        "arc-agi_evaluation-challenges.json",
        "arc-agi_evaluation-solutions.json",
        "arc-agi_test-challenges.json",
        "sample_submission.json"
    ]
    
    missing_files = []
    existing_files = []
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    print("Dataset file status:")
    print(f"  Found: {len(existing_files)} files")
    print(f"  Missing: {len(missing_files)} files")
    
    if existing_files:
        print("\nExisting files:")
        for filename in existing_files:
            print(f"  ✓ {filename}")
    
    if missing_files:
        print("\nMissing files:")
        for filename in missing_files:
            print(f"  ✗ {filename}")
    
    return len(missing_files) == 0


def download_instructions():
    """Print instructions for downloading the dataset."""
    print("\n" + "="*60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Go to the Kaggle competition page:")
    print("   https://kaggle.com/competitions/arc-prize-2025/data")
    print("\n2. Download the dataset (6.91 MB)")
    print("   - Click 'Download All' or use: kaggle competitions download -c arc-prize-2025")
    print("\n3. Extract the downloaded zip file")
    print("\n4. Move the following files to the 'data/' directory:")
    print("   - arc-agi_training-challenges.json")
    print("   - arc-agi_training-solutions.json")
    print("   - arc-agi_evaluation-challenges.json")
    print("   - arc-agi_evaluation-solutions.json")
    print("   - arc-agi_test-challenges.json")
    print("   - sample_submission.json")
    print("\n5. Run this script again to verify the setup")
    print("\nAlternative: Use the sample data for testing:")
    print("   python download_dataset.py --create-sample")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC Prize 2025 Dataset Setup")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data files for testing")
    parser.add_argument("--check", action="store_true",
                       help="Check if dataset files exist")
    
    args = parser.parse_args()
    
    print("ARC Prize 2025 - Dataset Setup")
    print("="*40)
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.check or not args.create_sample:
        if check_dataset_files():
            print("\n✓ All dataset files are present!")
            print("You can now run the pipeline:")
            print("  python src/main.py --evaluate --model baseline")
        else:
            download_instructions()


if __name__ == "__main__":
    main() 