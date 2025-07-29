#!/usr/bin/env python3
"""
Simple Kaggle Submission for ARC Prize 2025
Minimal, reliable code that should work without issues

Copy this entire code into a single cell in Kaggle and run it.
"""

import json
import numpy as np
import os

print("Simple ARC Prize 2025 Submission")
print("Target: 25% Performance")

# Simple pattern analyzer
def analyze_pattern(input_grid, output_grid):
    """Simple pattern detection."""
    try:
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        # Test basic patterns
        if np.array_equal(input_array, output_array):
            return 'identity'
        elif np.array_equal(np.rot90(input_array, k=1), output_array):
            return 'rotation_90'
        elif np.array_equal(np.rot90(input_array, k=2), output_array):
            return 'rotation_180'
        elif np.array_equal(np.rot90(input_array, k=3), output_array):
            return 'rotation_270'
        elif np.array_equal(np.fliplr(input_array), output_array):
            return 'horizontal_flip'
        elif np.array_equal(np.flipud(input_array), output_array):
            return 'vertical_flip'
        else:
            return 'unknown'
    except:
        return 'unknown'

# Simple predictor
def predict_output(input_grid, pattern):
    """Simple prediction based on pattern."""
    try:
        input_array = np.array(input_grid)
        
        if pattern == 'identity':
            return input_grid
        elif pattern == 'rotation_90':
            return np.rot90(input_array, k=1).tolist()
        elif pattern == 'rotation_180':
            return np.rot90(input_array, k=2).tolist()
        elif pattern == 'rotation_270':
            return np.rot90(input_array, k=3).tolist()
        elif pattern == 'horizontal_flip':
            return np.fliplr(input_array).tolist()
        elif pattern == 'vertical_flip':
            return np.flipud(input_array).tolist()
        else:
            return input_grid
    except:
        return input_grid

# Load data
def load_data():
    """Load evaluation data."""
    print("Loading data...")
    
    # Try different file locations
    file_locations = [
        'arc-agi_evaluation-challenges.json',
        'data/arc-agi_evaluation-challenges.json',
        '../input/arc-prize-2025/arc-agi_evaluation-challenges.json'
    ]
    
    challenges_file = None
    for loc in file_locations:
        if os.path.exists(loc):
            challenges_file = loc
            break
    
    if not challenges_file:
        print("Creating sample data...")
        return create_sample_data()
    
    try:
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)
        print(f"Loaded {len(challenges)} tasks")
        return challenges
    except Exception as e:
        print(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data."""
    return {
        "00576224": {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            "test": [
                {"input": [[0, 0], [1, 1]]}
            ]
        },
        "009d5c81": {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
            ],
            "test": [
                {"input": [[1, 1], [0, 0]]}
            ]
        }
    }

# Generate submission
def generate_submission():
    """Generate submission."""
    print("Generating submission...")
    
    challenges = load_data()
    submission = {}
    
    for task_id, task in challenges.items():
        print(f"Processing {task_id}...")
        
        # Analyze pattern from training data
        pattern = 'identity'
        train_pairs = task.get('train', [])
        
        if train_pairs:
            # Use first training pair to detect pattern
            first_pair = train_pairs[0]
            pattern = analyze_pattern(first_pair['input'], first_pair['output'])
        
        # Generate predictions for test inputs
        test_inputs = task.get('test', [])
        predictions = []
        
        for test_input in test_inputs:
            input_grid = test_input.get('input', [[0, 0], [0, 0]])
            
            # Generate two attempts
            attempt_1 = predict_output(input_grid, pattern)
            attempt_2 = input_grid  # Fallback to identity
            
            pred = {
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            }
            predictions.append(pred)
        
        submission[task_id] = predictions
    
    # Save submission
    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Submission saved: {len(submission)} tasks")
    return submission

# Main execution
if __name__ == "__main__":
    submission = generate_submission()
    print("Submission ready!")
    print("File: submission.json") 