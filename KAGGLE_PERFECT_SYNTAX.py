#!/usr/bin/env python3
"""
Perfect Syntax Kaggle Submission for ARC Prize 2025
No indentation errors, no syntax issues.

Copy this entire code into ONE cell in Kaggle and run it.
"""

import json
import os

print("Perfect Syntax ARC Prize 2025 Submission")

def load_data():
    """Load data with fallback."""
    try:
        locations = [
            'arc-agi_evaluation-challenges.json',
            'data/arc-agi_evaluation-challenges.json',
            '../input/arc-prize-2025/arc-agi_evaluation-challenges.json'
        ]
        
        for loc in locations:
            if os.path.exists(loc):
                with open(loc, 'r') as f:
                    data = json.load(f)
                print(f"Loaded data from {loc}")
                return data
    except Exception as e:
        print(f"Could not load data: {e}")
    
    print("Using sample data")
    return {
        "00576224": {
            "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[0, 0], [1, 1]]}]
        },
        "009d5c81": {
            "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
            "test": [{"input": [[1, 1], [0, 0]]}]
        }
    }

def simple_predict(input_grid):
    """Return the input grid as output."""
    return input_grid

def generate_submission():
    """Generate submission."""
    print("Generating submission...")
    
    challenges = load_data()
    submission = {}
    
    for task_id, task in challenges.items():
        print(f"Processing {task_id}")
        
        test_inputs = task.get('test', [])
        predictions = []
        
        for test_input in test_inputs:
            input_grid = test_input.get('input', [[0, 0], [0, 0]])
            
            pred = {
                "attempt_1": input_grid,
                "attempt_2": input_grid
            }
            predictions.append(pred)
        
        submission[task_id] = predictions
    
    try:
        with open('submission.json', 'w') as f:
            json.dump(submission, f, indent=2)
        print(f"Submission saved: {len(submission)} tasks")
        return True
    except Exception as e:
        print(f"Error saving submission: {e}")
        return False

# Main execution
success = generate_submission()
if success:
    print("SUCCESS: Submission created!")
    print("File: submission.json")
    print("Ready to submit to competition!")
else:
    print("ERROR: Failed to create submission") 