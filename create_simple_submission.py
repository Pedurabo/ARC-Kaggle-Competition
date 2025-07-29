#!/usr/bin/env python3
"""
Create Simple Kaggle Submission - Exact Sample Format
"""

import json
import copy

def create_simple_submission():
    """Create submission in exact sample format"""
    print("🔧 Creating simple submission in exact sample format...")
    
    # Load sample submission to get the exact format
    with open('data_actual/sample_submission.json', 'r') as f:
        sample_data = json.load(f)
    
    # Create submission by copying the sample format exactly
    submission = {}
    
    # Process each task ID from sample
    for task_id in sample_data.keys():
        # Use the exact same format as sample
        submission[task_id] = [
            {"attempt_1": [[0, 0]], "attempt_2": [[0, 0]]}
        ]
    
    # Save submission
    with open('simple_submission.json', 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    print(f"✅ Simple submission created with {len(submission)} tasks")
    print("📁 Saved as: simple_submission.json")
    
    return submission

if __name__ == "__main__":
    create_simple_submission() 