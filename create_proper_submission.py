#!/usr/bin/env python3
"""
Create Proper Kaggle Submission Format
"""

import json
import numpy as np
from src.models.ultimate_intelligence_integration import get_ultimate_intelligence_integration

def create_proper_submission():
    """Create submission in proper Kaggle format"""
    print("🔧 Creating properly formatted Kaggle submission...")
    
    # Load test challenges
    with open('data_actual/arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)
    
    # Initialize expert systems
    ultimate_intelligence = get_ultimate_intelligence_integration()
    
    # Create submission in proper format
    submission = {}
    
    for task_id, task_data in test_challenges.items():
        print(f"🔍 Processing task: {task_id}")
        
        # Generate solutions using expert systems
        try:
            solution = ultimate_intelligence.solve_task_with_ultimate_intelligence({
                'task_id': task_id,
                'data': task_data
            })
            
            # Extract predictions (assuming 13 predictions as per our system)
            predictions = solution.predictions if hasattr(solution, 'predictions') else []
            
            # Create proper format: list of solutions with attempt_1 and attempt_2
            formatted_solutions = []
            
            # Create 13 solutions as per our system
            for i in range(13):
                if i < len(predictions):
                    # Use actual prediction if available
                    pred = predictions[i]
                    if isinstance(pred, list) and len(pred) >= 2:
                        attempt_1 = pred[0] if isinstance(pred[0], list) else [[0, 0]]
                        attempt_2 = pred[1] if isinstance(pred[1], list) else [[0, 0]]
                    else:
                        attempt_1 = [[0, 0]]
                        attempt_2 = [[0, 0]]
                else:
                    # Default format
                    attempt_1 = [[0, 0]]
                    attempt_2 = [[0, 0]]
                
                formatted_solutions.append({
                    "attempt_1": attempt_1,
                    "attempt_2": attempt_2
                })
            
            submission[task_id] = formatted_solutions
            
        except Exception as e:
            print(f"⚠️ Error processing {task_id}: {e}")
            # Create default format for this task
            submission[task_id] = [
                {"attempt_1": [[0, 0]], "attempt_2": [[0, 0]]}
            ]
    
    # Save submission
    with open('kaggle_submission.json', 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    print(f"✅ Proper submission created with {len(submission)} tasks")
    print("📁 Saved as: kaggle_submission.json")
    
    return submission

if __name__ == "__main__":
    create_proper_submission() 