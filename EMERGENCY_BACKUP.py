#!/usr/bin/env python3
"""
EMERGENCY BACKUP - Minimal Submission Creator
This will create a submission file even if everything else fails.

Copy this into ONE cell in Kaggle and run it.
"""

import json

print("EMERGENCY BACKUP: Creating minimal submission")

# Create minimal submission directly
submission = {
    "00576224": [
        {
            "attempt_1": [[0, 0], [0, 0]],
            "attempt_2": [[0, 0], [0, 0]]
        }
    ],
    "009d5c81": [
        {
            "attempt_1": [[0, 0], [0, 0]],
            "attempt_2": [[0, 0], [0, 0]]
        }
    ]
}

# Save submission
try:
    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    print("EMERGENCY SUBMISSION CREATED!")
    print("File: submission.json")
    print("Ready to submit!")
except Exception as e:
    print(f"ERROR: {e}")
    print("Try manual submission") 