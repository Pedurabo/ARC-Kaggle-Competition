import json
print("Creating minimal submission...")
submission = {"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}], "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]}
with open('submission.json', 'w') as f: json.dump(submission, f, indent=2)
print("SUBMISSION CREATED: submission.json")
print("Ready to submit!") 