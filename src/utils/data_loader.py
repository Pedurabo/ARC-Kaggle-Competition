"""
Data loader utilities for ARC Prize 2025 competition.
Handles loading and preprocessing of ARC-AGI-2 dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image


class ARCDataset:
    """ARC dataset loader and utilities."""
    
    def __init__(self, data_dir: str):
        """
        Initialize ARC dataset loader.
        
        Args:
            data_dir: Path to directory containing ARC data files
        """
        self.data_dir = Path(data_dir)
        self.training_challenges = {}
        self.training_solutions = {}
        self.evaluation_challenges = {}
        self.evaluation_solutions = {}
        self.test_challenges = {}
        
    def load_training_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load training challenges and solutions."""
        training_challenges_file = self.data_dir / "arc-agi_training-challenges.json"
        training_solutions_file = self.data_dir / "arc-agi_training-solutions.json"
        
        if training_challenges_file.exists():
            with open(training_challenges_file, 'r') as f:
                self.training_challenges = json.load(f)
        
        if training_solutions_file.exists():
            with open(training_solutions_file, 'r') as f:
                self.training_solutions = json.load(f)
                
        return self.training_challenges, self.training_solutions
    
    def load_evaluation_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load evaluation challenges and solutions."""
        eval_challenges_file = self.data_dir / "arc-agi_evaluation-challenges.json"
        eval_solutions_file = self.data_dir / "arc-agi_evaluation-solutions.json"
        
        if eval_challenges_file.exists():
            with open(eval_challenges_file, 'r') as f:
                self.evaluation_challenges = json.load(f)
        
        if eval_solutions_file.exists():
            with open(eval_solutions_file, 'r') as f:
                self.evaluation_solutions = json.load(f)
                
        return self.evaluation_challenges, self.evaluation_solutions
    
    def load_test_data(self) -> Dict[str, Any]:
        """Load test challenges."""
        test_challenges_file = self.data_dir / "arc-agi_test-challenges.json"
        
        if test_challenges_file.exists():
            with open(test_challenges_file, 'r') as f:
                self.test_challenges = json.load(f)
                
        return self.test_challenges
    
    def get_task_by_id(self, task_id: str, split: str = "training") -> Optional[Dict[str, Any]]:
        """
        Get a specific task by ID.
        
        Args:
            task_id: Task identifier
            split: Dataset split ('training', 'evaluation', 'test')
            
        Returns:
            Task data or None if not found
        """
        if split == "training":
            return self.training_challenges.get(task_id)
        elif split == "evaluation":
            return self.evaluation_challenges.get(task_id)
        elif split == "test":
            return self.test_challenges.get(task_id)
        return None
    
    def get_solution_by_id(self, task_id: str, split: str = "training") -> Optional[List[List[List[int]]]]:
        """
        Get solution for a specific task by ID.
        
        Args:
            task_id: Task identifier
            split: Dataset split ('training', 'evaluation')
            
        Returns:
            Solution data or None if not found
        """
        if split == "training":
            return self.training_solutions.get(task_id)
        elif split == "evaluation":
            return self.evaluation_solutions.get(task_id)
        return None
    
    def visualize_task(self, task_id: str, split: str = "training") -> None:
        """
        Visualize a task's input-output pairs.
        
        Args:
            task_id: Task identifier
            split: Dataset split
        """
        task = self.get_task_by_id(task_id, split)
        if not task:
            print(f"Task {task_id} not found in {split} split")
            return
            
        print(f"Task {task_id}:")
        print(f"Number of train pairs: {len(task.get('train', []))}")
        print(f"Number of test inputs: {len(task.get('test', []))}")
        
        # Visualize first train pair
        if task.get('train'):
            train_pair = task['train'][0]
            print("\nFirst training pair:")
            print("Input:")
            self._print_grid(train_pair['input'])
            print("Output:")
            self._print_grid(train_pair['output'])
        
        # Show test inputs
        if task.get('test'):
            print(f"\nTest inputs ({len(task['test'])}):")
            for i, test_input in enumerate(task['test']):
                print(f"Test input {i+1}:")
                self._print_grid(test_input['input'])
    
    def _print_grid(self, grid: List[List[int]]) -> None:
        """Print a grid in a readable format."""
        for row in grid:
            print(' '.join(str(cell) for cell in row))
    
    def grid_to_image(self, grid: List[List[int]], cell_size: int = 20) -> np.ndarray:
        """
        Convert a grid to an image array.
        
        Args:
            grid: 2D grid of integers
            cell_size: Size of each cell in pixels
            
        Returns:
            Image array
        """
        height, width = len(grid), len(grid[0])
        img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
        
        # Simple color mapping (can be enhanced)
        colors = {
            0: [255, 255, 255],  # White
            1: [0, 0, 0],        # Black
            2: [255, 0, 0],      # Red
            3: [0, 255, 0],      # Green
            4: [0, 0, 255],      # Blue
            5: [255, 255, 0],    # Yellow
            6: [255, 0, 255],    # Magenta
            7: [0, 255, 255],    # Cyan
            8: [128, 128, 128],  # Gray
            9: [255, 165, 0],    # Orange
        }
        
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                color = colors.get(cell, [128, 128, 128])
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                img[y1:y2, x1:x2] = color
                
        return img
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        stats = {
            'training_tasks': len(self.training_challenges),
            'evaluation_tasks': len(self.evaluation_challenges),
            'test_tasks': len(self.test_challenges),
        }
        
        if self.training_challenges:
            train_pairs = sum(len(task.get('train', [])) for task in self.training_challenges.values())
            test_inputs = sum(len(task.get('test', [])) for task in self.training_challenges.values())
            stats['total_train_pairs'] = train_pairs
            stats['total_test_inputs'] = test_inputs
            
        return stats
    
    def get_all_task_ids(self, split: str = "training") -> List[str]:
        """
        Get all task IDs for a given split.
        
        Args:
            split: Dataset split ('training', 'evaluation', 'test')
            
        Returns:
            List of task IDs
        """
        if split == "training":
            return list(self.training_challenges.keys())
        elif split == "evaluation":
            return list(self.evaluation_challenges.keys())
        elif split == "test":
            return list(self.test_challenges.keys())
        return []


def load_submission_template(task_ids: List[str]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    """
    Create a submission template with the required format.
    
    Args:
        task_ids: List of task IDs to include in submission
        
    Returns:
        Submission template dictionary
    """
    template = {}
    for task_id in task_ids:
        # Default to empty 2x2 grid for each attempt
        template[task_id] = [{
            "attempt_1": [[0, 0], [0, 0]],
            "attempt_2": [[0, 0], [0, 0]]
        }]
    return template


def save_submission(submission: Dict[str, List[Dict[str, List[List[int]]]]], 
                   output_path: str = "submission.json") -> None:
    """
    Save submission to JSON file.
    
    Args:
        submission: Submission dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to {output_path}")


def validate_submission(submission: Dict[str, List[Dict[str, List[List[int]]]]]) -> bool:
    """
    Validate submission format.
    
    Args:
        submission: Submission dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        for task_id, predictions in submission.items():
            if not isinstance(predictions, list):
                print(f"Task {task_id}: predictions must be a list")
                return False
                
            for pred in predictions:
                if not isinstance(pred, dict):
                    print(f"Task {task_id}: each prediction must be a dictionary")
                    return False
                    
                if 'attempt_1' not in pred or 'attempt_2' not in pred:
                    print(f"Task {task_id}: missing attempt_1 or attempt_2")
                    return False
                    
                for attempt_key in ['attempt_1', 'attempt_2']:
                    attempt = pred[attempt_key]
                    if not isinstance(attempt, list):
                        print(f"Task {task_id}: {attempt_key} must be a list")
                        return False
                        
                    for row in attempt:
                        if not isinstance(row, list):
                            print(f"Task {task_id}: {attempt_key} rows must be lists")
                            return False
                            
                        for cell in row:
                            if not isinstance(cell, int):
                                print(f"Task {task_id}: {attempt_key} cells must be integers")
                                return False
                                
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def load_sample_submission(sample_file: str = "sample_submission.json") -> Dict[str, List[Dict[str, List[List[int]]]]]:
    """
    Load sample submission file to understand the format.
    
    Args:
        sample_file: Path to sample submission file
        
    Returns:
        Sample submission dictionary
    """
    with open(sample_file, 'r') as f:
        sample = json.load(f)
    return sample 