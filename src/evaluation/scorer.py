"""
Evaluation and scoring utilities for ARC Prize 2025 competition.
Implements the official competition scoring logic.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json


class ARCScorer:
    """
    ARC Prize competition scorer.
    
    Implements the official scoring logic where each task output gets a score of 1
    if any of the 2 predicted outputs matches the ground truth exactly, otherwise 0.
    """
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    def score_submission(self, 
                        submission: Dict[str, List[Dict[str, List[List[int]]]]],
                        ground_truth: Dict[str, List[List[List[int]]]]) -> Dict[str, Any]:
        """
        Score a submission against ground truth.
        
        Args:
            submission: Submission dictionary with predictions
            ground_truth: Ground truth dictionary with correct outputs
            
        Returns:
            Dictionary containing scoring results
        """
        total_outputs = 0
        correct_outputs = 0
        task_scores = {}
        
        for task_id, predictions in submission.items():
            if task_id not in ground_truth:
                print(f"Warning: Task {task_id} not found in ground truth")
                continue
                
            gt_outputs = ground_truth[task_id]
            task_total = len(gt_outputs)
            task_correct = 0
            
            for i, (pred, gt) in enumerate(zip(predictions, gt_outputs)):
                total_outputs += 1
                
                # Check if either attempt matches ground truth
                attempt_1_matches = self._grids_equal(pred['attempt_1'], gt)
                attempt_2_matches = self._grids_equal(pred['attempt_2'], gt)
                
                if attempt_1_matches or attempt_2_matches:
                    correct_outputs += 1
                    task_correct += 1
            
            # Store task-level score
            task_scores[task_id] = {
                'total': task_total,
                'correct': task_correct,
                'score': task_correct / task_total if task_total > 0 else 0.0
            }
        
        # Calculate overall score
        overall_score = correct_outputs / total_outputs if total_outputs > 0 else 0.0
        
        results = {
            'overall_score': overall_score,
            'total_outputs': total_outputs,
            'correct_outputs': correct_outputs,
            'task_scores': task_scores,
            'percentage': overall_score * 100
        }
        
        return results
    
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """
        Check if two grids are exactly equal.
        
        Args:
            grid1: First grid
            grid2: Second grid
            
        Returns:
            True if grids are equal, False otherwise
        """
        if len(grid1) != len(grid2):
            return False
            
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
                
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    return False
                    
        return True
    
    def load_ground_truth(self, ground_truth_path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Load ground truth from file.
        
        Args:
            ground_truth_path: Path to ground truth file
            
        Returns:
            Ground truth dictionary
        """
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        return ground_truth
    
    def evaluate_submission_file(self, 
                                submission_path: str,
                                ground_truth_path: str) -> Dict[str, Any]:
        """
        Evaluate a submission file against ground truth.
        
        Args:
            submission_path: Path to submission JSON file
            ground_truth_path: Path to ground truth JSON file
            
        Returns:
            Evaluation results
        """
        # Load submission
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        # Score submission
        results = self.score_submission(submission, ground_truth)
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a readable format.
        
        Args:
            results: Evaluation results dictionary
        """
        print("=" * 50)
        print("ARC PRIZE 2025 EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Score: {results['overall_score']:.4f} ({results['percentage']:.2f}%)")
        print(f"Correct Outputs: {results['correct_outputs']}/{results['total_outputs']}")
        print()
        
        print("Task-Level Scores:")
        print("-" * 30)
        for task_id, task_result in results['task_scores'].items():
            print(f"{task_id}: {task_result['correct']}/{task_result['total']} "
                  f"({task_result['score']:.2f})")
        
        print()
        print("=" * 50)


class CrossValidationScorer:
    """
    Cross-validation scorer for evaluating models on training data.
    """
    
    def __init__(self, dataset):
        """
        Initialize cross-validation scorer.
        
        Args:
            dataset: ARCDataset instance
        """
        self.dataset = dataset
    
    def evaluate_model(self, model, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a model using cross-validation on training tasks.
        
        Args:
            model: Model to evaluate
            task_ids: List of task IDs to evaluate (None for all)
            
        Returns:
            Evaluation results
        """
        if task_ids is None:
            task_ids = list(self.dataset.training_challenges.keys())
        
        total_outputs = 0
        correct_outputs = 0
        task_scores = {}
        
        for task_id in task_ids:
            task = self.dataset.get_task_by_id(task_id, 'training')
            if not task:
                continue
            
            # Get predictions for this task
            predictions = model.solve_task(task)
            
            # Get ground truth (test outputs from training solutions)
            gt_outputs = self.dataset.get_solution_by_id(task_id, 'training')
            if not gt_outputs:
                continue
            
            task_total = len(gt_outputs)
            task_correct = 0
            
            for i, (pred, gt) in enumerate(zip(predictions, gt_outputs)):
                total_outputs += 1
                
                # Check if either attempt matches ground truth
                attempt_1_matches = self._grids_equal(pred['attempt_1'], gt)
                attempt_2_matches = self._grids_equal(pred['attempt_2'], gt)
                
                if attempt_1_matches or attempt_2_matches:
                    correct_outputs += 1
                    task_correct += 1
            
            # Store task-level score
            task_scores[task_id] = {
                'total': task_total,
                'correct': task_correct,
                'score': task_correct / task_total if task_total > 0 else 0.0
            }
        
        # Calculate overall score
        overall_score = correct_outputs / total_outputs if total_outputs > 0 else 0.0
        
        results = {
            'overall_score': overall_score,
            'total_outputs': total_outputs,
            'correct_outputs': correct_outputs,
            'task_scores': task_scores,
            'percentage': overall_score * 100,
            'num_tasks': len(task_scores)
        }
        
        return results
    
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """
        Check if two grids are exactly equal.
        
        Args:
            grid1: First grid
            grid2: Second grid
            
        Returns:
            True if grids are equal, False otherwise
        """
        if len(grid1) != len(grid2):
            return False
            
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
                
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    return False
                    
        return True 