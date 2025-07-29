"""
Base model class for ARC Prize 2025 competition.
Defines the interface that all ARC solving models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BaseARCModel(ABC):
    """
    Abstract base class for ARC solving models.
    
    All models must implement the solve_task method to generate predictions
    for ARC tasks.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            model_config: Configuration dictionary for the model
        """
        self.model_config = model_config or {}
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def solve_task(self, task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """
        Solve an ARC task and return predictions.
        
        Args:
            task: Task dictionary containing train and test pairs
            
        Returns:
            List of prediction dictionaries, each with 'attempt_1' and 'attempt_2'
        """
        pass
    
    def preprocess_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a task for the model.
        
        Args:
            task: Raw task dictionary
            
        Returns:
            Preprocessed task dictionary
        """
        # Default implementation - can be overridden by subclasses
        return task
    
    def postprocess_predictions(self, predictions: List[List[List[int]]]) -> List[Dict[str, List[List[int]]]]:
        """
        Postprocess model predictions into the required format.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Formatted predictions with attempt_1 and attempt_2
        """
        formatted_predictions = []
        
        for pred in predictions:
            # Ensure we have exactly 2 attempts
            if len(pred) == 1:
                # Duplicate the single prediction for attempt_2
                formatted_pred = {
                    "attempt_1": pred[0],
                    "attempt_2": pred[0]
                }
            elif len(pred) >= 2:
                # Use first two predictions
                formatted_pred = {
                    "attempt_1": pred[0],
                    "attempt_2": pred[1]
                }
            else:
                # Fallback to empty grid
                formatted_pred = {
                    "attempt_1": [[0]],
                    "attempt_2": [[0]]
                }
            
            formatted_predictions.append(formatted_pred)
            
        return formatted_predictions
    
    def validate_predictions(self, predictions: List[Dict[str, List[List[int]]]]) -> bool:
        """
        Validate that predictions are in the correct format.
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            for pred in predictions:
                if not isinstance(pred, dict):
                    return False
                    
                if 'attempt_1' not in pred or 'attempt_2' not in pred:
                    return False
                    
                for attempt_key in ['attempt_1', 'attempt_2']:
                    attempt = pred[attempt_key]
                    if not isinstance(attempt, list):
                        return False
                        
                    for row in attempt:
                        if not isinstance(row, list):
                            return False
                            
                        for cell in row:
                            if not isinstance(cell, int):
                                return False
                                
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_config': self.model_config,
            'model_type': 'abstract'
        }


class SimpleBaselineModel(BaseARCModel):
    """
    Simple baseline model that returns empty grids.
    Useful for testing the submission pipeline.
    """
    
    def solve_task(self, task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """
        Solve task with simple baseline approach.
        
        Args:
            task: Task dictionary
            
        Returns:
            List of prediction dictionaries
        """
        # Get number of test inputs
        test_inputs = task.get('test', [])
        num_test_inputs = len(test_inputs)
        
        predictions = []
        for _ in range(num_test_inputs):
            # Create empty 2x2 grid for each test input
            empty_grid = [[0, 0], [0, 0]]
            pred = {
                "attempt_1": empty_grid,
                "attempt_2": empty_grid
            }
            predictions.append(pred)
            
        return predictions


class PatternMatchingModel(BaseARCModel):
    """
    Pattern matching model that looks for simple transformations.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        self.supported_transformations = [
            'identity',
            'rotation_90',
            'rotation_180', 
            'rotation_270',
            'flip_horizontal',
            'flip_vertical',
            'translation'
        ]
    
    def solve_task(self, task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """
        Solve task using pattern matching approach.
        
        Args:
            task: Task dictionary
            
        Returns:
            List of prediction dictionaries
        """
        train_pairs = task.get('train', [])
        test_inputs = task.get('test', [])
        
        if not train_pairs:
            # No training data, return baseline
            return self._baseline_predictions(len(test_inputs))
        
        # Analyze training pairs to find patterns
        patterns = self._analyze_patterns(train_pairs)
        
        predictions = []
        for test_input in test_inputs:
            # Apply best matching pattern
            prediction = self._apply_pattern(test_input, patterns)
            pred_dict = {
                "attempt_1": prediction,
                "attempt_2": prediction  # Same prediction for both attempts
            }
            predictions.append(pred_dict)
            
        return predictions
    
    def _analyze_patterns(self, train_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze training pairs to identify patterns.
        
        Args:
            train_pairs: List of training input-output pairs
            
        Returns:
            Dictionary of identified patterns
        """
        patterns = {
            'transformation_type': 'identity',
            'confidence': 0.0
        }
        
        # Simple pattern analysis - can be enhanced
        for pair in train_pairs:
            input_grid = pair['input']
            output_grid = pair['output']
            
            # Check for identity transformation
            if input_grid == output_grid:
                patterns['transformation_type'] = 'identity'
                patterns['confidence'] += 1.0
                
        patterns['confidence'] /= len(train_pairs)
        return patterns
    
    def _apply_pattern(self, test_input: List[List[int]], patterns: Dict[str, Any]) -> List[List[int]]:
        """
        Apply identified pattern to test input.
        
        Args:
            test_input: Test input grid
            patterns: Identified patterns
            
        Returns:
            Predicted output grid
        """
        transformation_type = patterns.get('transformation_type', 'identity')
        
        if transformation_type == 'identity':
            return [row[:] for row in test_input]  # Copy the input
        else:
            # For now, return input as-is
            return [row[:] for row in test_input]
    
    def _baseline_predictions(self, num_test_inputs: int) -> List[Dict[str, List[List[int]]]]:
        """
        Generate baseline predictions.
        
        Args:
            num_test_inputs: Number of test inputs
            
        Returns:
            List of baseline predictions
        """
        predictions = []
        for _ in range(num_test_inputs):
            pred = {
                "attempt_1": [[0, 0], [0, 0]],
                "attempt_2": [[0, 0], [0, 0]]
            }
            predictions.append(pred)
        return predictions 