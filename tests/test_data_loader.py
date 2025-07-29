"""
Unit tests for data loader functionality.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.data_loader import ARCDataset, load_submission_template, save_submission, validate_submission


class TestARCDataset(unittest.TestCase):
    """Test cases for ARCDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create test data structure
        self.create_test_data()
        
        # Initialize dataset
        self.dataset = ARCDataset(str(self.data_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Create test data files."""
        # Create training data
        training_dir = self.data_dir / "training"
        training_dir.mkdir(exist_ok=True)
        
        training_data = {
            "test_task_1": {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[1, 0], [0, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[0, 1], [1, 0]]
                    }
                ]
            },
            "test_task_2": {
                "train": [
                    {
                        "input": [[0, 0], [0, 0]],
                        "output": [[1, 1], [1, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 1], [1, 1]],
                        "output": [[0, 0], [0, 0]]
                    }
                ]
            }
        }
        
        with open(training_dir / "training.json", 'w') as f:
            json.dump(training_data, f)
        
        # Create evaluation data
        eval_dir = self.data_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        eval_data = {
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
        
        with open(eval_dir / "evaluation.json", 'w') as f:
            json.dump(eval_data, f)
    
    def test_load_training_data(self):
        """Test loading training data."""
        training_data = self.dataset.load_training_data()
        
        self.assertIsInstance(training_data, dict)
        self.assertEqual(len(training_data), 2)
        self.assertIn("test_task_1", training_data)
        self.assertIn("test_task_2", training_data)
    
    def test_load_evaluation_data(self):
        """Test loading evaluation data."""
        eval_data = self.dataset.load_evaluation_data()
        
        self.assertIsInstance(eval_data, dict)
        self.assertEqual(len(eval_data), 1)
        self.assertIn("eval_task_1", eval_data)
    
    def test_get_task_by_id(self):
        """Test getting task by ID."""
        # Load data first
        self.dataset.load_training_data()
        
        # Test existing task
        task = self.dataset.get_task_by_id("test_task_1", "training")
        self.assertIsNotNone(task)
        self.assertIn("train", task)
        self.assertIn("test", task)
        
        # Test non-existing task
        task = self.dataset.get_task_by_id("non_existing", "training")
        self.assertIsNone(task)
    
    def test_get_task_statistics(self):
        """Test getting task statistics."""
        # Load data first
        self.dataset.load_training_data()
        self.dataset.load_evaluation_data()
        
        stats = self.dataset.get_task_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['training_tasks'], 2)
        self.assertEqual(stats['evaluation_tasks'], 1)
        self.assertEqual(stats['test_tasks'], 0)
        self.assertEqual(stats['total_train_pairs'], 2)
        self.assertEqual(stats['total_test_pairs'], 2)
    
    def test_grid_to_image(self):
        """Test grid to image conversion."""
        grid = [[0, 1], [1, 0]]
        image = self.dataset.grid_to_image(grid, cell_size=10)
        
        self.assertIsInstance(image, type(image))
        self.assertEqual(image.shape, (20, 20, 3))  # 2x2 grid with 10px cells
    
    def test_visualize_task(self):
        """Test task visualization (should not raise errors)."""
        # Load data first
        self.dataset.load_training_data()
        
        # This should not raise any errors
        try:
            self.dataset.visualize_task("test_task_1", "training")
        except Exception as e:
            self.fail(f"visualize_task raised an exception: {e}")


class TestSubmissionUtils(unittest.TestCase):
    """Test cases for submission utility functions."""
    
    def test_load_submission_template(self):
        """Test loading submission template."""
        task_ids = ["task_1", "task_2", "task_3"]
        template = load_submission_template(task_ids)
        
        self.assertIsInstance(template, dict)
        self.assertEqual(len(template), 3)
        
        for task_id in task_ids:
            self.assertIn(task_id, template)
            self.assertIsInstance(template[task_id], list)
            self.assertEqual(len(template[task_id]), 1)
            
            prediction = template[task_id][0]
            self.assertIn("attempt_1", prediction)
            self.assertIn("attempt_2", prediction)
            self.assertEqual(prediction["attempt_1"], [[0, 0], [0, 0]])
            self.assertEqual(prediction["attempt_2"], [[0, 0], [0, 0]])
    
    def test_save_and_load_submission(self):
        """Test saving and loading submission."""
        # Create test submission
        submission = {
            "task_1": [{"attempt_1": [[0, 1], [1, 0]], "attempt_2": [[1, 0], [0, 1]]}],
            "task_2": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[1, 1], [1, 1]]}]
        }
        
        # Save submission
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        
        try:
            save_submission(submission, temp_file.name)
            
            # Load and verify
            with open(temp_file.name, 'r') as f:
                loaded_submission = json.load(f)
            
            self.assertEqual(submission, loaded_submission)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_validate_submission_valid(self):
        """Test validation of valid submission."""
        valid_submission = {
            "task_1": [{"attempt_1": [[0, 1], [1, 0]], "attempt_2": [[1, 0], [0, 1]]}],
            "task_2": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[1, 1], [1, 1]]}]
        }
        
        self.assertTrue(validate_submission(valid_submission))
    
    def test_validate_submission_invalid(self):
        """Test validation of invalid submissions."""
        # Missing attempt_1
        invalid_1 = {
            "task_1": [{"attempt_2": [[0, 0], [0, 0]]}]
        }
        self.assertFalse(validate_submission(invalid_1))
        
        # Missing attempt_2
        invalid_2 = {
            "task_1": [{"attempt_1": [[0, 0], [0, 0]]}]
        }
        self.assertFalse(validate_submission(invalid_2))
        
        # Non-integer cells
        invalid_3 = {
            "task_1": [{"attempt_1": [[0, "1"], [1, 0]], "attempt_2": [[1, 0], [0, 1]]}]
        }
        self.assertFalse(validate_submission(invalid_3))
        
        # Non-list rows
        invalid_4 = {
            "task_1": [{"attempt_1": [0, 1], "attempt_2": [[1, 0], [0, 1]]}]
        }
        self.assertFalse(validate_submission(invalid_4))


if __name__ == '__main__':
    unittest.main() 