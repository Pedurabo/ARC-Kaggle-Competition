"""
Main script for ARC Prize 2025 competition.
Orchestrates the complete pipeline from data loading to submission generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import ARCDataset, load_submission_template, save_submission, validate_submission
from models.base_model import SimpleBaselineModel, PatternMatchingModel
from models.advanced_models import SymbolicReasoningModel, EnsembleModel, FewShotLearningModel
from models.breakthrough_model import HumanLevelReasoningModel, BreakthroughEnsemble
from evaluation.scorer import ARCScorer, CrossValidationScorer


def main():
    """Main function to run the ARC Prize solution pipeline."""
    parser = argparse.ArgumentParser(description="ARC Prize 2025 Solution Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Directory containing ARC dataset")
    parser.add_argument("--model", type=str, default="baseline",
                       choices=["baseline", "pattern_matching", "symbolic", "ensemble", "few_shot", "human_reasoning", "breakthrough"],
                       help="Model to use for predictions")
    parser.add_argument("--output", type=str, default="submission.json",
                       help="Output submission file path")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model on training data")
    parser.add_argument("--task_ids", type=str, nargs="+",
                       help="Specific task IDs to process")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ARC PRIZE 2025 - SOLUTION PIPELINE")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = ARCDataset(args.data_dir)
    
    # Load training data
    training_challenges, training_solutions = dataset.load_training_data()
    if not training_challenges:
        print("Warning: No training challenges found!")
        print("Please ensure the dataset is properly downloaded and placed in the data directory.")
        print("Expected files:")
        print("  data/")
        print("    arc-agi_training-challenges.json")
        print("    arc-agi_training-solutions.json")
        print("    arc-agi_evaluation-challenges.json")
        print("    arc-agi_evaluation-solutions.json")
        print("    arc-agi_test-challenges.json")
        print("    sample_submission.json")
        return
    
    print(f"Loaded {len(training_challenges)} training challenges")
    print(f"Loaded {len(training_solutions)} training solutions")
    
    # Load evaluation data
    evaluation_challenges, evaluation_solutions = dataset.load_evaluation_data()
    if evaluation_challenges:
        print(f"Loaded {len(evaluation_challenges)} evaluation challenges")
        print(f"Loaded {len(evaluation_solutions)} evaluation solutions")
    
    # Load test data
    test_challenges = dataset.load_test_data()
    if test_challenges:
        print(f"Loaded {len(test_challenges)} test challenges")
    
    # Initialize model
    print(f"\n2. Initializing {args.model} model...")
    if args.model == "baseline":
        model = SimpleBaselineModel()
    elif args.model == "pattern_matching":
        model = PatternMatchingModel()
    elif args.model == "symbolic":
        model = SymbolicReasoningModel()
    elif args.model == "ensemble":
        model = EnsembleModel()
    elif args.model == "few_shot":
        model = FewShotLearningModel()
    elif args.model == "human_reasoning":
        model = HumanLevelReasoningModel()
    elif args.model == "breakthrough":
        model = BreakthroughEnsemble()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"Model: {model.model_name}")
    
    # Evaluate on training data if requested
    if args.evaluate:
        print("\n3. Evaluating model on training data...")
        cv_scorer = CrossValidationScorer(dataset)
        
        # Use specific task IDs if provided
        task_ids = args.task_ids if args.task_ids else None
        
        results = cv_scorer.evaluate_model(model, task_ids)
        
        print(f"Cross-validation results:")
        print(f"  Overall Score: {results['overall_score']:.4f} ({results['percentage']:.2f}%)")
        print(f"  Correct Outputs: {results['correct_outputs']}/{results['total_outputs']}")
        print(f"  Tasks Evaluated: {results['num_tasks']}")
        
        # Show top performing tasks
        print("\nTop performing tasks:")
        sorted_tasks = sorted(results['task_scores'].items(), 
                            key=lambda x: x[1]['score'], reverse=True)
        for i, (task_id, task_result) in enumerate(sorted_tasks[:10]):
            print(f"  {i+1:2d}. {task_id}: {task_result['score']:.2f} "
                  f"({task_result['correct']}/{task_result['total']})")
    
    # Generate predictions for evaluation tasks
    if evaluation_challenges:
        print(f"\n4. Generating predictions for {len(evaluation_challenges)} evaluation tasks...")
        
        submission = {}
        
        for task_id, task in evaluation_challenges.items():
            print(f"  Processing task {task_id}...")
            
            try:
                # Get predictions for this task
                predictions = model.solve_task(task)
                
                # Validate predictions
                if not model.validate_predictions(predictions):
                    print(f"    Warning: Invalid predictions for task {task_id}")
                    # Use baseline predictions as fallback
                    baseline_model = SimpleBaselineModel()
                    predictions = baseline_model.solve_task(task)
                
                submission[task_id] = predictions
                
            except Exception as e:
                print(f"    Error processing task {task_id}: {e}")
                # Use baseline predictions as fallback
                baseline_model = SimpleBaselineModel()
                predictions = baseline_model.solve_task(task)
                submission[task_id] = predictions
        
        # Validate submission
        print("\n5. Validating submission...")
        if validate_submission(submission):
            print("✓ Submission format is valid")
        else:
            print("✗ Submission format is invalid")
            return
        
        # Save submission
        print(f"\n6. Saving submission to {args.output}...")
        save_submission(submission, args.output)
        
        print(f"\n✓ Submission generated successfully!")
        print(f"  File: {args.output}")
        print(f"  Tasks: {len(submission)}")
        
        # Show submission statistics
        total_predictions = sum(len(predictions) for predictions in submission.values())
        print(f"  Total predictions: {total_predictions}")
        
    # Generate predictions for test tasks (for actual competition)
    elif test_challenges:
        print(f"\n4. Generating predictions for {len(test_challenges)} test tasks...")
        
        submission = {}
        
        for task_id, task in test_challenges.items():
            print(f"  Processing task {task_id}...")
            
            try:
                # Get predictions for this task
                predictions = model.solve_task(task)
                
                # Validate predictions
                if not model.validate_predictions(predictions):
                    print(f"    Warning: Invalid predictions for task {task_id}")
                    # Use baseline predictions as fallback
                    baseline_model = SimpleBaselineModel()
                    predictions = baseline_model.solve_task(task)
                
                submission[task_id] = predictions
                
            except Exception as e:
                print(f"    Error processing task {task_id}: {e}")
                # Use baseline predictions as fallback
                baseline_model = SimpleBaselineModel()
                predictions = baseline_model.solve_task(task)
                submission[task_id] = predictions
        
        # Validate submission
        print("\n5. Validating submission...")
        if validate_submission(submission):
            print("✓ Submission format is valid")
        else:
            print("✗ Submission format is invalid")
            return
        
        # Save submission
        print(f"\n6. Saving submission to {args.output}...")
        save_submission(submission, args.output)
        
        print(f"\n✓ Submission generated successfully!")
        print(f"  File: {args.output}")
        print(f"  Tasks: {len(submission)}")
        
        # Show submission statistics
        total_predictions = sum(len(predictions) for predictions in submission.values())
        print(f"  Total predictions: {total_predictions}")
        
    else:
        print("\nNo evaluation or test data found. Skipping submission generation.")
        print("To generate a submission, ensure evaluation or test data is available.")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


def generate_baseline_submission():
    """Generate a baseline submission for testing."""
    print("Generating baseline submission...")
    
    # Create a simple baseline submission
    baseline_tasks = ["task_001", "task_002", "task_003"]
    submission = load_submission_template(baseline_tasks)
    
    # Save baseline submission
    save_submission(submission, "baseline_submission.json")
    print("Baseline submission saved to baseline_submission.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("If this is your first run, try:")
        print("  1. Download the ARC-AGI-2 dataset from Kaggle")
        print("  2. Place it in the data/ directory")
        print("  3. Run: python src/main.py --evaluate")
        sys.exit(1) 