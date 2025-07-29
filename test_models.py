#!/usr/bin/env python3
"""
Comprehensive model testing script for ARC Prize 2025.
Tests all available models and compares their performance.
"""

import sys
from pathlib import Path
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.data_loader import ARCDataset
from models.base_model import SimpleBaselineModel, PatternMatchingModel
from models.advanced_models import SymbolicReasoningModel, EnsembleModel, FewShotLearningModel
from evaluation.scorer import CrossValidationScorer


def test_all_models():
    """Test all available models and compare performance."""
    print("=" * 60)
    print("ARC PRIZE 2025 - COMPREHENSIVE MODEL TESTING")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = ARCDataset('data')
    
    # Try to load training data
    try:
        training_challenges, training_solutions = dataset.load_training_data()
        if not training_challenges:
            print("No training data found. Creating sample data...")
            import subprocess
            subprocess.run(['python', 'download_dataset.py', '--create-sample'], check=True)
            training_challenges, training_solutions = dataset.load_training_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample data...")
        import subprocess
        subprocess.run(['python', 'download_dataset.py', '--create-sample'], check=True)
        training_challenges, training_solutions = dataset.load_training_data()
    
    print(f"Loaded {len(training_challenges)} training challenges")
    
    # Define all models to test
    models = {
        'baseline': SimpleBaselineModel(),
        'pattern_matching': PatternMatchingModel(),
        'symbolic': SymbolicReasoningModel(),
        'ensemble': EnsembleModel(),
        'few_shot': FewShotLearningModel()
    }
    
    # Test each model
    results = {}
    cv_scorer = CrossValidationScorer(dataset)
    
    # Use a subset of tasks for faster testing
    test_task_ids = list(training_challenges.keys())[:5]
    print(f"\nTesting on {len(test_task_ids)} tasks: {test_task_ids}")
    
    for model_name, model in models.items():
        print(f"\n2. Testing {model_name} model...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = cv_scorer.evaluate_model(model, test_task_ids)
            end_time = time.time()
            
            results[model_name] = {
                'score': result['overall_score'],
                'percentage': result['percentage'],
                'correct': result['correct_outputs'],
                'total': result['total_outputs'],
                'time': end_time - start_time,
                'tasks_evaluated': result['num_tasks']
            }
            
            print(f"  Score: {result['overall_score']:.4f} ({result['percentage']:.2f}%)")
            print(f"  Correct: {result['correct_outputs']}/{result['total_outputs']}")
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  Tasks: {result['num_tasks']}")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            results[model_name] = {
                'score': 0.0,
                'percentage': 0.0,
                'correct': 0,
                'total': 0,
                'time': 0.0,
                'tasks_evaluated': 0,
                'error': str(e)
            }
    
    # Display results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort models by score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"{'Model':<15} {'Score':<8} {'%':<6} {'Correct':<10} {'Time':<8} {'Tasks':<8}")
    print("-" * 60)
    
    for model_name, result in sorted_models:
        if 'error' in result:
            print(f"{model_name:<15} {'ERROR':<8} {'-':<6} {'-':<10} {'-':<8} {'-':<8}")
        else:
            print(f"{model_name:<15} {result['score']:<8.4f} {result['percentage']:<6.2f} "
                  f"{result['correct']:<10} {result['time']:<8.2f} {result['tasks_evaluated']:<8}")
    
    # Best performing model
    best_model = sorted_models[0]
    print(f"\n🏆 Best performing model: {best_model[0]} ({best_model[1]['percentage']:.2f}%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("1. MODEL SELECTION:")
    for i, (model_name, result) in enumerate(sorted_models[:3]):
        print(f"   {i+1}. {model_name}: {result['percentage']:.2f}% accuracy")
    
    print("\n2. NEXT STEPS:")
    print("   - Test on full dataset for more accurate results")
    print("   - Implement more sophisticated transformation detection")
    print("   - Add neural network components")
    print("   - Optimize ensemble weights")
    print("   - Focus on novel reasoning capabilities")
    
    print("\n3. COMPETITION STRATEGY:")
    print("   - Use ensemble approach for robustness")
    print("   - Implement confidence-based predictions")
    print("   - Leverage the 2-attempt format strategically")
    print("   - Focus on generalization, not memorization")
    
    # Save results
    results_file = "model_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


def test_specific_model(model_name: str):
    """Test a specific model in detail."""
    print(f"\nDetailed testing of {model_name} model...")
    
    # Load dataset
    dataset = ARCDataset('data')
    training_challenges, training_solutions = dataset.load_training_data()
    
    # Initialize model
    models = {
        'baseline': SimpleBaselineModel(),
        'pattern_matching': PatternMatchingModel(),
        'symbolic': SymbolicReasoningModel(),
        'ensemble': EnsembleModel(),
        'few_shot': FewShotLearningModel()
    }
    
    if model_name not in models:
        print(f"Unknown model: {model_name}")
        return
    
    model = models[model_name]
    
    # Test on a few specific tasks
    test_tasks = list(training_challenges.keys())[:3]
    
    for task_id in test_tasks:
        print(f"\nTesting task {task_id}:")
        task = training_challenges[task_id]
        
        try:
            predictions = model.solve_task(task)
            print(f"  Generated {len(predictions)} predictions")
            
            # Show first prediction
            if predictions:
                pred = predictions[0]
                print(f"  Attempt 1: {pred['attempt_1']}")
                print(f"  Attempt 2: {pred['attempt_2']}")
        
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ARC Prize 2025 models")
    parser.add_argument("--model", type=str, help="Test specific model")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer tasks")
    
    args = parser.parse_args()
    
    if args.model:
        test_specific_model(args.model)
    else:
        test_all_models() 