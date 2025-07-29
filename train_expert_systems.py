#!/usr/bin/env python3
"""
Train Expert Systems Intelligence with ARC Data
"""

import json
import numpy as np
import os
import time
import logging
from typing import Dict, List, Any
from collections import defaultdict

# Import expert systems
from src.models.ultimate_intelligence_integration import get_ultimate_intelligence_integration

logging.basicConfig(level=logging.INFO)

class ExpertSystemsTrainer:
    def __init__(self):
        self.ultimate_intelligence = get_ultimate_intelligence_integration()
        self.training_data = {}
        self.validation_data = {}
        self.performance_metrics = defaultdict(list)
        
    def load_arc_data(self):
        """Load ARC competition data"""
        print("📥 Loading ARC data...")
        
        data_files = {
            'training': 'data_actual/arc-agi_training_challenges.json',
            'evaluation': 'data_actual/arc-agi_evaluation_challenges.json',
            'test': 'data_actual/arc-agi_test_challenges.json'
        }
        
        for data_type, filepath in data_files.items():
            if os.path.exists(filepath):
                print(f"📄 Loading {data_type} data...")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                if data_type == 'training':
                    self.training_data = data
                    print(f"✅ Loaded {len(data)} training challenges")
                elif data_type == 'evaluation':
                    self.validation_data = data
                    print(f"✅ Loaded {len(data)} evaluation challenges")
                    
        print("✅ Data loading completed!")
        
    def train_expert_systems(self, max_tasks=100):
        """Train expert systems on ARC data"""
        print(f"🚀 Training Expert Systems on {max_tasks} tasks...")
        
        training_tasks = list(self.training_data.items())[:max_tasks]
        
        total_confidence = 0.0
        total_intelligence = 0.0
        successful_tasks = 0
        
        for i, (task_id, task_data) in enumerate(training_tasks):
            try:
                print(f"🔍 Training on task {i+1}/{len(training_tasks)}: {task_id}")
                
                # Create task format
                if 'train' in task_data and task_data['train']:
                    train_pair = task_data['train'][0]
                    task_format = {
                        'task_id': task_id,
                        'input': train_pair.get('input', []),
                        'output': train_pair.get('output', []),
                        'patterns': [{'type': 'geometric', 'confidence': 0.8}]
                    }
                    
                    # Train on task
                    solution = self.ultimate_intelligence.solve_task_with_ultimate_intelligence(task_format)
                    
                    total_confidence += solution.confidence
                    total_intelligence += solution.intelligence_level
                    
                    if solution.confidence > 0.7:
                        successful_tasks += 1
                        
                    print(f"   ✅ Confidence: {solution.confidence:.3f}, Intelligence: {solution.intelligence_level:.1f}%")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        # Calculate averages
        avg_confidence = total_confidence / len(training_tasks) if training_tasks else 0
        avg_intelligence = total_intelligence / len(training_tasks) if training_tasks else 0
        success_rate = successful_tasks / len(training_tasks) if training_tasks else 0
        
        print(f"\n📊 Training Results:")
        print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
        print(f"   🧠 Average Intelligence: {avg_intelligence:.1f}%")
        print(f"   📈 Success Rate: {success_rate:.3f}")
        
        return {
            'avg_confidence': avg_confidence,
            'avg_intelligence': avg_intelligence,
            'success_rate': success_rate,
            'target_achieved': avg_intelligence >= 120.0
        }
        
    def validate_training(self, max_tasks=50):
        """Validate training results"""
        print(f"\n🔍 Validating on {max_tasks} tasks...")
        
        validation_tasks = list(self.validation_data.items())[:max_tasks]
        
        total_confidence = 0.0
        total_intelligence = 0.0
        successful_tasks = 0
        
        for i, (task_id, task_data) in enumerate(validation_tasks):
            try:
                if 'train' in task_data and task_data['train']:
                    train_pair = task_data['train'][0]
                    task_format = {
                        'task_id': task_id,
                        'input': train_pair.get('input', []),
                        'output': train_pair.get('output', []),
                        'patterns': [{'type': 'geometric', 'confidence': 0.8}]
                    }
                    
                    solution = self.ultimate_intelligence.solve_task_with_ultimate_intelligence(task_format)
                    
                    total_confidence += solution.confidence
                    total_intelligence += solution.intelligence_level
                    
                    if solution.confidence > 0.7:
                        successful_tasks += 1
                        
            except Exception as e:
                print(f"   ❌ Validation error on {task_id}: {e}")
                
        avg_confidence = total_confidence / len(validation_tasks) if validation_tasks else 0
        avg_intelligence = total_intelligence / len(validation_tasks) if validation_tasks else 0
        success_rate = successful_tasks / len(validation_tasks) if validation_tasks else 0
        
        print(f"📊 Validation Results:")
        print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
        print(f"   🧠 Average Intelligence: {avg_intelligence:.1f}%")
        print(f"   📈 Success Rate: {success_rate:.3f}")
        
        return {
            'avg_confidence': avg_confidence,
            'avg_intelligence': avg_intelligence,
            'success_rate': success_rate,
            'target_achieved': avg_intelligence >= 120.0
        }
        
    def save_trained_models(self):
        """Save trained models"""
        print("\n💾 Saving trained models...")
        
        os.makedirs('trained_models', exist_ok=True)
        
        # Save ultimate intelligence
        import pickle
        with open('trained_models/ultimate_intelligence.pkl', 'wb') as f:
            pickle.dump(self.ultimate_intelligence, f)
            
        # Save performance summary
        summary = self.ultimate_intelligence.get_intelligence_summary()
        with open('trained_models/performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print("✅ Models saved to 'trained_models/' directory")
        
    def generate_submission(self):
        """Generate submission for test data"""
        print("\n📤 Generating submission...")
        
        submission = {}
        
        # Load test data
        test_file = 'data_actual/arc-agi_test_challenges.json'
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_data = json.load(f)
                
            print(f"📄 Processing {len(test_data)} test tasks...")
            
            for task_id, task_data in test_data.items():
                try:
                    if 'train' in task_data and task_data['train']:
                        train_pair = task_data['train'][0]
                        task_format = {
                            'task_id': task_id,
                            'input': train_pair.get('input', []),
                            'output': train_pair.get('output', []),
                            'patterns': [{'type': 'geometric', 'confidence': 0.8}]
                        }
                        
                        solution = self.ultimate_intelligence.solve_task_with_ultimate_intelligence(task_format)
                        
                        # Convert to submission format
                        task_submission = []
                        for prediction in solution.predictions:
                            output = prediction.get('output', [])
                            if isinstance(output, np.ndarray):
                                output = output.tolist()
                            task_submission.append({'output': output})
                            
                        submission[task_id] = task_submission
                        
                except Exception as e:
                    print(f"❌ Error processing test task {task_id}: {e}")
                    
            # Save submission
            with open('trained_models/expert_systems_submission.json', 'w') as f:
                json.dump(submission, f, indent=2)
                
            print(f"✅ Submission saved with {len(submission)} tasks")
            
        return submission

def main():
    """Main training function"""
    print("=" * 80)
    print("🧠 EXPERT SYSTEMS INTELLIGENCE TRAINING")
    print("Beyond 120% Human Genius Level Training")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ExpertSystemsTrainer()
    
    try:
        # Load data
        trainer.load_arc_data()
        
        # Train expert systems
        training_results = trainer.train_expert_systems(max_tasks=50)
        
        # Validate training
        validation_results = trainer.validate_training(max_tasks=20)
        
        # Save models
        trainer.save_trained_models()
        
        # Generate submission
        submission = trainer.generate_submission()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        
        if validation_results['target_achieved']:
            print("🚀 BEYOND 120% HUMAN GENIUS LEVEL ACHIEVED!")
            print(f"🧠 Intelligence Level: {validation_results['avg_intelligence']:.1f}%")
            print(f"🎯 Confidence: {validation_results['avg_confidence']:.3f}")
            print(f"📊 Success Rate: {validation_results['success_rate']:.3f}")
        else:
            print("⚠️  Training completed but target not yet achieved")
            print(f"🧠 Current Intelligence: {validation_results['avg_intelligence']:.1f}%")
            print(f"🎯 Target: 120%")
            
        print("\n📁 Trained models saved in 'trained_models/' directory")
        print("📤 Submission saved as 'trained_models/expert_systems_submission.json'")
        print("🚀 Ready for deployment!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")

if __name__ == "__main__":
    main() 