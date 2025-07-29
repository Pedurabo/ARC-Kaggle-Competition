#!/usr/bin/env python3
"""
Training script for breakthrough modules to bridge the 80% gap to 95% performance.
Implements advanced training techniques including curriculum learning, meta-learning, and ensemble optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Any, Optional
import wandb
from tqdm import tqdm
import optuna

# Import our modules
from src.models.breakthrough_modules import (
    AbstractReasoningModule, AdvancedMetaLearner, MultiModalReasoner,
    ConceptLearner, RuleInductor, AnalogyMaker, CreativeSolver
)
from src.utils.data_loader import ARCDataset
from src.evaluation.scorer import CrossValidationScorer


class BreakthroughDataset(Dataset):
    """Dataset for training breakthrough modules."""
    
    def __init__(self, data: List[Dict], difficulty_estimator=None):
        self.data = data
        self.difficulty_estimator = difficulty_estimator or self._simple_difficulty_estimator
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task = self.data[idx]
        difficulty = self.difficulty_estimator(task)
        return {
            'task': task,
            'difficulty': difficulty,
            'task_id': task.get('task_id', str(idx))
        }
    
    def _simple_difficulty_estimator(self, task: Dict) -> float:
        """Simple difficulty estimation based on task characteristics."""
        train_pairs = task.get('train', [])
        test_pairs = task.get('test', [])
        
        # Factors that increase difficulty
        difficulty = 0.0
        
        # More training examples = easier
        difficulty += len(train_pairs) * 0.1
        
        # More test examples = harder
        difficulty += len(test_pairs) * 0.2
        
        # Larger grids = harder
        for pair in train_pairs:
            input_grid = pair['input']
            output_grid = pair['output']
            max_size = max(len(input_grid), len(output_grid))
            if input_grid and input_grid[0]:
                max_size = max(max_size, len(input_grid[0]))
            if output_grid and output_grid[0]:
                max_size = max(max_size, len(output_grid[0]))
            difficulty += max_size * 0.01
        
        # Normalize to 0-1
        return min(1.0, difficulty / 10.0)


class CurriculumScheduler:
    """Schedules curriculum learning based on difficulty."""
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty
    
    def get_current_difficulty(self) -> float:
        return self.current_difficulty
    
    def step(self, performance: float):
        """Update difficulty based on performance."""
        if performance > 0.8:
            # Good performance, increase difficulty
            self.current_difficulty = min(self.max_difficulty, self.current_difficulty + 0.1)
        elif performance < 0.3:
            # Poor performance, decrease difficulty
            self.current_difficulty = max(self.initial_difficulty, self.current_difficulty - 0.05)


class BreakthroughTrainer:
    """Advanced trainer for breakthrough modules."""
    
    def __init__(self, model_type: str = "abstract_reasoning", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == "abstract_reasoning":
            self.model = AbstractReasoningModule()
        elif model_type == "meta_learning":
            self.model = AdvancedMetaLearner()
        elif model_type == "multi_modal":
            self.model = MultiModalReasoner()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Curriculum learning
        self.curriculum_scheduler = CurriculumScheduler()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'difficulty': []
        }
        
        # Initialize wandb (disabled for now)
        try:
            wandb.init(project="arc-breakthrough", name=f"{model_type}_training", mode="disabled")
        except:
            print("⚠️  wandb disabled, continuing without experiment tracking")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            tasks = batch['task']
            difficulties = batch['difficulty']
            
            # Filter by current curriculum difficulty
            current_difficulty = self.curriculum_scheduler.get_current_difficulty()
            valid_indices = [i for i, diff in enumerate(difficulties) 
                           if diff <= current_difficulty]
            
            if not valid_indices:
                continue
            
            valid_tasks = [tasks[i] for i in valid_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            for task in valid_tasks:
                try:
                    # Get model prediction
                    if self.model_type == "abstract_reasoning":
                        prediction = self.model(task)
                    elif self.model_type == "meta_learning":
                        train_examples = task.get('train', [])
                        prediction = self.model.adapt_to_new_task(train_examples)
                    elif self.model_type == "multi_modal":
                        prediction = self.model.reason(task)
                    
                    # Calculate loss (simplified)
                    loss = self._calculate_loss(prediction, task)
                    batch_loss += loss
                    
                    # Calculate accuracy (simplified)
                    accuracy = self._calculate_accuracy(prediction, task)
                    batch_correct += accuracy
                    batch_total += 1
                    
                except Exception as e:
                    print(f"Error processing task: {e}")
                    continue
            
            if batch_total > 0:
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += batch_loss.item()
                correct_predictions += batch_correct
                total_predictions += batch_total
                
                # Update progress bar
                avg_loss = total_loss / (len(progress_bar) + 1)
                avg_acc = correct_predictions / max(total_predictions, 1)
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.4f}',
                    'difficulty': f'{current_difficulty:.2f}'
                })
        
        # Update curriculum based on performance
        if total_predictions > 0:
            performance = correct_predictions / total_predictions
            self.curriculum_scheduler.step(performance)
        
        return {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            'accuracy': correct_predictions / max(total_predictions, 1),
            'difficulty': current_difficulty
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                tasks = batch['task']
                
                for task in tasks:
                    try:
                        # Get model prediction
                        if self.model_type == "abstract_reasoning":
                            prediction = self.model(task)
                        elif self.model_type == "meta_learning":
                            train_examples = task.get('train', [])
                            prediction = self.model.adapt_to_new_task(train_examples)
                        elif self.model_type == "multi_modal":
                            prediction = self.model.reason(task)
                        
                        # Calculate loss and accuracy
                        loss = self._calculate_loss(prediction, task)
                        accuracy = self._calculate_accuracy(prediction, task)
                        
                        total_loss += loss.item()
                        correct_predictions += accuracy
                        total_predictions += 1
                        
                    except Exception as e:
                        print(f"Error processing task: {e}")
                        continue
        
        return {
            'loss': total_loss / max(total_predictions, 1),
            'accuracy': correct_predictions / max(total_predictions, 1)
        }
    
    def _calculate_loss(self, prediction: torch.Tensor, task: Dict) -> torch.Tensor:
        """Calculate loss for a prediction."""
        # Simplified loss calculation
        # In practice, this would be more sophisticated
        test_pairs = task.get('test', [])
        
        if not test_pairs:
            return torch.tensor(0.0, device=self.device)
        
        # Convert prediction to grid format
        predicted_grid = self._tensor_to_grid(prediction)
        
        # Calculate loss based on ground truth
        total_loss = 0.0
        for test_pair in test_pairs:
            expected_output = test_pair.get('output', [])
            if expected_output:
                # Simple MSE loss
                expected_tensor = torch.tensor(expected_output, dtype=torch.float, device=self.device)
                predicted_tensor = torch.tensor(predicted_grid, dtype=torch.float, device=self.device)
                
                # Pad to same size
                max_rows = max(len(expected_output), len(predicted_grid))
                max_cols = max(
                    max(len(row) for row in expected_output) if expected_output else 0,
                    max(len(row) for row in predicted_grid) if predicted_grid else 0
                )
                
                expected_padded = torch.zeros(max_rows, max_cols, device=self.device)
                predicted_padded = torch.zeros(max_rows, max_cols, device=self.device)
                
                for i, row in enumerate(expected_output):
                    for j, val in enumerate(row):
                        expected_padded[i, j] = val
                
                for i, row in enumerate(predicted_grid):
                    for j, val in enumerate(row):
                        predicted_padded[i, j] = val
                
                loss = F.mse_loss(predicted_padded, expected_padded)
                total_loss += loss
        
        return torch.tensor(total_loss, device=self.device)
    
    def _calculate_accuracy(self, prediction: torch.Tensor, task: Dict) -> int:
        """Calculate accuracy for a prediction."""
        # Simplified accuracy calculation
        # In practice, this would use the actual evaluation metric
        test_pairs = task.get('test', [])
        
        if not test_pairs:
            return 0
        
        predicted_grid = self._tensor_to_grid(prediction)
        
        correct = 0
        for test_pair in test_pairs:
            expected_output = test_pair.get('output', [])
            if expected_output == predicted_grid:
                correct += 1
        
        return correct
    
    def _tensor_to_grid(self, tensor: torch.Tensor) -> List[List[int]]:
        """Convert tensor to grid format."""
        # Simplified conversion
        # In practice, this would be more sophisticated
        if tensor.dim() == 1:
            # Flattened tensor
            size = int(np.sqrt(tensor.numel()))
            grid = tensor.view(size, size).cpu().numpy()
        elif tensor.dim() == 2:
            # 2D tensor
            grid = tensor.cpu().numpy()
        else:
            # Higher dimensional tensor
            grid = tensor.mean(dim=0).cpu().numpy()
        
        # Convert to integer grid
        grid = np.round(grid).astype(int)
        grid = np.clip(grid, 0, 9)  # ARC uses values 0-9
        
        return grid.tolist()
    
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              epochs: int = 100, batch_size: int = 8) -> Dict[str, List[float]]:
        """Train the model."""
        print(f"Training {self.model_type} model for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        # Create datasets and dataloaders
        train_dataset = BreakthroughDataset(train_data)
        val_dataset = BreakthroughDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['difficulty'].append(train_metrics['difficulty'])
            
            # Log to wandb (if available)
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': val_metrics['accuracy'],
                    'difficulty': train_metrics['difficulty'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            except:
                pass  # wandb not available
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Difficulty: {train_metrics['difficulty']:.2f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'history': self.history
                }, f'best_{self.model_type}_model.pth')
                print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from {checkpoint_path}")


def objective(trial):
    """Objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical('d_model', [256, 512, 1024])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    # Create trainer with optimized parameters
    trainer = BreakthroughTrainer(model_type="abstract_reasoning")
    
    # Modify model architecture
    trainer.model = AbstractReasoningModule(d_model=d_model)
    trainer.model.to(trainer.device)
    
    # Update optimizer
    trainer.optimizer = optim.AdamW(trainer.model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Train for a few epochs
    history = trainer.train(train_data, val_data, epochs=10, batch_size=batch_size)
    
    # Return validation accuracy
    return max(history['val_acc'])


def main():
    parser = argparse.ArgumentParser(description="Train breakthrough modules")
    parser.add_argument("--model_type", type=str, default="abstract_reasoning",
                       choices=["abstract_reasoning", "meta_learning", "multi_modal"],
                       help="Type of breakthrough model to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--optimize_hyperparameters", action="store_true",
                       help="Use Optuna for hyperparameter optimization")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading ARC dataset...")
    try:
        dataset = ARCDataset('data')
        train_data, _ = dataset.load_training_data()
        eval_data, _ = dataset.load_evaluation_data()
        
        # Split evaluation data for validation
        split_idx = len(eval_data) // 2
        val_data = eval_data[:split_idx]
        test_data = eval_data[split_idx:]
        
        print(f"Train data: {len(train_data)} samples")
        print(f"Val data: {len(val_data)} samples")
        print(f"Test data: {len(test_data)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data for testing...")
        
        # Create sample data
        train_data = []
        for i in range(100):
            train_data.append({
                'task_id': f'train_{i}',
                'train': [
                    {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                    {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
                ],
                'test': [
                    {'input': [[0, 0], [1, 1]], 'output': [[1, 1], [0, 0]]}
                ]
            })
        
        val_data = train_data[:20]
        test_data = train_data[20:40]
    
    # Hyperparameter optimization
    if args.optimize_hyperparameters:
        print("Running hyperparameter optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print("Best hyperparameters:")
        print(study.best_params)
        print(f"Best validation accuracy: {study.best_value:.4f}")
        
        # Train with best hyperparameters
        best_params = study.best_params
        trainer = BreakthroughTrainer(model_type=args.model_type, device=args.device)
        
        # Apply best parameters
        trainer.model = AbstractReasoningModule(d_model=best_params['d_model'])
        trainer.model.to(trainer.device)
        trainer.optimizer = optim.AdamW(trainer.model.parameters(), 
                                      lr=best_params['lr'], weight_decay=1e-5)
        
    else:
        # Standard training
        trainer = BreakthroughTrainer(model_type=args.model_type, device=args.device)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train the model
    history = trainer.train(train_data, val_data, epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = BreakthroughDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_metrics = trainer.validate(test_loader)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history,
        'test_metrics': test_metrics
    }, f'final_{args.model_type}_model.pth')
    
    print(f"Training complete! Final model saved as final_{args.model_type}_model.pth")


if __name__ == "__main__":
    main() 