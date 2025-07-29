#!/usr/bin/env python3
"""
Training script for breakthrough models to achieve 95% performance on ARC Prize 2025.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.data_loader import ARCDataset
from models.breakthrough_model import HumanLevelReasoningModel, BreakthroughEnsemble
from evaluation.scorer import CrossValidationScorer


class BreakthroughTrainer:
    """Trainer for breakthrough models."""
    
    def __init__(self, model_type: str = "human_reasoning", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Initialize model
        if model_type == "human_reasoning":
            self.model = HumanLevelReasoningModel()
        elif model_type == "breakthrough":
            self.model = BreakthroughEnsemble()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def prepare_training_data(self, dataset: ARCDataset):
        """Prepare training data for the model."""
        training_challenges, training_solutions = dataset.load_training_data()
        
        # Convert to training format
        training_data = []
        for task_id, task in training_challenges.items():
            solution = training_solutions.get(task_id)
            if solution:
                training_data.append({
                    'task_id': task_id,
                    'task': task,
                    'solution': solution
                })
        
        return training_data
    
    def train_epoch(self, training_data: list) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        
        progress_bar = tqdm(training_data, desc="Training")
        
        for data in progress_bar:
            task = data['task']
            solution = data['solution']
            
            try:
                # Get model predictions
                predictions = self.model.solve_task(task)
                
                # Calculate loss (simplified - in practice, you'd need more sophisticated loss)
                loss = self.calculate_loss(predictions, solution)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(predictions, solution)
                total_correct += accuracy['correct']
                total_predictions += accuracy['total']
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy['accuracy']:.2f}"
                })
                
            except Exception as e:
                print(f"Warning: Error training on task {data['task_id']}: {e}")
                continue
        
        avg_loss = total_loss / len(training_data)
        avg_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def validate(self, validation_data: list) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data in validation_data:
                task = data['task']
                solution = data['solution']
                
                try:
                    # Get model predictions
                    predictions = self.model.solve_task(task)
                    
                    # Calculate loss
                    loss = self.calculate_loss(predictions, solution)
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    accuracy = self.calculate_accuracy(predictions, solution)
                    total_correct += accuracy['correct']
                    total_predictions += accuracy['total']
                    
                except Exception as e:
                    print(f"Warning: Error validating on task {data['task_id']}: {e}")
                    continue
        
        avg_loss = total_loss / len(validation_data)
        avg_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def calculate_loss(self, predictions: list, solution: list) -> torch.Tensor:
        """Calculate loss between predictions and solution."""
        # Simplified loss calculation
        # In practice, you'd need more sophisticated loss functions
        
        total_loss = 0.0
        num_predictions = 0
        
        for pred, sol in zip(predictions, solution):
            # Convert predictions to tensors
            pred_tensor = torch.tensor(pred['attempt_1'], dtype=torch.long)
            sol_tensor = torch.tensor(sol, dtype=torch.long)
            
            # Flatten for loss calculation
            pred_flat = pred_tensor.flatten()
            sol_flat = sol_tensor.flatten()
            
            # Calculate cross-entropy loss
            loss = self.criterion(pred_flat.float(), sol_flat)
            total_loss += loss
            num_predictions += 1
        
        return total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0)
    
    def calculate_accuracy(self, predictions: list, solution: list) -> dict:
        """Calculate accuracy between predictions and solution."""
        correct = 0
        total = 0
        
        for pred, sol in zip(predictions, solution):
            pred_grid = pred['attempt_1']
            sol_grid = sol
            
            if pred_grid == sol_grid:
                correct += 1
            total += 1
        
        return {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def save_checkpoint(self, epoch: int, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.model_type}_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_type': self.model_type
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded: {filename}")
        return checkpoint['epoch']
    
    def train(self, dataset: ARCDataset, epochs: int = 100, 
              validation_split: float = 0.2, save_interval: int = 10):
        """Train the model."""
        print(f"Training {self.model_type} model for {epochs} epochs")
        print(f"Device: {self.device}")
        
        # Prepare training data
        training_data = self.prepare_training_data(dataset)
        
        # Split into train/validation
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_data)
            
            # Validation
            val_loss, val_acc = self.validate(val_data)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, f"best_{self.model_type}.pth")
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train breakthrough models for ARC Prize 2025")
    parser.add_argument("--model", type=str, default="human_reasoning",
                       choices=["human_reasoning", "breakthrough"],
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing dataset")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Load checkpoint file")
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = ARCDataset(args.data_dir)
    
    # Initialize trainer
    trainer = BreakthroughTrainer(args.model, args.device)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train model
    history = trainer.train(dataset, epochs=args.epochs)
    
    # Save training history
    with open(f"training_history_{args.model}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved: training_history_{args.model}.json")


if __name__ == "__main__":
    main() 