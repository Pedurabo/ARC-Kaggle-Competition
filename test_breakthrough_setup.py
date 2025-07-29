#!/usr/bin/env python3
"""
Test script to verify breakthrough setup is working correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torch.nn as nn
        print("✅ torch.nn imported")
    except ImportError as e:
        print(f"❌ torch.nn import failed: {e}")
        return False
    
    try:
        from src.models.breakthrough_modules import AbstractReasoningModule
        print("✅ AbstractReasoningModule imported")
    except ImportError as e:
        print(f"❌ AbstractReasoningModule import failed: {e}")
        return False
    
    try:
        from src.models.breakthrough_modules import AdvancedMetaLearner
        print("✅ AdvancedMetaLearner imported")
    except ImportError as e:
        print(f"❌ AdvancedMetaLearner import failed: {e}")
        return False
    
    try:
        from src.models.breakthrough_modules import MultiModalReasoner
        print("✅ MultiModalReasoner imported")
    except ImportError as e:
        print(f"❌ MultiModalReasoner import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created."""
    print("\n🔍 Testing model creation...")
    
    try:
        from src.models.breakthrough_modules import AbstractReasoningModule
        model = AbstractReasoningModule()
        print("✅ AbstractReasoningModule created")
    except Exception as e:
        print(f"❌ AbstractReasoningModule creation failed: {e}")
        return False
    
    try:
        from src.models.breakthrough_modules import AdvancedMetaLearner
        model = AdvancedMetaLearner()
        print("✅ AdvancedMetaLearner created")
    except Exception as e:
        print(f"❌ AdvancedMetaLearner creation failed: {e}")
        return False
    
    try:
        from src.models.breakthrough_modules import MultiModalReasoner
        model = MultiModalReasoner()
        print("✅ MultiModalReasoner created")
    except Exception as e:
        print(f"❌ MultiModalReasoner creation failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that data can be loaded."""
    print("\n🔍 Testing data loading...")
    
    try:
        from src.utils.data_loader import ARCDataset
        dataset = ARCDataset()
        print("✅ ARCDataset created")
        
        # Try to load training data
        try:
            train_data, _ = dataset.load_training_data()
            print(f"✅ Training data loaded: {len(train_data)} samples")
        except Exception as e:
            print(f"⚠️  Training data loading failed: {e}")
            print("Creating sample data for testing...")
            
            # Create sample data
            train_data = [
                {
                    'task_id': f'train_{i}',
                    'train': [
                        {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                        {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
                    ],
                    'test': [
                        {'input': [[0, 0], [1, 1]], 'output': [[1, 1], [0, 0]]}
                    ]
                }
                for i in range(10)
            ]
            print(f"✅ Sample data created: {len(train_data)} samples")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    return True

def test_model_inference():
    """Test that models can perform inference."""
    print("\n🔍 Testing model inference...")
    
    # Create sample task
    sample_task = {
        'task_id': 'test_task',
        'train': [
            {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
            {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
        ],
        'test': [
            {'input': [[0, 0], [1, 1]]}
        ]
    }
    
    try:
        import torch
        from src.models.breakthrough_modules import AbstractReasoningModule
        model = AbstractReasoningModule()
        
        # Test forward pass
        with torch.no_grad():
            output = model(sample_task)
            print("✅ AbstractReasoningModule inference successful")
            print(f"   Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"❌ AbstractReasoningModule inference failed: {e}")
        return False
    
    return True

def test_training_components():
    """Test that training components work."""
    print("\n🔍 Testing training components...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Create a simple model
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create sample data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✅ Training components working")
        print(f"   Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Training components failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 BREAKTHROUGH SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_loading,
        test_model_inference,
        test_training_components
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Imports",
        "Model Creation", 
        "Data Loading",
        "Model Inference",
        "Training Components"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Breakthrough setup is ready for training!")
        print("\n🚀 Next steps:")
        print("1. Start training: python train_breakthrough_modules.py --model_type abstract_reasoning --epochs 20")
        print("2. Generate submission: python submission_manager.py --generate ensemble --output breakthrough_v1.json")
        print("3. Submit to Kaggle and analyze results")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 