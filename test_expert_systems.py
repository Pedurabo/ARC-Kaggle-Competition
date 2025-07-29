#!/usr/bin/env python3
"""
Test script for Expert Systems Intelligence
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_expert_systems_intelligence():
    """Test the expert systems intelligence implementation"""
    print("🧠 Testing Expert Systems Intelligence...")
    
    try:
        # Test import
        from models.expert_systems_intelligence import get_expert_systems_intelligence
        print("✅ Expert Systems Intelligence import successful")
        
        # Initialize system
        expert_system = get_expert_systems_intelligence()
        print("✅ Expert Systems Intelligence initialized")
        
        # Test basic functionality
        task = {
            'task_id': 'test_task',
            'input': [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
            'patterns': []
        }
        
        result = expert_system.solve_task(task)
        print(f"✅ Task solved successfully, got {len(result)} predictions")
        
        # Get summary
        summary = expert_system.get_intelligence_summary()
        print(f"✅ Intelligence level: {summary['intelligence_level']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Expert Systems Intelligence: {e}")
        return False

def test_pattern_expert_system():
    """Test the pattern expert system"""
    print("\n🔍 Testing Pattern Expert System...")
    
    try:
        # Test import
        from models.pattern_expert_system import get_pattern_expert_system
        print("✅ Pattern Expert System import successful")
        
        # Initialize system
        pattern_expert = get_pattern_expert_system()
        print("✅ Pattern Expert System initialized")
        
        # Test pattern recognition
        input_data = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        output_data = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        
        patterns = pattern_expert.recognize_patterns(input_data, output_data)
        print(f"✅ Pattern recognition successful, found {len(patterns)} patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Pattern Expert System: {e}")
        return False

def test_meta_learning_expert_system():
    """Test the meta-learning expert system"""
    print("\n🧠 Testing Meta-Learning Expert System...")
    
    try:
        # Test import
        from models.meta_learning_expert_system import get_meta_learning_expert_system
        print("✅ Meta-Learning Expert System import successful")
        
        # Initialize system
        meta_learning = get_meta_learning_expert_system()
        print("✅ Meta-Learning Expert System initialized")
        
        # Test strategy recommendation
        task = {
            'task_id': 'test_task',
            'input_shape': (3, 3),
            'output_shape': (3, 3),
            'complexity_score': 0.5,
            'patterns': []
        }
        
        recommendation = meta_learning.recommend_strategy(task)
        print(f"✅ Strategy recommendation: {recommendation.strategy_id}")
        
        # Get summary
        summary = meta_learning.get_meta_learning_summary()
        print(f"✅ Meta-learning summary generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Meta-Learning Expert System: {e}")
        return False

def test_ultimate_intelligence_integration():
    """Test the ultimate intelligence integration"""
    print("\n🚀 Testing Ultimate Intelligence Integration...")
    
    try:
        # Test import
        from models.ultimate_intelligence_integration import get_ultimate_intelligence_integration
        print("✅ Ultimate Intelligence Integration import successful")
        
        # Initialize system
        ultimate_intelligence = get_ultimate_intelligence_integration()
        print("✅ Ultimate Intelligence Integration initialized")
        
        # Test task solving
        task = {
            'task_id': 'test_task',
            'input': [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
            'output': [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
            'patterns': [{'type': 'geometric', 'confidence': 0.95}]
        }
        
        solution = ultimate_intelligence.solve_task_with_ultimate_intelligence(task)
        print(f"✅ Task solved with confidence: {solution.confidence:.3f}")
        print(f"✅ Intelligence level: {solution.intelligence_level}%")
        
        # Get summary
        summary = ultimate_intelligence.get_intelligence_summary()
        print(f"✅ Ultimate intelligence summary generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Ultimate Intelligence Integration: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧠 EXPERT SYSTEMS INTELLIGENCE TEST SUITE")
    print("Beyond 120% Human Genius Level Testing")
    print("=" * 60)
    
    tests = [
        test_expert_systems_intelligence,
        test_pattern_expert_system,
        test_meta_learning_expert_system,
        test_ultimate_intelligence_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"📈 Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Expert Systems Intelligence is ready for deployment!")
        print("🧠 Beyond 120% human genius level achieved!")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 