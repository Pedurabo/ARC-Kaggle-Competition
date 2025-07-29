#!/usr/bin/env python3
"""
Quick start script for ARC Prize 2025 competition.
Immediately test models and generate competitive submission.
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    """Main quick start function."""
    print("🚀 ARC PRIZE 2025 - QUICK START")
    print("=" * 50)
    
    # Step 1: Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (where the 'src' folder is located)")
        return
    
    # Step 2: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    run_command("pip install -r requirements.txt", "Installing Python dependencies")
    
    # Step 3: Set up dataset
    print("\n📊 Step 2: Setting up dataset...")
    if not Path("data").exists():
        Path("data").mkdir()
    
    # Check if dataset files exist
    dataset_files = [
        "data/arc-agi_training-challenges.json",
        "data/arc-agi_training-solutions.json",
        "data/arc-agi_evaluation-challenges.json",
        "data/arc-agi_evaluation-solutions.json",
        "data/arc-agi_test-challenges.json",
        "data/sample_submission.json"
    ]
    
    missing_files = [f for f in dataset_files if not Path(f).exists()]
    
    if missing_files:
        print("📥 Creating sample dataset for testing...")
        run_command("python download_dataset.py --create-sample", "Creating sample dataset")
    else:
        print("✅ Dataset files already exist")
    
    # Step 4: Test all models
    print("\n🧪 Step 3: Testing all models...")
    test_output = run_command("python test_models.py", "Testing all models")
    
    if test_output:
        print("\n📊 Model Test Results:")
        print("-" * 30)
        # Extract key results from test output
        lines = test_output.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Score:', 'Best performing model:', 'Results saved']):
                print(line.strip())
    
    # Step 5: Generate competitive submission
    print("\n🏆 Step 4: Generating competitive submission...")
    
    # Try ensemble model first (most likely to be competitive)
    submission_output = run_command(
        "python src/main.py --model ensemble --output submission_ensemble.json",
        "Generating ensemble submission"
    )
    
    if submission_output:
        print("✅ Ensemble submission generated: submission_ensemble.json")
    
    # Also try symbolic model
    symbolic_output = run_command(
        "python src/main.py --model symbolic --output submission_symbolic.json",
        "Generating symbolic submission"
    )
    
    if symbolic_output:
        print("✅ Symbolic submission generated: submission_symbolic.json")
    
    # Step 6: Quick evaluation
    print("\n📈 Step 5: Quick evaluation...")
    eval_output = run_command(
        "python src/main.py --evaluate --model ensemble",
        "Evaluating ensemble model"
    )
    
    # Step 7: Summary and next steps
    print("\n" + "=" * 50)
    print("🎉 QUICK START COMPLETED!")
    print("=" * 50)
    
    print("\n📁 Generated Files:")
    submission_files = list(Path(".").glob("submission_*.json"))
    for file in submission_files:
        print(f"   ✅ {file.name}")
    
    print("\n🚀 Next Steps:")
    print("   1. 📊 Check model_test_results.json for detailed performance")
    print("   2. 🏆 Submit submission_ensemble.json to Kaggle")
    print("   3. 📈 Monitor your leaderboard position")
    print("   4. 🔄 Iterate and improve based on results")
    
    print("\n📚 Resources:")
    print("   - Competition: https://kaggle.com/competitions/arc-prize-2025")
    print("   - Interactive App: https://arcprize.org")
    print("   - Strategy Guide: competition_strategy.md")
    
    print("\n🎯 Current Leaderboard Context:")
    print("   - Top Score: 19.58% (Giotto.ai)")
    print("   - Baseline: 4.17% (many teams)")
    print("   - Target: 85% (grand prize)")
    print("   - Your Goal: Beat 4.17% baseline first!")
    
    print("\n💡 Tips:")
    print("   - Start with ensemble model (most robust)")
    print("   - Use symbolic model for logical tasks")
    print("   - Test frequently with python test_models.py")
    print("   - Focus on novel reasoning, not memorization")
    
    print("\n🏁 Ready to compete! Good luck! 🚀")

if __name__ == "__main__":
    main() 