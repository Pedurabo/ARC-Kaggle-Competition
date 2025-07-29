# Getting Started with ARC Prize 2025

This guide will help you get up and running with the ARC Prize 2025 project quickly and efficiently.

## 🎯 Project Overview

The ARC Prize 2025 competition challenges participants to create AI systems capable of **novel reasoning** - solving problems they've never seen before. Unlike traditional AI that relies on extensive training data, this competition focuses on **few-shot learning** and **generalization**.

**Key Challenge**: Current AI systems score ~4% on ARC, while humans achieve 100%. Your goal is to reach 85% to unlock the $700,000 grand prize!

## 📁 Updated Project Structure

```
ARC/
├── 📄 README.md                    # Project documentation
├── 📄 QUICKSTART.md               # Quick start guide
├── 📄 GETTING_STARTED.md          # This file
├── 📄 download_dataset.py         # Dataset setup script
├── 📄 requirements.txt            # Python dependencies
├── 📄 setup.py                    # Installation script
├── 📄 .gitignore                  # Git ignore rules
├── 📁 src/                        # Source code
│   ├── 📁 models/                 # AI model implementations
│   │   ├── __init__.py
│   │   └── base_model.py          # Base model class + baseline models
│   ├── 📁 utils/                  # Data loading utilities
│   │   ├── __init__.py
│   │   └── data_loader.py         # Updated for ARC-AGI-2 format
│   ├── 📁 evaluation/             # Scoring and evaluation
│   │   ├── __init__.py
│   │   └── scorer.py              # Competition scoring logic
│   └── 📄 main.py                 # Main pipeline script
├── 📁 notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb  # Data exploration notebook
├── 📁 configs/                    # Configuration files
│   └── default.yaml               # Updated configuration
├── 📁 tests/                      # Unit tests
│   ├── __init__.py
│   └── test_data_loader.py        # Data loader tests
└── 📁 data/                       # Dataset files (not in repo)
    ├── arc-agi_training-challenges.json
    ├── arc-agi_training-solutions.json
    ├── arc-agi_evaluation-challenges.json
    ├── arc-agi_evaluation-solutions.json
    ├── arc-agi_test-challenges.json
    └── sample_submission.json
```

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Dataset

**Option A: Download Real Dataset**
```bash
# Check current dataset status
python download_dataset.py --check

# Follow instructions to download from Kaggle
# https://kaggle.com/competitions/arc-prize-2025/data
```

**Option B: Use Sample Data (for testing)**
```bash
# Create sample data files
python download_dataset.py --create-sample
```

### 3. Test the Pipeline

```bash
# Test with sample data
python src/main.py --evaluate --model baseline

# Test pattern matching model
python src/main.py --evaluate --model pattern_matching
```

## 📊 Understanding the Dataset

The ARC-AGI-2 dataset has a **new format** different from previous competitions:

### File Structure
- **`arc-agi_training-challenges.json`**: Training tasks with input/output pairs
- **`arc-agi_training-solutions.json`**: Solutions for training tasks
- **`arc-agi_evaluation-challenges.json`**: Evaluation tasks
- **`arc-agi_evaluation-solutions.json`**: Solutions for evaluation tasks
- **`arc-agi_test-challenges.json`**: Test tasks (for competition scoring)
- **`sample_submission.json`**: Example submission format

### Task Format
Each task contains:
```json
{
  "task_id": {
    "train": [
      {
        "input": [[0, 1], [1, 0]],  // Input grid
        "output": [[1, 0], [0, 1]]  // Expected output
      }
    ],
    "test": [
      {
        "input": [[1, 0], [0, 1]]   // Test input (no output provided)
      }
    ]
  }
}
```

### Submission Format
```json
{
  "task_id": [
    {
      "attempt_1": [[0, 1], [1, 0]],
      "attempt_2": [[1, 0], [0, 1]]
    }
  ]
}
```

## 🔧 Key Components

### 1. Data Loading (`src/utils/data_loader.py`)
- **`ARCDataset`**: Main dataset loader for the new format
- **`load_submission_template()`**: Create submission templates
- **`validate_submission()`**: Validate submission format
- **`save_submission()`**: Save submissions to JSON

### 2. Models (`src/models/base_model.py`)
- **`BaseARCModel`**: Abstract base class for all models
- **`SimpleBaselineModel`**: Basic baseline implementation
- **`PatternMatchingModel`**: Pattern recognition approach

### 3. Evaluation (`src/evaluation/scorer.py`)
- **`ARCScorer`**: Official competition scoring
- **`CrossValidationScorer`**: Training data evaluation

### 4. Main Pipeline (`src/main.py`)
- Complete pipeline from data loading to submission generation
- Support for evaluation and test data
- Automatic fallback to baseline models

## 🧪 Running Experiments

### Basic Evaluation
```bash
# Evaluate baseline model on training data
python src/main.py --evaluate --model baseline

# Evaluate pattern matching model
python src/main.py --evaluate --model pattern_matching

# Evaluate specific tasks
python src/main.py --evaluate --model baseline --task_ids task1 task2
```

### Generate Submissions
```bash
# Generate evaluation submission
python src/main.py --model pattern_matching --output eval_submission.json

# Generate test submission (for competition)
python src/main.py --model pattern_matching --output submission.json
```

### Explore Data
```bash
# Start Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🎯 Competition Strategy

### Understanding ARC Problems
1. **Study the Interactive App**: Visit [arcprize.org](https://arcprize.org) to understand the challenge
2. **Analyze Patterns**: Use the data exploration notebook to understand common transformations
3. **Focus on Generalization**: The key is solving novel problems, not memorizing patterns

### Model Development Approaches
1. **Pattern Recognition**: Identify common transformations (rotation, flip, color mapping)
2. **Symbolic Reasoning**: Use rule-based systems for logical transformations
3. **Neural Networks**: Train models on pattern recognition
4. **Ensemble Methods**: Combine multiple approaches
5. **Few-Shot Learning**: Focus on learning from few examples

### Key Insights from Dataset
- Grid sizes: 1x1 to 30x30
- Colors: 0-9 (10 different colors)
- Most tasks have 2-4 training pairs
- Some tasks have multiple test inputs
- Exact matching required (no partial credit)

## 🏆 Competition Timeline

- **Entry Deadline**: October 27, 2025
- **Team Merger Deadline**: October 27, 2025
- **Final Submission**: November 3, 2025
- **Paper Award**: November 9, 2025

## 💡 Development Tips

### 1. Start Simple
```bash
# Begin with baseline model
python src/main.py --evaluate --model baseline
```

### 2. Iterate Quickly
- Use sample data for rapid testing
- Focus on understanding patterns first
- Implement simple transformations before complex ones

### 3. Validate Constantly
```bash
# Check submission format
python -c "
from src.utils.data_loader import validate_submission
import json
with open('submission.json', 'r') as f:
    submission = json.load(f)
print('Valid:', validate_submission(submission))
"
```

### 4. Monitor Performance
- Track performance on training data
- Use cross-validation to avoid overfitting
- Focus on generalization, not memorization

## 🔍 Common Issues & Solutions

### "No training challenges found"
- Ensure dataset files are in `data/` directory
- Check file names match exactly
- Use `python download_dataset.py --check` to verify

### Import errors
- Make sure you're in the project root
- Verify virtual environment is activated
- Check all dependencies are installed

### Memory issues
- Process tasks in smaller batches
- Use smaller model architectures
- Reduce grid size limits

## 📚 Resources

- **[Competition Page](https://kaggle.com/competitions/arc-prize-2025)**: Official competition
- **[ARC Prize Website](https://arcprize.org)**: Interactive app and resources
- **[Dataset](https://github.com/fchollet/ARC)**: Original ARC dataset
- **[Research Paper](https://arxiv.org/abs/1911.01547)**: Original ARC paper
- **[Community Discussions](https://kaggle.com/competitions/arc-prize-2025/discussion)**: Kaggle forums

## 🎉 Next Steps

1. **Download the dataset** and explore it
2. **Run the baseline models** to understand the challenge
3. **Study the interactive app** at arcprize.org
4. **Implement your own models** by extending `BaseARCModel`
5. **Experiment with different approaches** and iterate quickly
6. **Submit to the competition** and track your progress

Good luck with your ARC Prize 2025 journey! 🚀

Remember: The goal is **novel reasoning**, not memorization. Focus on building systems that can generalize to unseen problems! 