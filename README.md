# ARC Prize 2025 - AI Novel Reasoning Challenge

## Overview

This project aims to develop an AI system capable of novel reasoning for the [ARC Prize 2025](https://kaggle.com/competitions/arc-prize-2025) competition. The goal is to create an AI that can efficiently learn new skills and solve open-ended problems without relying exclusively on extensive pre-training datasets.

## Competition Details

- **Prize Pool**: $1,000,000 total ($125,000 progress prizes + $700,000 grand prize + $175,000 additional)
- **Grand Prize**: Unlocked if any team achieves ≥85% accuracy
- **Deadline**: November 3, 2025 (Final submission)
- **Current Best AI**: ~4% accuracy vs Human: 100% accuracy

## Project Structure

```
ARC/
├── data/                   # Competition data and datasets
├── src/                    # Source code
│   ├── models/            # AI model implementations
│   ├── preprocessing/     # Data preprocessing utilities
│   ├── evaluation/        # Evaluation and scoring functions
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for experimentation
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── submission.json        # Competition submission file
```

## Key Features

- **Novel Reasoning**: Focus on generalization beyond training data
- **Efficient Learning**: Rapid skill acquisition from a few examples
- **Open-Ended Problem Solving**: Handle previously unseen problem types
- **Human-Level Performance**: Target 85%+ accuracy to unlock grand prize

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Competition Data**:
   - Download the ARC-AGI-2 dataset from [Kaggle](https://kaggle.com/competitions/arc-prize-2025/data)
   - Place the following files in the `data/` directory:
     - `arc-agi_training-challenges.json`
     - `arc-agi_training-solutions.json`
     - `arc-agi_evaluation-challenges.json`
     - `arc-agi_evaluation-solutions.json`
     - `arc-agi_test-challenges.json`
     - `sample_submission.json`

3. **Run Experiments**:
   ```bash
   python src/main.py --config configs/experiment.yaml
   ```

4. **Generate Submission**:
   ```bash
   python src/generate_submission.py
   ```

## Available Models

### Basic Models
- **`SimpleBaselineModel`**: Basic baseline implementation for testing
- **`PatternMatchingModel`**: Pattern recognition approach for common transformations

### Advanced Models
- **`SymbolicReasoningModel`**: Symbolic reasoning using logical transformations (inspired by ILP approaches)
- **`EnsembleModel`**: Combines multiple approaches for improved robustness
- **`FewShotLearningModel`**: Few-shot learning inspired by self-supervised approaches (BYOL, DINO, SimCLR)

### Usage
```bash
# Test different models
python src/main.py --evaluate --model baseline
python src/main.py --evaluate --model pattern_matching
python src/main.py --evaluate --model symbolic
python src/main.py --evaluate --model ensemble
python src/main.py --evaluate --model few_shot

# Comprehensive model testing
python test_models.py

# Submission management (respects 1 per day limit)
python submission_manager.py --generate ensemble --output submission.json
python submission_manager.py --status
python submission_manager.py --leaderboard
python submission_manager.py --validate submission.json
```

## Approach Strategy

### Phase 1: Understanding ARC Problems
- Analyze problem patterns and abstractions
- Identify common transformation types
- Study human problem-solving strategies

### Phase 2: Model Development
- Implement novel reasoning architectures
- Focus on few-shot learning capabilities
- Develop pattern recognition systems

### Phase 3: Optimization
- Fine-tune for accuracy and efficiency
- Optimize for competition constraints
- Ensure reproducibility and robustness

## Evaluation Metrics

- **Primary**: Percentage of correct predictions
- **Format**: 2 attempts per task output
- **Scoring**: Highest score per task output averaged across all outputs

## Submission Format

```json
{
  "task_id": [
    {
      "attempt_1": [[0, 0], [0, 0]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ]
}
```

## Competition Rules & Strategy

### Key Rules
- **Daily Limit**: 1 submission per day
- **Team Size**: Maximum 5 members
- **Open Source**: Winners must open source their solutions
- **Timeline**: Entry deadline October 27, 2025; Final submission November 3, 2025

### Strategy Documents
- `competition_strategy.md` - Overall competition strategy
- `competition_rules_strategy.md` - Rules-based optimization
- `submission_manager.py` - Daily submission management

### Quick Start
```bash
# Get everything set up
python quick_start.py

# Check submission status
python submission_manager.py --status

# Generate today's submission
python submission_manager.py --generate ensemble
```

## Resources

- [Competition Page](https://kaggle.com/competitions/arc-prize-2025)
- [ARC Prize Website](https://arcprize.org)
- [ARC-AGI-2 Dataset](https://github.com/fchollet/ARC)
- [Research Paper](https://arxiv.org/abs/1911.01547)
- [Interactive App](https://arcprize.org) - Highly recommended for understanding the competition

## Team

[Your team information here]

## License

This project is open source as required by the competition rules. 