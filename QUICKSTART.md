# Quick Start Guide - ARC Prize 2025

This guide will help you get started with the ARC Prize 2025 project in minutes.

## Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of Python and machine learning

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd ARC
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### 1. Download the Dataset

The ARC-AGI-2 dataset is required for this project. You can download it from:
- [Kaggle Competition Page](https://kaggle.com/competitions/arc-prize-2025/data)

**Important**: The dataset contains 6 files:
- `arc-agi_training-challenges.json` - Training tasks with input/output pairs
- `arc-agi_training-solutions.json` - Solutions for training tasks
- `arc-agi_evaluation-challenges.json` - Evaluation tasks
- `arc-agi_evaluation-solutions.json` - Solutions for evaluation tasks  
- `arc-agi_test-challenges.json` - Test tasks (for competition scoring)
- `sample_submission.json` - Example submission format

Place the downloaded data in the `data/` directory with this structure:
```
data/
├── arc-agi_training-challenges.json
├── arc-agi_training-solutions.json
├── arc-agi_evaluation-challenges.json
├── arc-agi_evaluation-solutions.json
├── arc-agi_test-challenges.json
└── sample_submission.json
```

### 2. Run Your First Experiment

```bash
# Test the baseline model on training data
python src/main.py --evaluate --model baseline

# Test the pattern matching model
python src/main.py --evaluate --model pattern_matching

# Generate a submission (if you have evaluation data)
python src/main.py --model pattern_matching --output my_submission.json
```

### 3. Explore the Data

Open the Jupyter notebook to explore the dataset:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_loader.py
```

## Project Structure

```
ARC/
├── data/                   # Dataset files (not included in repo)
├── src/                    # Source code
│   ├── models/            # AI model implementations
│   ├── utils/             # Data loading and utilities
│   ├── evaluation/        # Scoring and evaluation
│   └── main.py           # Main pipeline script
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Key Components

### Models (`src/models/`)
- `BaseARCModel`: Abstract base class for all models
- `SimpleBaselineModel`: Basic baseline implementation
- `PatternMatchingModel`: Pattern recognition approach

### Data Loading (`src/utils/`)
- `ARCDataset`: Main dataset loader
- Submission utilities for competition format

### Evaluation (`src/evaluation/`)
- `ARCScorer`: Official competition scoring
- `CrossValidationScorer`: Training data evaluation

## Competition Submission

To generate a competition submission:

1. **Ensure you have evaluation data** in `data/evaluation/evaluation.json`

2. **Run the pipeline**:
   ```bash
   python src/main.py --model your_model --output submission.json
   ```

3. **Validate the submission**:
   ```bash
   python -c "
   from src.utils.data_loader import validate_submission
   import json
   with open('submission.json', 'r') as f:
       submission = json.load(f)
   print('Valid:', validate_submission(submission))
   "
   ```

4. **Submit to Kaggle**:
   - Upload `submission.json` to the competition
   - Or use the Kaggle API

## Development Workflow

1. **Create a new model**:
   - Inherit from `BaseARCModel`
   - Implement the `solve_task` method
   - Add tests in `tests/`

2. **Experiment with notebooks**:
   - Use `notebooks/01_data_exploration.ipynb` as a starting point
   - Create new notebooks for your experiments

3. **Test your changes**:
   ```bash
   python -m pytest tests/ -v
   ```

4. **Evaluate performance**:
   ```bash
   python src/main.py --evaluate --model your_model
   ```

## Common Issues

### "No training data found"
- Ensure the dataset is downloaded and placed in `data/`
- Check the file structure matches the expected format

### Import errors
- Make sure you're in the project root directory
- Verify the virtual environment is activated
- Check that all dependencies are installed

### Memory issues
- Reduce batch sizes in configuration
- Use smaller model architectures
- Process tasks in smaller batches

## Next Steps

1. **Study the ARC problems**: Understand the patterns and transformations
2. **Implement better models**: Focus on novel reasoning approaches
3. **Experiment with different architectures**: Try neural networks, symbolic reasoning, etc.
4. **Optimize for performance**: Balance accuracy with computational efficiency
5. **Collaborate**: Join the community discussions and share insights

## Resources

- [ARC Prize 2025 Competition](https://kaggle.com/competitions/arc-prize-2025)
- [ARC Prize Website](https://arcprize.org)
- [ARC-AGI-2 Dataset](https://github.com/fchollet/ARC)
- [Research Paper](https://arxiv.org/abs/1911.01547)

## Support

- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs or questions
- Join the competition discussions on Kaggle

Good luck with your ARC Prize 2025 journey! 🚀 