# PyTorch CPU Trainer

A high-performance PyTorch training framework optimized for CPU environments with automated hyperparameter tuning and CPU-specific optimizations.

## Features
- CPU-optimized training with Intel Extensions for PyTorch (IPEX)
- Automated hyperparameter tuning with Optuna
- Built-in performance features:
  - Stochastic Weight Averaging (SWA)
  - Learning rate warmup and scheduling
  - Label smoothing
  - Gradient accumulation
  - Early stopping and pruning
  - Mixed precision with BFloat16 (when available)
- MLP architecture with residual connections
- SQLite-based experiment tracking

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Clone repository
git clone https://github.com/yourusername/pytorch_cpu_trainer.git
cd pytorch_cpu_trainer
```

## Quick Start

1. Prepare your data:
```bash
# Place your CSV files in the input directory
mkdir -p pytorch_cpu_trainer/input_data
cp train.csv pytorch_cpu_trainer/input_data/
cp val.csv pytorch_cpu_trainer/input_data/
```

2. Run training:
```bash
# First run - uses original defaults
python run.py --mode train --config config.yaml

# Second run - automatically uses averaged best parameters
python run.py --mode train --config config.yaml

# Force retrain with new parameters
python run.py --mode train --config config.yaml --retrain
```

3. Run inference:
```bash
python run.py --mode inference --config config.yaml
```

## Project Structure
```
pytorch_cpu_trainer/
├── run.py                 # Main entry point
├── config.yaml           # Configuration file
├── pytorch_cpu_trainer/  # Core package
│   ├── models.py        # Model architectures
│   ├── optimizers.py    # CPU optimizations
│   ├── trainers/       # Training logic
│   └── tuners/         # Hyperparameter tuning
├── input_data/          # Data directory
│   ├── train.csv
│   └── val.csv
├── logs/               # Logging output
└── checkpoints/        # Model checkpoints
```

## Configuration

Key sections in `config.yaml`:
```yaml
data:
  train_path: pytorch_cpu_trainer/input_data/train.csv
  val_path: pytorch_cpu_trainer/input_data/val.csv

training:
  batch_size: 128
  epochs: 10
  optimizer_choice: Adam
  device: cpu
  
cpu_optimization:
  enable_mkldnn: true
  num_threads: auto
  use_bfloat16: true
```

## Experiment Tracking

View training history:
```bash
sqlite3 pytorch_cpu_trainer/checkpoints/model_history.db

# Show all experiments
SELECT 
    datetime(timestamp) as date,
    metric_value as score,
    json_extract(architecture, '$.hidden_layers') as layers
FROM model_experiments 
ORDER BY metric_value DESC;
```

## Contact

For questions and feedback, please open an issue in the repository.

## License

MIT License