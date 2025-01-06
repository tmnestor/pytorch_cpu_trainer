# PyTorch CPU Trainer

A high-performance PyTorch training framework optimized for CPU environments, featuring automated hyperparameter tuning, CPU-specific optimizations, and robust error handling.

## Features

- CPU-optimized training with Intel Extensions for PyTorch (IPEX)
- Automated hyperparameter tuning using Optuna
- Stochastic Weight Averaging (SWA) support
- Learning rate warmup and scheduling
- Label smoothing
- Gradient accumulation
- Early stopping and pruning
- Comprehensive logging system
- Residual connections in MLP architecture

## Requirements

```
torch
intel_extension_for_pytorch
optuna
pandas
numpy
pyyaml
tqdm
scikit-learn
psutil
py-cpuinfo
seaborn
matplotlib
```

## Project Structure

```
pytorch_cpu_trainer/
├── pytorch_cpu_trainer.py  # Main training script
├── config.yaml            # Configuration file
├── input_data/           # Data directory
│   ├── train.csv
│   └── val.csv
├── logs/                 # Logging directory
└── checkpoints/          # Model checkpoints
```

## Configuration

The `config.yaml` file contains all configurable parameters:

- Model architecture
- Training parameters
- CPU optimization settings
- Logging configuration
- Data paths
- Hyperparameter tuning settings

## Usage

1. Prepare your data in CSV format with features and a target column
2. Update the config.yaml file with your settings
3. Run the trainer:

```bash
python pytorch_cpu_trainer.py
```

## CPU Optimizations

The trainer includes several CPU-specific optimizations:

- Automatic thread configuration
- MKL-DNN/oneDNN support
- BFloat16 mixed precision (when supported)
- IPEX optimizations
- Memory format optimization
- JIT compilation

## Hyperparameter Tuning

The framework uses Optuna for automated hyperparameter optimization:

- Network architecture (number and size of layers)
- Learning rate
- Dropout rate
- Batch normalization
- Weight decay

## Monitoring

Training progress is monitored through:

- Comprehensive logging
- Learning curves visualization
- F1-score and accuracy metrics
- CPU resource utilization

## Model Architecture

The MLP classifier includes:

- Configurable hidden layers
- Residual connections
- Batch normalization
- Dropout regularization
- GELU activation

## Error Handling

Robust error handling for:

- Data validation
- Model initialization
- Training process
- Hyperparameter tuning
- Resource management

## Contributing

Feel free to submit issues and pull requests.

## License

MIT License

## Contact

For questions and feedback, please open an issue in the repository.
