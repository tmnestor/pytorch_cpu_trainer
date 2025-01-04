# Advanced CPU Optimizations for Pytorch MLP Classification Training

A PyTorch-based multi-class classifier implementation with advanced CPU optimizations, adaptive learning rate strategies, and comprehensive validation.

## Key Features

### 1. Memory Management
- Memory-mapped dataset handling for large datasets (>1GB)
- Automatic memory threshold detection
- Efficient cleanup with proper resource management
- Chunked data loading with synchronization

### 2. CPU Optimizations
- MKL-DNN acceleration where available
- MKL optimizations for linear algebra operations
- Automatic thread count optimization
- Mixed precision (bfloat16) support for compatible CPUs
- Dynamic batch size adjustment

### 3. Advanced Learning Rate Management
- Three-phase learning rate strategy:
  - Warmup phase with linear scaling
  - Cycling phase for exploration
  - Plateau detection for fine-tuning
- Adaptive optimization switching based on performance
- Automatic strategy adjustment based on metric improvement

### 4. Hyperparameter Tuning
- Optuna-based hyperparameter optimization
- Efficient trial pruning for poor performers
- Median pruning for below-average trials
- Automatic checkpoint management
- Configuration persistence

### 5. Validation and Monitoring
- Comprehensive model validation
- Confidence analysis for predictions
- Cross-validation with k-folds
- Detailed performance metrics
- Confusion matrix visualization

### 6. Logging and Debugging
- Hierarchical logging system
- Separate console and file handlers
- Performance monitoring
- CPU optimization status tracking
- Training progress visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your model in `config.yaml`
2. Prepare your data in CSV format
3. Run the trainer:

```bash
python MLP_trainer.py
```

## Configuration

The system uses a YAML configuration file with the following key sections:
- Data configuration
- Model architecture
- Optimization settings
- Training parameters
- CPU optimization flags
- Logging preferences

## Performance Features

1. **Data Loading**
   - Persistent workers
   - Prefetch queue
   - Memory mapping for large datasets
   - Automatic worker count optimization

2. **CPU Utilization**
   - Thread count optimization
   - MKL/MKL-DNN acceleration
   - Mixed precision support
   - Memory-efficient operations

3. **Training Optimization**
   - Adaptive batch sizing
   - Gradient accumulation
   - Learning rate adaptation
   - Early stopping with patience

## Validation Metrics

The model provides comprehensive validation metrics:

- Overall accuracy and F1-score
- Per-class precision, recall, and F1-score
- Confidence analysis for correct/incorrect predictions
- Visual confusion matrix
- Cross-validation results with standard deviation

## Performance Monitoring

- CPU optimization status logging
- Training progress visualization
- Resource utilization tracking
- Detailed logging with configurable verbosity

## Examples

See `examples/` directory for usage examples and notebook demonstrations.

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{mlp_classifier,
  title={MLP Classifier with Enhanced CPU Performance},
  author={Tod M. Nestor},
  year={2024},
  url={https://github.com/tmnestor/multiclass_classifier}
}
