import gc

import random
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import f1_score


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import psutil
import cpuinfo
import intel_extension_for_pytorch as ipex
import multiprocessing


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
# Set up logging
def setup_logger(config, name='MLPTrainer'):
    """Set up logging with both file and console handlers."""
    # Create logging directory if it doesn't exist
    log_dir = config['logging']['directory']
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to catch all levels
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler - use component specific config if available
    if name.lower() in config['logging']['handlers']:
        handler_config = config['logging']['handlers'][name.lower()]
        log_path = os.path.join(log_dir, handler_config['filename'])
        fh = logging.FileHandler(log_path)
        fh.setLevel(getattr(logging, handler_config['level']))
    else:
        # Default file handler
        log_path = os.path.join(log_dir, f'{name}.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(getattr(logging, config['logging']['file_level']))
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, config['logging']['console_level']))
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class CustomDataset(Dataset):
    def __init__(self, df, target_column, batch_size=1000):
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
            
        self.df = df
        self.target_column = target_column
        self.batch_size = batch_size
        
        # Pre-process all data at init for better performance
        self.features = torch.FloatTensor(df.drop(target_column, axis=1).values)
        self.labels = torch.LongTensor(df[target_column].values)
        
        # Verify tensors are created successfully
        if self.features.nelement() == 0 or self.labels.nelement() == 0:
            raise ValueError("Failed to create data tensors")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes=3, dropout_rate=0.2, use_batch_norm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        
class PyTorchTrainer:
    """A generic PyTorch trainer class.
    
    Attributes:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU/GPU)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    
    def __init__(self, model, criterion, optimizer, device='cpu', verbose=False, scheduler=None, config=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.scheduler = scheduler
        self.gradient_clip_val = 1.0
        
        # Set default values if config is not provided
        if config is None:
            self.grad_accum_steps = 1
            self.batch_multiplier = 1
        else:
            self.grad_accum_steps = config['training']['memory_management']['optimization']['grad_accumulation']['max_steps']
            self.batch_multiplier = config['training']['performance']['batch_size_multiplier']
        
        # Initialize mixed precision settings based on hardware support
        self.use_mixed_precision = hasattr(ipex, 'core') and ipex.core.onednn_has_bf16_support()
        if self.use_mixed_precision:
            self.scaler = torch.cpu.amp.GradScaler()
        
    def train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Remove channels_last format since we're using 2D data
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cpu.amp.autocast(enabled=self.use_mixed_precision, dtype=torch.bfloat16):
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y) / self.grad_accum_steps
            
            # Backward pass with gradient accumulation
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                accumulated_loss = 0
            
            # Update metrics
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Memory cleanup
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def evaluate(self, val_loader):
        """Evaluates the model on validation data."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        if len(val_loader) == 0:
            self.logger.warning("Empty validation loader!")
            return float('inf'), 0.0, 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Remove channels_last format here too
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        try:
            accuracy = 100 * correct / total if total > 0 else 0.0
            f1 = f1_score(all_labels, all_preds, average='weighted') if total > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            accuracy = 0.0
            f1 = 0.0
        
        return (total_loss / len(val_loader)) if len(val_loader) > 0 else float('inf'), accuracy, f1

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model for specified number of epochs. 
        Monitors specified validation metric for early stopping."""
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        best_val_metric = 0
        
        for epoch in tqdm(range(epochs), desc='Training'):
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # Select metric based on config
            train_metric = train_accuracy
            val_metric = val_f1 if metric == 'f1' else val_accuracy
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            best_val_metric = max(best_val_metric, val_metric)
            
            if self.verbose:
                metric_name = 'F1' if metric == 'f1' else 'Accuracy'
                metric_value = val_f1 if metric == 'f1' else val_accuracy
                print(f'Epoch {epoch+1}/{epochs}: Val {metric_name}: {metric_value:.2f}%')
            
            del outputs, loss
            gc.collect()  # Use only gc.collect() for CPU memory cleanup
        
        self.plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, 
                                metric_name='F1-Score' if metric == 'f1' else 'Accuracy')
        
        return train_losses, val_losses, train_metrics, val_metrics, best_val_metric
    
    @staticmethod
    def plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='Accuracy'):
        """Plots the learning curves for loss and chosen metric (accuracy or F1)."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Normalize values for better visualization
        max_loss = max(max(train_losses), max(val_losses))
        max_metric = max(max(train_metrics), max(val_metrics))
        
        epochs = range(1, len(train_losses) + 1)
        
        sns.lineplot(data={
            f"Training {metric_name}": [x/max_metric for x in train_metrics],
            f"Validation {metric_name}": [x/max_metric for x in val_metrics],
            "Training Loss": [x/max_loss for x in train_losses],
            "Validation Loss": [x/max_loss for x in val_losses]
        })
        
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Value")
        plt.title(f"Training and Validation Loss and {metric_name} Curves")
        plt.legend()
        plt.savefig('learning_curves.png')
        plt.close()

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.best_trial_value = float('-inf')
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_params = None
        self.logger = setup_logger(config, 'hyperparameter_tuning')
        os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
    
    def save_best_model(self, model, optimizer, trial_value, params):
        """Save the best model and its metadata."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_name': self.config['training']['optimizer_choice'],
            'metric_name': self.config['training']['optimization_metric'],
            'metric_value': trial_value,
            'hyperparameters': params
        }
        torch.save(checkpoint, self.config['model']['save_path'])
    
    def create_model_and_optimizer(self, trial):
        # Extract hyperparameters from trial
        hidden_layers = []
        n_layers = trial.suggest_int('n_layers', 1, 4)
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'hidden_layer_{i}', 32, 512))
        
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        weight_decay = 0.0 if use_batch_norm else trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        
        # Create model
        model = MLPClassifier(
            input_size=self.config['model']['input_size'],
            hidden_layers=hidden_layers,
            num_classes=self.config['model']['num_classes'],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Create optimizer
        optimizer = getattr(torch.optim, self.config['training']['optimizer_choice'])(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        trial_params = {
            'n_layers': n_layers,
            'hidden_layers': hidden_layers,
            'lr': lr,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'weight_decay': weight_decay
        }
        
        return model, optimizer, trial_params
    
    def objective(self, trial, train_loader, val_loader):
        """Optimization objective for hyperparameter tuning."""
        model, optimizer, trial_params = self.create_model_and_optimizer(trial)
        criterion = getattr(nn, self.config['training']['loss_function'])()
        
        # Create trainer with minimal components for tuning and pass config
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.config['training']['device'],
            config=self.config  # Pass config to trainer
        )
        
        best_metric = float('-inf')
        patience_counter = 0
        running_metrics = []
        
        try:
            for epoch in range(self.config['training']['epochs']):
                if len(train_loader) == 0 or len(val_loader) == 0:
                    self.logger.warning("Empty data loader detected!")
                    raise optuna.TrialPruned()
                
                # Training step
                trainer.train_epoch(train_loader)
                # Evaluation step
                val_loss, accuracy, f1 = trainer.evaluate(val_loader)
                
                if val_loss == float('inf') or (accuracy == 0.0 and f1 == 0.0):
                    self.logger.warning("Invalid metrics detected, pruning trial")
                    raise optuna.TrialPruned()
                
                # Get appropriate metric
                metric = f1 if self.config['training']['optimization_metric'] == 'f1' else accuracy
                trial.report(metric, epoch)
                
                # Update running metrics
                running_metrics.append(metric)
                if len(running_metrics) > 3:
                    running_metrics.pop(0)
                
                # Early stopping check
                if metric > best_metric + self.config['optimization']['early_stopping']['min_delta']:
                    best_metric = metric
                    patience_counter = 0
                    if metric > self.best_trial_value:
                        self.best_trial_value = metric
                        self.save_best_model(model, optimizer, metric, trial_params)
                else:
                    patience_counter += 1
                
                
                # Pruning check
                if epoch >= self.config['optimization']['pruning']['warm_up_epochs']:
                    avg_metric = sum(running_metrics) / len(running_metrics)
                    if (best_metric - avg_metric) / best_metric > self.config['optimization']['pruning']['deterioration_threshold']:
                        raise optuna.TrialPruned()
                
                if patience_counter >= self.config['optimization']['early_stopping']['patience']:
                    break
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
        except Exception as e:
            if not isinstance(e, optuna.TrialPruned):
                self.logger.error(f"Trial failed with error: {str(e)}")
            raise
            
        return best_metric
    
    def tune(self, train_loader, val_loader):
        self.logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader),
            n_trials=self.config['optimization']['n_trials']
        )
        
        return study.best_trial, study.best_params

def restore_best_model(config):
    """Utility function to restore the best model and its optimizer."""
    checkpoint_path = config['model']['save_path']
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Creating new model...")
        # Create a new model with default parameters
        model = MLPClassifier(
            input_size=config['model']['input_size'],
            hidden_layers=[256, 128, 64],  # Default architecture
            num_classes=config['model']['num_classes'],
            dropout_rate=0.2,
            use_batch_norm=True
        )
        
        optimizer = getattr(torch.optim, config['training']['optimizer_choice'])(
            model.parameters(),
            **config['training']['optimizer_params'][config['training']['optimizer_choice']]
        )
        
        return {
            'model': model,
            'optimizer': optimizer,
            'metric_name': config['training']['optimization_metric'],
            'metric_value': 0.0,
            'hyperparameters': {
                'hidden_layers': [256, 128, 64],
                'dropout_rate': 0.2,
                'lr': config['training']['optimizer_params'][config['training']['optimizer_choice']]['lr'],
                'use_batch_norm': True,
                'weight_decay': 0.0
            }
        }

    # Load existing checkpoint with weights_only=True
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # Create model with saved hyperparameters
    model = MLPClassifier(
        input_size=config['model']['input_size'],
        hidden_layers=checkpoint['hyperparameters']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=checkpoint['hyperparameters']['dropout_rate'],
        use_batch_norm=checkpoint['hyperparameters']['use_batch_norm']
    )
    
    # Create optimizer
    optimizer = getattr(torch.optim, checkpoint['optimizer_name'])(
        model.parameters(),
        lr=checkpoint['hyperparameters']['lr'],
        weight_decay=checkpoint['hyperparameters'].get('weight_decay', 0.0)
    )
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'model': model,
        'optimizer': optimizer,
        'metric_name': checkpoint['metric_name'],
        'metric_value': checkpoint['metric_value'],
        'hyperparameters': checkpoint['hyperparameters']
    }

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create best_model section if it doesn't exist
    if 'best_model' not in config:
        config['best_model'] = {}
    
    # Format parameters for config
    hidden_layers = [best_params[f'hidden_layer_{i}'] for i in range(best_params['n_layers'])]
    
    config['best_model'].update({
        'hidden_layers': hidden_layers,
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': best_params['lr'],
        'use_batch_norm': best_params['use_batch_norm'],
        'weight_decay': best_params.get('weight_decay', 0.0),
        'best_metric_name': config['training']['optimization_metric'],
        'best_metric_value': best_trial.value
    })
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def train_final_model(config, train_loader, val_loader):
    """Train model using parameters from config."""
    best_model_config = config['best_model']
    
    final_model = MLPClassifier(
        input_size=config['model']['input_size'],
        hidden_layers=best_model_config['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=best_model_config['dropout_rate'],
        use_batch_norm=best_model_config['use_batch_norm']
    )
    
    criterion = getattr(nn, config['training']['loss_function'])()
    optimizer = getattr(torch.optim, config['training']['optimizer_choice'])(
        final_model.parameters(),
        lr=best_model_config['learning_rate'],
        weight_decay=best_model_config['weight_decay']
    )
    
    final_trainer = PyTorchTrainer(
        final_model, criterion, optimizer,
        device=config['training']['device'],
        verbose=True
    )
    
    return final_trainer.train(
        train_loader, 
        val_loader, 
        config['training']['epochs'],
        metric=config['training']['optimization_metric']
    )

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CPUOptimizer:
    """Handles CPU-specific optimizations for PyTorch training."""
    
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.logger = setup_logger(config, 'cpu_optimization')  # Initialize logger directly
        self.cpu_info = cpuinfo.get_cpu_info()
        
    def detect_cpu_features(self):
        """Detect CPU features and capabilities."""
        features = {
            'processor': self.cpu_info.get('brand_raw', 'Unknown'),
            'architecture': self.cpu_info.get('arch', 'Unknown'),
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'avx512': 'avx512' in self.cpu_info.get('flags', []),
            'avx2': 'avx2' in self.cpu_info.get('flags', []),
            'mkl': hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available(),
            'ipex': hasattr(torch, 'xpu') or hasattr(torch, 'ipex'),
            'bf16_supported': hasattr(ipex, 'core') and ipex.core.onednn_has_bf16_support()
        }
        return features
        
    def configure_thread_settings(self):
        """Configure thread settings before any PyTorch operations"""
        if self.config['training']['cpu_optimization']['num_threads'] == 'auto':
            num_threads = psutil.cpu_count(logical=True)
        else:
            num_threads = self.config['training']['cpu_optimization']['num_threads']
            
        # Set thread configurations
        torch.set_num_threads(num_threads)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(min(4, num_threads))
            
        # Pin CPU threads
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        return num_threads
        
    def configure_optimizations(self):
        """Configure CPU-specific optimizations based on detected features."""
        features = self.detect_cpu_features()
        optimizations = {}
        
        # Configure number of threads
        optimizations['num_threads'] = self.config['training']['cpu_optimization']['num_threads']
        
        # Configure MKL-DNN
        optimizations['enable_mkldnn'] = (
            features['avx512'] or features['avx2']
        ) and self.config['training']['cpu_optimization']['enable_mkldnn']
        
        # Configure data types based on hardware support
        optimizations['use_bfloat16'] = (
            features['bf16_supported'] and 
            self.config['training']['cpu_optimization']['use_bfloat16']
        )
        
        # Enable MKL-DNN if available
        if optimizations['enable_mkldnn']:
            torch.backends.mkldnn.enabled = True
        
        # Configure IPEX if available and model exists
        if features['ipex'] and self.model is not None and hasattr(self, 'optimizer'):
            try:
                # Remove channels_last memory format for 2D data
                dtype = torch.bfloat16 if features['bf16_supported'] else torch.float32
                self.logger.info(f"Using dtype: {dtype}")
                
                optimized = ipex.optimize(
                    self.model,
                    optimizer=self.optimizer,
                    dtype=dtype,
                    inplace=True,
                    auto_kernel_selection=True,
                    weights_prepack=features['bf16_supported']  # Only prepack weights if BF16 is supported
                )
                
                if isinstance(optimized, tuple):
                    self.model, self.optimizer = optimized
                else:
                    self.model = optimized
                
                # JIT trace for better performance if enabled
                if self.config['training']['cpu_optimization']['jit_compile']:
                    sample_input = torch.randn(1, self.config['model']['input_size'])
                    if dtype == torch.bfloat16:
                        sample_input = sample_input.to(torch.bfloat16)
                    self.model = torch.jit.trace(self.model, sample_input)
                    
            except Exception as e:
                self.logger.warning(f"IPEX optimization failed: {str(e)}")
                self.logger.warning("Falling back to default PyTorch execution")
                optimizations['use_bfloat16'] = False
        
        self.log_optimization_config(features, optimizations)
        return optimizations
        
    def log_optimization_config(self, features, optimizations):
        """Log CPU features and applied optimizations."""
        self.logger.info("CPU Configuration:")
        self.logger.info(f"Processor: {features['processor']}")
        self.logger.info(f"Architecture: {features['architecture']}")
        self.logger.info(f"Physical cores: {features['cores']}")
        self.logger.info(f"Logical threads: {features['threads']}")
        self.logger.info("\nCPU Features:")
        self.logger.info(f"AVX-512 support: {features['avx512']}")
        self.logger.info(f"AVX2 support: {features['avx2']}")
        self.logger.info(f"MKL support: {features['mkl']}")
        self.logger.info(f"IPEX support: {features['ipex']}")
        self.logger.info("\nApplied Optimizations:")
        self.logger.info(f"Number of threads: {optimizations['num_threads']}")
        self.logger.info(f"MKL-DNN enabled: {optimizations['enable_mkldnn']}")
        self.logger.info(f"BFloat16 enabled: {optimizations['use_bfloat16']}")

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Create necessary directories first
    os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
    os.makedirs(config['logging']['directory'], exist_ok=True)
    
    # Set up main logger
    logger = setup_logger(config, 'MLPTrainer')
    logger.info("Starting training process...")
    
    # Initialize CPU optimization (logger is now initialized in constructor)
    cpu_optimizer = CPUOptimizer(config)
    cpu_optimizer.configure_thread_settings()
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    logger.info(f"Set random seed to {config['training']['seed']}")
    
    # Continue with the rest of initialization
    optimizations = cpu_optimizer.configure_optimizations()
    
    # Create datasets and dataloaders with validation
    try:
        train_df = pd.read_csv(config['data']['train_path'])
        val_df = pd.read_csv(config['data']['val_path'])
        
        # Verify data is not empty
        if train_df.empty or val_df.empty:
            raise ValueError("Empty dataset detected")
            
        logger.info(f"Loaded training data: {train_df.shape}, validation data: {val_df.shape}")
        
        # Verify target column exists
        target_column = config['data']['target_column']
        if target_column not in train_df.columns or target_column not in val_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        train_dataset = CustomDataset(train_df, target_column)
        val_dataset = CustomDataset(val_df, target_column)
        
        # Verify datasets are not empty
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty dataset after preprocessing")
            
        logger.info(f"Created datasets - train: {len(train_dataset)}, val: {len(val_dataset)}")
        
        # Enhanced DataLoader configuration with validation
        dataloader_args = {
            'batch_size': config['training']['batch_size'] * config['training']['performance']['batch_size_multiplier'],
            'num_workers': optimal_workers,
            'pin_memory': config['training']['dataloader']['pin_memory'],
            'persistent_workers': config['training']['dataloader']['persistent_workers'],
            'prefetch_factor': config['training']['dataloader']['prefetch_factor'],
            'drop_last': config['training']['drop_last']
        }
        
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_args)
        
        # Verify loaders are properly initialized
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Empty data loader detected")
            
        logger.info(f"Created data loaders - train batches: {len(train_loader)}, val batches: {len(val_loader)}")
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise
    
    # If best parameters don't exist in config, run hyperparameter tuning
    if 'best_model' not in config:
        tuner = HyperparameterTuner(config)
        best_trial, best_params = tuner.tune(train_loader, val_loader)
        save_best_params_to_config(config_path, best_trial, best_params)
        # Reload config with saved parameters
        config = load_config(config_path)
    
    print("\nBest model parameters from config:")
    for key, value in config['best_model'].items():
        print(f"    {key}: {value}")
    
    # Restore best model from checkpoint
    print("\nRestoring best model from checkpoint...")
    restored = restore_best_model(config)
    model = restored['model']
    optimizer = restored['optimizer']
    
    # Create criterion for evaluation
    criterion = getattr(nn, config['training']['loss_function'])()
    
    # Add scheduler with corrected parameters and type conversion
    scheduler_params = config['training']['scheduler']['params'].copy()
    max_lr_factor = float(scheduler_params.pop('max_lr_factor', 10.0))
    base_lr = config['training']['optimizer_params']['Adam']['lr']
    max_lr = base_lr * max_lr_factor
    
    # Convert string values to float
    for key in ['div_factor', 'final_div_factor', 'pct_start']:
        if key in scheduler_params:
            scheduler_params[key] = float(scheduler_params[key])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader),
        **scheduler_params
    )
    
    # Initialize CPU optimization with model and optimizer
    cpu_optimizer = CPUOptimizer(config, model)
    cpu_optimizer.optimizer = optimizer  # Add optimizer to CPU optimizer
    optimizations = cpu_optimizer.configure_optimizations()
    # Get the optimized model and optimizer back
    model = cpu_optimizer.model
    optimizer = cpu_optimizer.optimizer
    
    # Create trainer for evaluation
    trainer = PyTorchTrainer(
        model, criterion, optimizer,
        device=config['training']['device'],
        verbose=True,
        scheduler=scheduler,
        config=config
    )
    
    # Evaluate restored model
    print("\nEvaluating restored model on validation set...")
    val_loss, val_accuracy, val_f1 = trainer.evaluate(val_loader)
    
    metric_name = config['training']['optimization_metric']
    metric_value = val_f1 if metric_name == 'f1' else val_accuracy
    
    print(f"\nRestored model performance:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")
    print(f"Current {metric_name.upper()}: {metric_value:.4f}")

if __name__ == "__main__":
    main()