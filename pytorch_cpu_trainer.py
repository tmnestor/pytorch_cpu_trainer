import gc
import datetime  # Add this import at the top with other imports
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
import torch.optim.swa_utils as swa_utils
import argparse
from model_history import ModelHistory, update_default_config


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
    log_path = os.path.join(log_dir, f'{name}.log')
    # Clear existing log file
    with open(log_path, 'w') as f:
        # Write header
        f.write(f"=== {name} Log ===\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
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
    def __init__(self, input_size, hidden_layers, num_classes=3, dropout_rate=0.2, use_batch_norm=True, config=None):
        super(MLPClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.residuals = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            # Main layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # BatchNorm1d with fixed momentum for stable training
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_size, momentum=0.1))
            else:
                self.norms.append(nn.Identity())  # No normalization
                
            self.drops.append(nn.Dropout(dropout_rate))
            
            # Residual connection if sizes match, otherwise projection
            if prev_size == hidden_size:
                self.residuals.append(nn.Identity())
            else:
                self.residuals.append(nn.Linear(prev_size, hidden_size))
                
            prev_size = hidden_size
        
        self.final = nn.Linear(prev_size, num_classes)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        prev_x = x
        for i, (layer, norm, drop, residual) in enumerate(zip(
            self.layers, self.norms, self.drops, self.residuals)):
            # Main branch
            x = layer(x)
            x = norm(x)
            x = self.gelu(x)
            x = drop(x)
            
            # Add residual connection
            x = x + residual(prev_x)
            prev_x = x
            
        return self.final(x)

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
        
        # Enable oneDNN optimizations
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
            torch.backends.mkl.enabled = True
            # Set optimized memory format
            model = model.to(memory_format=torch.channels_last)
            
        # Initialize mixed precision settings based on hardware support
        self.use_mixed_precision = hasattr(ipex, 'core') and ipex.core.onednn_has_bf16_support()
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
            # Enable graph mode for better performance
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
        
        # Initialize SWA if enabled
        swa_enabled = config.get('training', {}).get('swa', {}).get('enabled', False)
        if swa_enabled:
            self.swa_model = swa_utils.AveragedModel(model)
            self.swa_scheduler = swa_utils.SWALR(
                optimizer,
                swa_lr=config['training']['swa']['lr']
            )
            self.swa_start = config['training']['swa']['start_epoch']
        else:
            self.swa_model = None
            self.swa_scheduler = None
            self.swa_start = float('inf')
        
        # Initialize warmup with safe defaults
        if config and config.get('training', {}).get('warmup', {}).get('enabled', False):
            warmup_steps = config['training']['warmup'].get('max_steps', 1000)
            initial_lr = config['training']['optimizer_params'][config['training']['optimizer_choice']]['lr'] / 100
            target_lr = config['training']['optimizer_params'][config['training']['optimizer_choice']]['lr']
            self.warmup_scheduler = CPUWarmupScheduler(optimizer, warmup_steps, initial_lr, target_lr)
        else:
            self.warmup_scheduler = None
        
    def train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
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
            with torch.amp.autocast('cpu', enabled=self.use_mixed_precision, dtype=torch.bfloat16):
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
            
            # Accumulate total loss properly
            total_loss += loss.item() * self.grad_accum_steps
            
            # Update metrics
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
            
            # Memory cleanup
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Calculate metrics properly
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_preds) > 0 else 0.0
        return total_loss / len(train_loader), accuracy, f1
    
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
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # Store predictions and labels for F1 score calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        try:
            accuracy = correct / total if total > 0 else 0.0
            f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_preds) > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            accuracy = 0.0
            f1 = 0.0
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        return avg_loss, accuracy, f1

    def plot_learning_curves(self, train_losses, val_losses, train_metrics, val_metrics, metric_name='Accuracy'):
        """Plots the learning curves for loss and chosen metric (accuracy or F1)."""
        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_title(f'Training and Validation {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save in figures directory
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'learning_curves.png'))
        plt.close()

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model for specified number of epochs."""
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        best_val_metric = float('-inf')
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Warmup and SWA handling
            if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler and epoch < self.warmup_scheduler.warmup_steps:
                self.warmup_scheduler.step()
            elif hasattr(self, 'swa_model') and self.swa_model is not None and epoch >= getattr(self, 'swa_start', float('inf')):
                self.swa_model.update_parameters(self.model)
                if hasattr(self, 'swa_scheduler'):
                    self.swa_scheduler.step()
            else:
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Training and evaluation
            train_loss, train_accuracy, train_f1 = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # Select metric based on config
            train_metric = train_f1 if metric == 'f1' else train_accuracy
            val_metric = val_f1 if metric == 'f1' else val_accuracy
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            best_val_metric = max(best_val_metric, val_metric)
            
            if self.verbose:
                metric_name = 'F1' if metric == 'f1' else 'Accuracy'
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f'Train {metric_name}: {train_metric:.4f}, Val {metric_name}: {val_metric:.4f}')
            
            # Memory cleanup
            gc.collect()
        
        # Plot learning curves at the end of training
        self.plot_learning_curves(
            train_losses, val_losses,
            train_metrics, val_metrics,
            metric_name='F1-Score' if metric == 'f1' else 'Accuracy'
        )
        
        return train_losses, val_losses, train_metrics, val_metrics, best_val_metric

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.best_trial_value = float('-inf')  # Fix syntax error here
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
        n_layers = trial.suggest_int('n_layers', 1, 4)
        # Single width for all layers
        layer_width = trial.suggest_int('layer_width', 32, 512)
        hidden_layers = [layer_width] * n_layers
        
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        weight_decay = 0.0 if use_batch_norm else trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        
        # Create model with config
        model = MLPClassifier(
            input_size=self.config['model']['input_size'],
            hidden_layers=hidden_layers,
            num_classes=self.config['model']['num_classes'],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            config=self.config
        )
        
        # Create optimizer
        optimizer = getattr(torch.optim, self.config['training']['optimizer_choice'])(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        trial_params = {
            'n_layers': n_layers,
            'layer_width': layer_width,
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
    logger = setup_logger(config, 'MLPTrainer')
    checkpoint_path = config['model']['save_path']
    
    try:
        # Try to load checkpoint first
        if os.path.exists(checkpoint_path):
            # ...existing checkpoint loading code...
            pass
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}. Using default model.")
    
    # Use default model configuration from historical best if available
    if 'default_model' in config:
        logger.info("Using configuration from historical best performers")
        hidden_layers = config['default_model']['hidden_layers']
        dropout_rate = config['default_model']['dropout_rate']
        learning_rate = config['default_model']['learning_rate']
        use_batch_norm = config['default_model']['use_batch_norm']
        weight_decay = config['default_model'].get('weight_decay', 0.0)
    else:
        logger.info("No historical best configuration found, using hardcoded defaults")
        hidden_layers = [128] * 3
        dropout_rate = 0.2
        learning_rate = config['training']['optimizer_params'][config['training']['optimizer_choice']]['lr']
        use_batch_norm = True
        weight_decay = 0.0
    
    model = MLPClassifier(
        input_size=config['model']['input_size'],
        hidden_layers=hidden_layers,
        num_classes=config['model']['num_classes'],
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        config=config
    )
    
    optimizer = getattr(torch.optim, config['training']['optimizer_choice'])(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'metric_name': config['training']['optimization_metric'],
        'metric_value': 0.0,
        'hyperparameters': {
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'lr': learning_rate,
            'use_batch_norm': use_batch_norm,
            'weight_decay': weight_decay
        }
    }

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create best_model section if it doesn't exist
    if 'best_model' not in config:
        config['best_model'] = {}
    
    # Create hidden layers list from layer width and number of layers
    hidden_layers = [best_params['layer_width']] * best_params['n_layers']
    
    config['best_model'].update({
        'hidden_layers': hidden_layers,
        'layer_width': best_params['layer_width'],
        'n_layers': best_params['n_layers'],
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
        use_batch_norm=best_model_config['use_batch_norm'],
        config=config  # Pass config here
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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))

class CPUWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0
        
    def step(self):
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch CPU Trainer')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'],
                      required=True, help='Mode to run the model: train or inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to config file (default: config.yaml)')
    parser.add_argument('--retrain', action='store_true',
                      help='Retrain model even if best model exists')
    return parser.parse_args()

def inference(model, val_loader, config, logger):
    """Run inference using the best saved model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Running inference with best saved model...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(val_loader, desc='Inference'):
            batch_X = batch_X.to(config['training']['device'])
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f"\nInference Results:")
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"F1-Score: {f1 * 100:.2f}%")
    
    return accuracy, f1

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories first
    os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
    os.makedirs(config['logging']['directory'], exist_ok=True)
    os.makedirs('input_data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Set up main logger BEFORE using it
    logger = setup_logger(config, 'MLPTrainer')
    logger.info(f"Starting in {args.mode} mode...")
    
    # Now we can use logger for update_default_config
    logger.info("Checking for historical best configurations")
    update_default_config(args.config)
    
    # Reload config after update
    config = load_config(args.config)
    logger.info("Configuration loaded with defaults:" + 
                f"\ndefault_model: {config.get('default_model', 'Not found')}")
    
    # Initialize CPU optimization early
    cpu_optimizer = CPUOptimizer(config)
    cpu_optimizer.configure_thread_settings()
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Load and validate data files
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Data files not found: {train_path} or {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Data validation
    target_column = config['data']['target_column']
    if target_column not in train_df.columns or target_column not in val_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Create datasets and dataloaders
    val_dataset = CustomDataset(val_df, target_column)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    if args.mode == 'inference':
        # Load best model and run inference
        if not os.path.exists(config['model']['save_path']):
            raise FileNotFoundError("No saved model found for inference")
        
        restored = restore_best_model(config)
        model = restored['model']
        logger.info(f"Loaded model with {restored['metric_name']} = {restored['metric_value']:.4f}")
        
        accuracy, f1 = inference(model, val_loader, config, logger)
        
    else:  # Training mode
        train_dataset = CustomDataset(train_df, target_column)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        should_train = args.retrain or 'best_model' not in config
        
        if should_train:
            logger.info("Starting training process...")
            tuner = HyperparameterTuner(config)
            best_trial, best_params = tuner.tune(train_loader, val_loader)
            save_best_params_to_config(args.config, best_trial, best_params)
            config = load_config(args.config)
        
        # Rest of the training code remains the same
        restored = restore_best_model(config)
        model = restored['model']
        optimizer = restored['optimizer']
        
        # Create criterion with label smoothing if enabled
        if config['training'].get('label_smoothing', {}).get('enabled', False):
            criterion = LabelSmoothingLoss(
                num_classes=config['model']['num_classes'],
                smoothing=config['training']['label_smoothing']['factor']
            )
        else:
            criterion = getattr(nn, config['training']['loss_function'])()
            
        # Configure scheduler
        scheduler_params = config['training']['scheduler']['params'].copy()
        max_lr = float(config['training']['optimizer_params']['Adam']['lr']) * \
                 float(scheduler_params.pop('max_lr_factor', 10.0))
                 
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=float(scheduler_params.get('pct_start', 0.3)),
            div_factor=float(scheduler_params.get('div_factor', 25.0)),
            final_div_factor=float(scheduler_params.get('final_div_factor', 1e4))
        )
        
        # Create trainer
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=config['training']['device'],
            verbose=True,
            scheduler=scheduler,
            config=config
        )
        
        # Train the model
        logger.info("Starting model training...")
        train_losses, val_losses, train_metrics, val_metrics, best_val_metric = trainer.train(
            train_loader,
            val_loader,
            config['training']['epochs'],
            metric=config['training']['optimization_metric']
        )
        
        logger.info(f"Training completed. Best validation {config['training']['optimization_metric']}: {best_val_metric:.4f}")
        
        # Final evaluation
        val_loss, val_accuracy, val_f1 = trainer.evaluate(val_loader)
        logger.info("\nFinal Model Performance:")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        logger.info(f"Validation F1-Score: {val_f1 * 100:.2f}%")
        
        # After training completes, save experiment results
        history = ModelHistory(args.config)  # Pass config path instead of using default
        history.save_experiment(
            args.config,
            best_val_metric,
            config['training']['optimization_metric']
        )

if __name__ == "__main__":
    main()