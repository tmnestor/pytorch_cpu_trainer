# File: pytorch_cpu_trainer.py

import random
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings
from typing import Dict, Optional, List, Any, Union
import platform
import time  # Add this import

# Add rich imports
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.backends.mkldnn
import torch.backends.mkl
import multiprocessing
from torch.utils.data import DataLoader, Dataset
import psutil
import optuna
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, ReduceLROnPlateau
from torch.optim import Optimizer

import contextlib
import tempfile
import atexit

from safetensors.torch import save_model, load_model
from collections import OrderedDict

import shutil
import json
import hashlib
from datetime import datetime

# ...existing imports...
import torch.cuda.memory as cuda_memory
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import psutil
from functools import partial

# Replace profiler imports with a safer implementation
try:
    from torch.profiler import profile, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

import torchmetrics
import inspect

import pathlib

# Add after imports
CHECKPOINT_DIR = pathlib.Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class SafeProfiler:
    """Safe profiler wrapper with graceful fallbacks"""
    def __init__(self, enabled=False, activities=None):
        self.enabled = enabled and PROFILER_AVAILABLE
        self.activities = activities or []
        self.start_mem = None
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_mem = psutil.Process().memory_info().rss
        if self.enabled:
            try:
                import torch.autograd.profiler as profiler
                profiler.enabled = True
            except:
                pass
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            try:
                import torch.autograd.profiler as profiler
                profiler.enabled = False
            except:
                pass
        
        # Always collect basic metrics
        end_mem = psutil.Process().memory_info().rss
        end_time = time.time()
        mem_diff_mb = (end_mem - self.start_mem) / (1024 * 1024)
        time_diff_s = end_time - self.start_time
        
        # Store metrics for later access
        self.metrics = {
            'memory_change_mb': mem_diff_mb,
            'duration_seconds': time_diff_s,
        }

class MemoryManager:
    """Manages memory optimizations including gradient accumulation and profiling"""
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('MemoryManager')
        self.grad_accum_steps = config['training']['performance'].get('grad_accum_steps', 1)
        self.enable_profiling = (config['training']['profiling'].get('enabled', False) 
                               and PROFILER_AVAILABLE)
        if config['training']['profiling'].get('enabled', False) and not PROFILER_AVAILABLE:
            self.logger.warning("Profiling was enabled in config but torch.profiler is not available. "
                              "Using basic memory tracking instead.")
        self.profile_data = {}
        self.initial_batch_size = config['training']['batch_size']
        self._safe_profiler = None
        
    def find_optimal_batch_size(self, model, sample_input, target_batch_size, 
                              min_batch_size=1, max_memory_usage=0.95):
        """Find largest possible batch size that fits in memory"""
        self.logger.info("Finding optimal batch size...")
        
        def try_batch_size(batch_size):
            try:
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Get current memory usage
                process = psutil.Process()
                start_mem = process.memory_info().rss
                
                # Try a forward and backward pass
                inputs = torch.stack([sample_input] * batch_size)
                outputs = model(inputs)
                loss = outputs.sum()
                loss.backward()
                
                # Check memory usage
                end_mem = process.memory_info().rss
                mem_used = (end_mem - start_mem) / 1024**3  # Convert to GB
                total_mem = psutil.virtual_memory().total / 1024**3
                
                # Clear test tensors
                del inputs, outputs, loss
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                return mem_used / total_mem < max_memory_usage
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return False
                raise e
        
        # Binary search for largest working batch size
        left, right = min_batch_size, target_batch_size
        optimal_batch_size = min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            if try_batch_size(mid):
                optimal_batch_size = mid
                left = mid + 1
            else:
                right = mid - 1
        
        self.logger.info(f"Found optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def start_memory_profiling(self):
        """Start memory usage profiling with robust fallback"""
        self._safe_profiler = SafeProfiler(
            enabled=self.enable_profiling,
            activities=[ProfilerActivity.CPU] if PROFILER_AVAILABLE else None
        )
        return self._safe_profiler
    
    @contextlib.contextmanager
    def _basic_memory_context(self):
        """Basic memory tracking context manager"""
        try:
            start_mem = psutil.Process().memory_info().rss
            yield
        finally:
            end_mem = psutil.Process().memory_info().rss
            mem_diff = end_mem - start_mem
            if mem_diff > 1024 * 1024:  # Only log if difference is more than 1MB
                self.logger.debug(f"Memory change: {mem_diff / (1024 * 1024):.2f} MB")

    def log_memory_stats(self):
        """Log current memory usage statistics with profiler metrics"""
        stats = {
            'cpu_percent': psutil.Process().cpu_percent(),
            'memory_rss_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'memory_percent': psutil.Process().memory_percent()
        }
        
        # Add profiler metrics if available
        if self._safe_profiler and hasattr(self._safe_profiler, 'metrics'):
            stats.update(self._safe_profiler.metrics)
        
        self.logger.debug("Memory Stats:")
        for key, value in stats.items():
            self.logger.debug(f"  {key}: {value:.2f}")
        
        return stats
    
    def should_accumulate_gradients(self, batch_idx):
        """Check if gradients should be accumulated"""
        return (batch_idx + 1) % self.grad_accum_steps != 0
    
    def get_effective_batch_size(self, batch_size):
        """Get effective batch size with gradient accumulation"""
        return batch_size * self.grad_accum_steps
    
    def optimize_memory(self, model, sample_input):
        """Run all memory optimizations"""
        # Find optimal batch size
        optimal_batch_size = self.find_optimal_batch_size(
            model, sample_input, self.initial_batch_size
        )
        
        # Update grad accumulation steps if needed
        if optimal_batch_size < self.initial_batch_size:
            self.grad_accum_steps = max(
                self.grad_accum_steps,
                self.initial_batch_size // optimal_batch_size
            )
            self.logger.info(f"Adjusted gradient accumulation steps to {self.grad_accum_steps}")
        
        # Log memory optimization results
        self.logger.info("Memory optimization results:")
        self.logger.info(f"  Optimal batch size: {optimal_batch_size}")
        self.logger.info(f"  Gradient accumulation steps: {self.grad_accum_steps}")
        self.logger.info(f"  Effective batch size: {self.get_effective_batch_size(optimal_batch_size)}")
        
        return optimal_batch_size

class CPUOptimizationManager:
    _instance = None
    _initialized = False
    _mixed_precision_logged = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CPUOptimizationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Set up logger with both file and console handlers
            self.logger = logging.getLogger("CPUOptimizer")
            self.logger.setLevel(logging.INFO)
            
            # Remove any existing handlers
            self.logger.handlers = []
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler('logs/cpu_optimization.log')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.enabled_features = []
            self.disabled_features = []
            self._log_initial_status()
            self._initialized = True
    
    def _log_initial_status(self):
        """Log CPU optimization status once at initialization"""
        self.logger.info("CPU Optimization Status:")
        self.logger.info(f"CPU Architecture: {platform.processor()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Try enabling MKL-DNN
        try:
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                self.enabled_features.append('MKL-DNN')
            else:
                self.disabled_features.append('MKL-DNN')
        except:
            self.disabled_features.append('MKL-DNN')
            
        # Try enabling MKL
        try:
            if hasattr(torch.backends, 'mkl'):
                torch.backends.mkl.enabled = True
                self.enabled_features.append('MKL')
            else:
                self.disabled_features.append('MKL')
        except:
            self.disabled_features.append('MKL')
            
        # Configure threading
        try:
            num_threads = multiprocessing.cpu_count()
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            self.enabled_features.append(f'Multi-threading ({num_threads} threads)')
        except:
            self.disabled_features.append('Custom thread configuration')
            
        # Check bfloat16 support
        try:
            if hasattr(torch, 'bfloat16'):
                self.enabled_features.append('Mixed Precision (bfloat16)')
                if not self._mixed_precision_logged:
                    self.logger.info("Using mixed precision training with bfloat16")
                    self._mixed_precision_logged = True
            else:
                self.disabled_features.append('Mixed Precision')
        except:
            self.disabled_features.append('Mixed Precision')

        # Log enabled features
        if self.enabled_features:
            self.logger.info("Enabled optimizations:")
            for feature in self.enabled_features:
                self.logger.info(f"  {feature}")
        
        # Log disabled features
        if self.disabled_features:
            self.logger.info("Disabled/Unavailable optimizations:")
            for feature in self.disabled_features:
                self.logger.info(f"  {feature}")

    def supports_mixed_precision(self):
        return 'Mixed Precision (bfloat16)' in self.enabled_features

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
# Set up logging
def setup_logger(name='MLPTrainer', config=None):
    """Enhanced logger setup with rich formatting"""
    if config is None:
        log_dir = "logs"
        console_level = "WARNING"  # Default to WARNING for console
        file_level = "INFO"
    else:
        log_dir = config.get('logging', {}).get('directory', 'logs')
        console_level = config.get('logging', {}).get('console_level', 'WARNING')  # Default to WARNING
        file_level = config.get('logging', {}).get('file_level', 'INFO')

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Rich console handler with custom formatting
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    rich_handler.setLevel(getattr(logging, console_level))
    
    # File handler with detailed formatting
    log_file = os.path.join(log_dir, f"{name.lower()}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, file_level))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    
    logger.addHandler(rich_handler)
    logger.addHandler(fh)
    
    return logger

def set_performance_configs():
    """Configure PyTorch CPU performance settings"""
    torch.backends.mkldnn.enabled = True
    torch.backends.mkl.enabled = True
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())
    
    try:
        proc = psutil.Process()
        proc.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except:
        pass

class MemoryEfficientDataset(Dataset):
    def __init__(self, df, target_column, chunk_size=1000):
        """Memory efficient dataset implementation using memory mapping with proper cleanup"""
        # Create temporary directory for mmap files that will be automatically cleaned up
        self.temp_dir = tempfile.mkdtemp()
        self.features_file = os.path.join(self.temp_dir, 'features.mmap')
        self.labels_file = os.path.join(self.temp_dir, 'labels.mmap')
        self.chunk_size = chunk_size
        
        # Register cleanup on program exit
        atexit.register(self.cleanup)
        
        try:
            # Save data to memory-mapped files
            features = df.drop(target_column, axis=1).values
            labels = df[target_column].values
            
            self.features_mmap = np.memmap(
                self.features_file, 
                dtype='float32', 
                mode='w+', 
                shape=features.shape
            )
            self.labels_mmap = np.memmap(
                self.labels_file, 
                dtype='int64', 
                mode='w+', 
                shape=labels.shape
            )
            
            # Write data in chunks with proper synchronization
            for i in range(0, len(features), chunk_size):
                end_idx = min(i + chunk_size, len(features))
                self.features_mmap[i:end_idx] = features[i:end_idx]
                self.labels_mmap[i:end_idx] = labels[i:end_idx]
                # Force synchronization to disk
                self.features_mmap.flush()
                self.labels_mmap.flush()
            
            self.len = len(features)
            
        except Exception as e:
            self.cleanup()
            raise e
    
    def cleanup(self):
        """Clean up memory-mapped files and temporary directory"""
        try:
            if hasattr(self, 'features_mmap'):
                self.features_mmap._mmap.close()
                del self.features_mmap
            if hasattr(self, 'labels_mmap'):
                self.labels_mmap._mmap.close()
                del self.labels_mmap
            
            # Remove temporary files
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.features_file)
                os.remove(self.labels_file)
                os.rmdir(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Error during cleanup: {e}")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Load data from memory-mapped files
        X = torch.FloatTensor(self.features_mmap[idx])
        y = torch.LongTensor([self.labels_mmap[idx]])
        return X, y.squeeze()
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()

# Update CustomDataset to use MemoryEfficientDataset for large datasets
class CustomDataset(Dataset):
    def __init__(self, df, target_column):
        memory_threshold = 1e9  # 1GB
        estimated_memory = df.memory_usage().sum()
        
        if estimated_memory > memory_threshold:
            self.dataset = MemoryEfficientDataset(df, target_column)
            self.use_memory_efficient = True
        else:
            self.use_memory_efficient = False
            # Convert to tensors without pinning memory initially
            self.features = torch.FloatTensor(df.drop(target_column, axis=1).values)
            self.labels = torch.LongTensor(df[target_column].values)
            
            # Only pin memory if CUDA is available
            if torch.cuda.is_available():
                self.features = self.features.pin_memory()
                self.labels = self.labels.pin_memory()
    
    def __len__(self):
        return len(self.dataset) if self.use_memory_efficient else len(self.features)
    
    def __getitem__(self, idx):
        if self.use_memory_efficient:
            return self.dataset[idx]
        return self.features[idx], self.labels[idx]
    
    def __del__(self):
        """Ensure proper cleanup"""
        if self.use_memory_efficient:
            del self.dataset
        else:
            del self.features
            del self.labels

class BatchNormManager:
    """Manages BatchNorm layers with enhanced monitoring and control"""
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger('BatchNormManager')
        self.bn_layers = {name: module for name, module in model.named_modules() 
                         if isinstance(module, nn.BatchNorm1d)}
        self.stats_history = {}
        self.momentum_range = (0.01, 0.1)  # Add momentum range
        self.variance_threshold = 0.01      # Add variance stability threshold
        self._setup_tracking()
    
    def _setup_tracking(self):
        """Initialize statistics tracking for each BatchNorm layer"""
        for name, layer in self.bn_layers.items():
            self.stats_history[name] = {
                'running_mean': [],
                'running_var': [],
                'momentum': [],
                'num_batches_tracked': []
            }
    
    def update_momentum(self, epoch, total_epochs):
        """Improved momentum scheduling with more gradual changes"""
        min_momentum, max_momentum = self.momentum_range
        # More gradual momentum change using sigmoid instead of cosine
        progress = epoch / total_epochs
        momentum = min_momentum + (max_momentum - min_momentum) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        
        for layer in self.bn_layers.values():
            layer.momentum = momentum
            # Add eps to prevent division by zero
            layer.eps = 1e-5  
            
        self.logger.debug(f"Updated BatchNorm momentum to {momentum:.4f}")
        
    def capture_statistics(self):
        """Capture current BatchNorm statistics"""
        for name, layer in self.bn_layers.items():
            self.stats_history[name]['running_mean'].append(
                layer.running_mean.cpu().numpy().copy()
            )
            self.stats_history[name]['running_var'].append(
                layer.running_var.cpu().numpy().copy()
            )
            self.stats_history[name]['momentum'].append(layer.momentum)
            self.stats_history[name]['num_batches_tracked'].append(
                layer.num_batches_tracked.item()
            )
    
    def validate_state(self, threshold=0.1, early_warning=False):
        """Enhanced validation with early warning detection"""
        issues = []
        dead_neurons = []
        unstable_vars = []
        
        for name, layer in self.bn_layers.items():
            # Check for dead neurons (constant mean/var)
            if len(self.stats_history[name]['running_mean']) > 1:
                mean_diff = np.abs(np.diff(self.stats_history[name]['running_mean'][-2:], axis=0))
                var_diff = np.abs(np.diff(self.stats_history[name]['running_var'][-2:], axis=0))
                
                # Early warning checks for trending issues
                if early_warning and len(self.stats_history[name]['running_mean']) > 5:
                    mean_trend = np.abs(np.diff(self.stats_history[name]['running_mean'][-5:], axis=0))
                    var_trend = np.abs(np.diff(self.stats_history[name]['running_var'][-5:], axis=0))
                    
                    if np.any(mean_trend.mean(axis=0) < 1e-7):
                        issues.append(f"Layer {name}: Potential dead neurons detected (low mean activity)")
                    if np.any(var_trend.mean(axis=0) > threshold * 2):
                        issues.append(f"Layer {name}: Potential unstable variances detected")
                
                current_dead = np.where(mean_diff < 1e-6)[0]
                current_unstable = np.where(var_diff > threshold)[0]
                
                if len(current_dead) > 0:
                    dead_neurons.extend(current_dead)
                    issues.append(f"Layer {name}: {len(current_dead)} dead neurons detected")
                if len(current_unstable) > 0:
                    unstable_vars.extend(current_unstable)
                    issues.append(f"Layer {name}: {len(current_unstable)} neurons with unstable variance")
                
                # Check for saturated activations
                if np.any(np.abs(layer.running_mean.cpu().numpy()) > 3):
                    issues.append(f"Layer {name}: Saturated activations detected")
        
        if issues:
            self.logger.warning("BatchNorm issues detected, applying fixes...")
            self._apply_fixes(dead_neurons, unstable_vars)
            
        return issues
    
    def _apply_fixes(self, dead_neurons, unstable_vars):
        """Enhanced fixes for BatchNorm issues"""
        for name, layer in self.bn_layers.items():
            if len(dead_neurons) > 0:
                # Reinitialize dead neurons with small random values
                layer.running_mean[dead_neurons] = torch.randn(len(dead_neurons)) * 0.1
                layer.running_var[dead_neurons] = torch.ones(len(dead_neurons))
                
            if len(unstable_vars) > 0:
                # Stabilize unstable variances with exponential moving average
                layer.momentum = max(layer.momentum * 0.8, 0.01)  # More aggressive momentum reduction
                layer.eps = max(layer.eps * 1.2, 1e-4)  # More aggressive numerical stability
                
                # Add variance smoothing
                current_vars = layer.running_var[unstable_vars].cpu().numpy()
                smoothed_vars = np.maximum(
                    np.exp(np.log(current_vars).mean()),  # Geometric mean
                    np.ones_like(current_vars) * 0.1  # Minimum variance
                )
                layer.running_var[unstable_vars] = torch.tensor(smoothed_vars, device=layer.running_var.device)
    
    def get_statistics_summary(self):
        """Generate summary of BatchNorm statistics"""
        summary = {}
        for name, stats in self.stats_history.items():
            means = np.array(stats['running_mean'])
            vars = np.array(stats['running_var'])
            
            summary[name] = {
                'mean_stability': np.mean(np.abs(np.diff(means, axis=0))),
                'var_stability': np.mean(np.abs(np.diff(vars, axis=0))),
                'momentum_range': (min(stats['momentum']), max(stats['momentum'])),
                'batches_tracked': stats['num_batches_tracked'][-1]
            }
        return summary
    
    def plot_statistics(self, save_dir='figures'):
        """Plot BatchNorm statistics evolution"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, stats in self.stats_history.items():
            means = np.array(stats['running_mean'])
            vars = np.array(stats['running_var'])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot mean evolution
            ax1.plot(means)
            ax1.set_title(f'{name} Running Mean Evolution')
            ax1.set_xlabel('Update Step')
            ax1.set_ylabel('Mean Value')
            
            # Plot variance evolution
            ax2.plot(vars)
            ax2.set_title(f'{name} Running Variance Evolution')
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Variance Value')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'bn_stats_{name}.png'))
            plt.close()

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes=3, dropout_rate=0.2, use_batch_norm=True):
        super(MLPClassifier, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.bn_layers = OrderedDict()  # Track BatchNorm layers
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                bn = nn.BatchNorm1d(hidden_size)
                layers.append(bn)
                self.bn_layers[f'bn_{i}'] = bn  # Register BatchNorm layer
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
        
        # Initialize BatchNorm manager
        if use_batch_norm:
            self.bn_manager = BatchNormManager(self)
    
    def get_bn_statistics(self):
        """Get BatchNorm running statistics for verification"""
        stats = {}
        if self.use_batch_norm:
            for name, layer in self.bn_layers.items():
                stats[name] = {
                    'mean': layer.running_mean.clone(),
                    'var': layer.running_var.clone(),
                }
        return stats
    
    def forward(self, x):
        return self.model(x)
        
class CPUOptimizer:
    """Handles CPU-specific optimizations with graceful fallbacks"""
    _instance = None
    _initialized = False
    
    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(CPUOptimizer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config, logger):
        # Only initialize once
        if not self._initialized:
            self.config = config
            self.logger = logger
            self.enabled_features = []
            self.disabled_features = []
            self.setup_cpu_optimizations()
            self._initialized = True
            self._log_optimization_status()

    def setup_cpu_optimizations(self) -> None:
        """Configure available CPU optimizations"""
        perf_config = self.config['training'].get('performance', {})
        
        # Try enabling MKL-DNN
        if perf_config.get('enable_mkldnn', False):
            try:
                import torch.backends.mkldnn
                torch.backends.mkldnn.enabled = True
                self.enabled_features.append('MKL-DNN')
            except:
                self.disabled_features.append('MKL-DNN')
                self.logger.warning("MKL-DNN optimization not available")

        # Try enabling MKL
        if perf_config.get('enable_mkl', False):
            try:
                import torch.backends.mkl
                torch.backends.mkl.enabled = True
                self.enabled_features.append('MKL')
            except:
                self.disabled_features.append('MKL')
                self.logger.warning("MKL optimization not available")

        # Configure number of threads
        try:
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            self.enabled_features.append(f'Multi-threading ({num_threads} threads)')
        except:
            self.disabled_features.append('Custom thread configuration')
            self.logger.warning("Failed to set custom thread configuration")

        # Try enabling mixed precision
        if perf_config.get('mixed_precision', False):
            try:
                # Check if the CPU supports bfloat16
                if not hasattr(torch, 'bfloat16'):
                    raise RuntimeError("bfloat16 not supported")
                self.enabled_features.append('Mixed Precision (bfloat16)')
            except:
                self.disabled_features.append('Mixed Precision')
                self.logger.warning("Mixed precision training not available on this CPU")

    def _log_optimization_status(self) -> None:
        """Log the status of CPU optimizations"""
        self.logger.info("CPU Optimization Status:")
        self.logger.info(f"CPU Architecture: {platform.processor()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        if self.enabled_features:
            self.logger.info("Enabled optimizations:")
            for feature in self.enabled_features:
                self.logger.info(f"  - {feature}")
        
        if self.disabled_features:
            self.logger.info("Disabled/Unavailable optimizations:")
            for feature in self.disabled_features:
                self.logger.info(f"  - {feature}")

class WarmupScheduler:
    """Handles learning rate warmup and scheduling"""
    def __init__(self, optimizer: Optimizer, config: Dict[str, Any], num_training_steps: int):
        self.warmup_steps = config['best_model']['warmup_steps']
        if config['optimization']['warmup']['enabled']:
            # Linear warmup followed by cosine decay
            self.warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            # OneCycleLR for the rest of training
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config['best_model']['learning_rate'],
                total_steps=num_training_steps - self.warmup_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            self.warmup = None
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config['best_model']['learning_rate'],
                total_steps=num_training_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

    def step(self, step_num: int):
        if self.warmup and step_num < self.warmup_steps:
            self.warmup.step()
        else:
            self.scheduler.step()

class CheckpointManager:
    """Handles atomic checkpoint operations and versioning"""
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('CheckpointManager')
        self.checkpoint_dir = CHECKPOINT_DIR  # Changed from config['model']['save_path']
        self.max_checkpoints = config.get('training', {}).get('checkpointing', {}).get('max_checkpoints', 5)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint metadata if exists
        self.metadata_file = CHECKPOINT_DIR / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load or initialize checkpoint metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning("Corrupted metadata file, initializing new metadata")
        return {'checkpoints': [], 'version': 0}
    
    def _save_metadata(self):
        """Save checkpoint metadata atomically"""
        tmp_file = f"{self.metadata_file}.tmp"
        try:
            with open(tmp_file, 'w') as f:
                json.dump(self.metadata, f)
            os.replace(tmp_file, self.metadata_file)
        except Exception as e:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _compute_hash(self, model):
        """Compute model state hash for verification"""
        state_dict = model.state_dict()
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            hasher.update(state_dict[key].cpu().numpy().tobytes())
        return hasher.hexdigest()
    
    def save_checkpoint(self, model, optimizer, epoch, metric_value, params):
        """Save checkpoint atomically with verification"""
        # Create checkpoint version and filename
        version = self.metadata['version'] + 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"checkpoint_v{version}_{timestamp}"
        
        # Create temporary directory for atomic save
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, f"{base_name}.safetensors")
            meta_path = os.path.join(tmp_dir, f"{base_name}.pt")
            
            # Save model weights using safetensors
            save_model(model, model_path)
            
            # Compute model state hash
            model_hash = self._compute_hash(model)
            
            # Save metadata and additional info
            checkpoint_meta = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'metric_value': metric_value,
                'hyperparameters': params,
                'model_hash': model_hash,
                'version': version,
                'timestamp': timestamp
            }
            
            # Save metadata atomically
            torch.save(checkpoint_meta, meta_path)
            
            # Verify files were written correctly
            if not (os.path.exists(model_path) and os.path.exists(meta_path)):
                raise IOError("Failed to save checkpoint files")
            
            # Move files to final location
            final_model_path = os.path.join(self.checkpoint_dir, f"{base_name}.safetensors")
            final_meta_path = os.path.join(self.checkpoint_dir, f"{base_name}.pt")
            
            shutil.move(model_path, final_model_path)
            shutil.move(meta_path, final_meta_path)
            
            # Update metadata
            checkpoint_info = {
                'version': version,
                'timestamp': timestamp,
                'model_file': final_model_path,
                'meta_file': final_meta_path,
                'metric_value': metric_value,
                'model_hash': model_hash
            }
            
            self.metadata['checkpoints'].append(checkpoint_info)
            self.metadata['version'] = version
            self._save_metadata()
            
            # Rotate old checkpoints
            self._rotate_checkpoints()
            
            self.logger.info(f"Saved checkpoint version {version} with metric value {metric_value:.4f}")
            
            return checkpoint_info
    
    def load_checkpoint(self, version=None):
        """Load and verify checkpoint"""
        if not self.metadata['checkpoints']:
            return None
        
        # Get checkpoint info
        if version is None:
            checkpoint_info = max(self.metadata['checkpoints'], 
                                key=lambda x: (x['metric_value'], x['version']))
        else:
            checkpoint_info = next((cp for cp in self.metadata['checkpoints'] 
                                  if cp['version'] == version), None)
            if not checkpoint_info:
                raise ValueError(f"Checkpoint version {version} not found")
        
        try:
            # Load metadata first
            checkpoint_meta = torch.load(checkpoint_info['meta_file'])
            
            # Create model and load weights
            model = MLPClassifier(
                input_size=self.config['model']['input_size'],
                hidden_layers=checkpoint_meta['hyperparameters']['hidden_layers'],
                num_classes=self.config['model']['num_classes'],
                dropout_rate=checkpoint_meta['hyperparameters']['dropout_rate'],
                use_batch_norm=checkpoint_meta['hyperparameters']['use_batch_norm']
            )
            
            # Load weights using safetensors
            load_model(model, checkpoint_info['model_file'])
            
            # Verify model state
            current_hash = self._compute_hash(model)
            if current_hash != checkpoint_info['model_hash']:
                raise ValueError("Model state verification failed")
            
            self.logger.info(f"Successfully loaded checkpoint version {checkpoint_info['version']}")
            return checkpoint_meta, model
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints"""
        if len(self.metadata['checkpoints']) > self.max_checkpoints:
            # Sort by metric value and version, keep best ones
            sorted_checkpoints = sorted(
                self.metadata['checkpoints'],
                key=lambda x: (x['metric_value'], x['version']),
                reverse=True
            )
            
            # Remove excess checkpoints
            for checkpoint in sorted_checkpoints[self.max_checkpoints:]:
                try:
                    os.remove(checkpoint['model_file'])
                    os.remove(checkpoint['meta_file'])
                    self.metadata['checkpoints'].remove(checkpoint)
                except Exception as e:
                    self.logger.error(f"Error removing checkpoint: {e}")
            
            self._save_metadata()

class TrainingHistory:
    """Maintains training history and handles checkpointing"""
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger or logging.getLogger('TrainingHistory')
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': [],
            'checkpoints': []
        }
        self.checkpoint_dir = os.path.dirname(config['model']['save_path'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(config, logger)
        
    def update(self, epoch, train_loss, val_loss, train_metric, val_metric, lr):
        """Update training history"""
        self.history['epochs'].append(epoch)
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['train_metrics'].append(train_metric)
        self.history['val_metrics'].append(val_metric)
        self.history['learning_rates'].append(lr)
        
    def save_checkpoint(self, epoch, model, optimizer, metric_value, params):
        """Save checkpoint using CheckpointManager"""
        return self.checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, metric_value, params
        )
    
    def load_checkpoint(self, version=None):
        """Load checkpoint using CheckpointManager"""
        return self.checkpoint_manager.load_checkpoint(version)

class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, config=None, device='cpu', verbose=False):
        """Initialize trainer with optional config parameter."""
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.config = config
        
        # Get or create logger
        self.logger = logging.getLogger('PyTorchTrainer')
        
        # Initialize scheduler-related attributes with safe defaults
        self.warmup_steps = 0
        self.scheduler = None
        self.current_step = 0
        
        # Only try to access config if it exists and has best_model
        if config is not None and 'best_model' in config:
            self.warmup_steps = config['best_model'].get('warmup_steps', 0)
            if config['optimization']['warmup']['enabled']:
                num_training_steps = config['training']['epochs']
                self.setup_warmup_scheduler(num_training_steps)
        
        # Initialize training history with the trainer's logger
        self.history = TrainingHistory(config, self.logger) if config else None

        self.adaptive_optimizer = AdaptiveOptimizer(optimizer, config) if config else None

        # Set up rich progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console()
        )

        self.memory_manager = MemoryManager(config, self.logger)

        self.metrics = MetricsManager(
            config=config,
            num_classes=config['model']['num_classes'],
            device=device
        )

    def setup_warmup_scheduler(self, num_training_steps):
        """Setup the learning rate scheduler with warmup."""
        if self.warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                total_steps=num_training_steps - self.warmup_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model with rich progress display."""
        if self.scheduler is None and self.warmup_steps > 0:
            num_training_steps = len(train_loader) * epochs
            self.setup_warmup_scheduler(num_training_steps)
            
        train_losses = []
        val_losses = []
        train_metrics_list = []  # Changed from train_metrics to train_metrics_list
        val_metrics_list = []    # Changed from val_metrics to val_metrics_list
        best_val_metric = 0
        lr_history = []
        
        with self.progress:
            epoch_task = self.progress.add_task("[green]Training epochs...", total=epochs)
            
            for epoch in range(epochs):
                # Track learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                
                # Train and evaluate
                train_loss, train_metrics = self.train_epoch(train_loader)
                val_loss, val_metrics = self.evaluate(val_loader)
                
                # Update metrics and progress
                train_metric = train_metrics[metric]
                val_metric = val_metrics[metric]
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_metrics_list.append(train_metric)  # Updated variable name
                val_metrics_list.append(val_metric)      # Updated variable name
                
                best_val_metric = max(best_val_metric, val_metric)
                
                # Update progress with rich formatting
                self.progress.update(
                    epoch_task,
                    advance=1,
                    description=f"[green]Epoch {epoch+1}/{epochs}"
                )
                
                if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                    # Update metric name display
                    metric_name = 'F1-Score' if metric == 'f1_score' else 'Accuracy'
                    
                    # Create rich table for metrics
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", justify="right", style="green")
                    table.add_row(f"Val {metric_name}", f"{val_metric:.2f}%")
                    table.add_row("Learning Rate", f"{current_lr:.6f}")
                    
                    rprint(table)
                
                # Update history and save checkpoints
                if self.history is not None:
                    self.history.update(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_metric=train_metric,
                        val_metric=val_metric,
                        lr=current_lr
                    )
                    
                    if (epoch + 1) % self.config.get('training', {}).get('checkpoint_frequency', 10) == 0:
                        self.history.save_checkpoint(
                            epoch=epoch,
                            model=self.model,
                            optimizer=self.optimizer,
                            metric_value=val_metric,
                            params=self.config.get('best_model', {})
                        )
                
                # Update BatchNorm momentum if model uses it
                if hasattr(self.model, 'bn_manager'):
                    self.model.bn_manager.update_momentum(epoch, epochs)
                
                # Capture BatchNorm statistics after epoch
                if hasattr(self.model, 'bn_manager'):
                    self.model.bn_manager.capture_statistics()
                    self.model.bn_manager.validate_state()
        
        # Plot BatchNorm statistics at end of training
        if hasattr(self.model, 'bn_manager'):
            self.model.bn_manager.plot_statistics(
                save_dir=self.config.get('logging', {}).get('figures_dir', 'figures')
            )
            stats_summary = self.model.bn_manager.get_statistics_summary()
            self.logger.info("\nBatchNorm Statistics Summary:")
            for layer, stats in stats_summary.items():
                self.logger.info(f"\n{layer}:")
                self.logger.info(f"  Mean stability: {stats['mean_stability']:.6f}")
                self.logger.info(f"  Variance stability: {stats['var_stability']:.6f}")
                self.logger.info(f"  Momentum range: {stats['momentum_range']}")
                self.logger.info(f"  Batches tracked: {stats['batches_tracked']}")
        
        # Display final summary
        summary = Table(title="Training Summary", show_header=True, header_style="bold magenta")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Best Value", justify="right", style="green")
        summary.add_row("Best Validation Metric", f"{best_val_metric:.4f}")
        summary.add_row("Final Learning Rate", f"{current_lr:.6f}")
        rprint(Panel(summary, title="Training Complete", border_style="green"))
        
        return train_losses, val_losses, train_metrics_list, val_metrics_list, best_val_metric  # Updated return values

    def train_epoch(self, train_loader):
        """Trains the model for one epoch with memory optimizations"""
        self.model.train()
        total_loss = 0
        self.metrics.reset()
        
        # Use the safer profiling context
        with self.memory_manager.start_memory_profiling():
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Move data to device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y) / self.memory_manager.grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if gradients should not be accumulated
                if not self.memory_manager.should_accumulate_gradients(batch_idx):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update metrics with logits instead of class predictions
                self.metrics.update(outputs, batch_y)
                
                # Update metrics
                total_loss += loss.item() * self.memory_manager.grad_accum_steps
                
                # Periodically log memory stats
                if batch_idx % 100 == 0:
                    self.memory_manager.log_memory_stats()
        
        epoch_loss = total_loss / len(train_loader)
        metrics = self.metrics.compute_all()
        
        return epoch_loss, metrics

    def evaluate(self, val_loader):
        """Evaluates the model on validation data."""
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                self.metrics.update(outputs, batch_y)
        
        metrics = self.metrics.compute_all()
        return total_loss / len(val_loader), metrics

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.best_trial_value = float('-inf')
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_params = None
        os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
        
        # Enhanced logging setup with config
        self.logger = setup_logger('HyperparameterTuner', config)
        
        # Initialize CPU optimization
        self.cpu_optimizer = CPUOptimizer(config, self.logger)
        self.optimizer_types = ['Adam', 'SGD']  # Add supported optimizers
        self.checkpoint_manager = CheckpointManager(config, self.logger)
        
    def save_best_model(self, model, optimizer, trial_value, params):
        """Save the best model and its metadata."""
        try:
            # Use CheckpointManager for atomic saving
            checkpoint_info = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=-1,  # Special epoch value for tuning
                metric_value=trial_value,
                params={
                    'optimizer_type': params['optimizer_type'],
                    'hidden_layers': params['hidden_layers'],
                    'dropout_rate': params['dropout_rate'],
                    'use_batch_norm': params['use_batch_norm'],
                    'weight_decay': params['weight_decay'],
                    'learning_rate': params['learning_rate'],
                    **{k: v for k, v in params.items() if k not in 
                       ['optimizer_type', 'hidden_layers', 'dropout_rate', 'use_batch_norm', 'weight_decay', 'learning_rate']}
                }
            )
            self.logger.info(f"Saved best model with metric value: {trial_value:.4f}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")
            return False

    def create_model_and_optimizer(self, trial, train_loader=None):
        """
        Create model, optimizer and scheduler
        Args:
            trial: Optuna trial object
            train_loader: DataLoader for training data, needed for scheduler setup
        """
        # Alternate between optimizers based on trial number
        optimizer_name = self.optimizer_types[trial.number % len(self.optimizer_types)]
        
        # Get model architecture parameters
        hidden_layers = []
        n_layers = trial.suggest_int('n_layers', 1, 4)
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'hidden_layer_{i}', 32, 512))
        
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
        
        # Get optimizer parameters based on type
        if optimizer_name == 'Adam':
            lr = float(self.config['training']['optimizer_params']['Adam'].get('base_lr', 1e-4))
            betas = tuple(self.config['training']['optimizer_params']['Adam'].get('betas', (0.9, 0.999)))
            eps = float(self.config['training']['optimizer_params']['Adam'].get('eps', 1e-8))
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        else:  # SGD
            lr = float(self.config['training']['optimizer_params']['SGD'].get('lr', 0.01))
            momentum = float(self.config['training']['optimizer_params']['SGD'].get('momentum', 0.9))
            
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        
        # Record trial parameters
        trial_params = {
            'optimizer_type': optimizer_name,
            'n_layers': n_layers,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'weight_decay': weight_decay,
            'learning_rate': lr
        }
        
        # Add optimizer-specific parameters
        if optimizer_name == 'Adam':
            trial_params.update({
                'betas': betas,
                'eps': eps
            })
        else:  # SGD
            trial_params.update({
                'momentum': momentum
            })
        
        # Create scheduler if needed
        scheduler = None
        if train_loader is not None and self.config['training'].get('scheduler', {}).get('type'):
            scheduler = self._create_scheduler(optimizer, train_loader)
        
        return model, optimizer, scheduler, trial_params

    def _create_scheduler(self, optimizer, train_loader):
        """Helper method to create learning rate scheduler"""
        total_steps = len(train_loader) * self.config['training']['epochs']
        scheduler_config = self.config['training'].get('scheduler', {})
        
        if scheduler_config.get('type') == 'OneCycleLR':
            try:
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=optimizer.param_groups[0]['lr'] * float(scheduler_config['params']['max_lr_factor']),
                    total_steps=total_steps,
                    div_factor=float(scheduler_config['params']['div_factor']),
                    final_div_factor=float(scheduler_config['params']['final_div_factor']),
                    pct_start=float(scheduler_config['params']['pct_start']),
                    anneal_strategy=scheduler_config['params']['anneal_strategy']
                )
                self.logger.info("Created OneCycleLR scheduler")
                return scheduler
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Failed to create OneCycleLR scheduler: {e}")
                return None
        return None

    # Update objective method to use scheduler
    def objective(self, trial, train_loader, val_loader):
        model, optimizer, scheduler, trial_params = self.create_model_and_optimizer(
            trial, train_loader=train_loader
        )
        criterion = getattr(nn, self.config['training']['loss_function'])()
        
        # Enhanced trial start logging with detailed optimizer info
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Starting Trial {trial.number}")
        
        # Log optimizer details
        optimizer_type = trial_params['optimizer_type']
        self.logger.info("\nOptimizer Configuration:")
        self.logger.info(f"  Type: {optimizer_type}")
        self.logger.info(f"  Learning Rate: {trial_params['learning_rate']}")
        self.logger.info(f"  Weight Decay: {trial_params['weight_decay']}")
        
        if optimizer_type == 'Adam':
            self.logger.info(f"  Betas: {trial_params['betas']}")
            self.logger.info(f"  Epsilon: {trial_params['eps']}")
        else:  # SGD
            self.logger.info(f"  Momentum: {trial_params['momentum']}")
        
        # Log model architecture
        self.logger.info("\nModel Architecture:")
        self.logger.info(f"  Number of Layers: {trial_params['n_layers']}")
        self.logger.info(f"  Hidden Layers: {trial_params['hidden_layers']}")
        self.logger.info(f"  Dropout Rate: {trial_params['dropout_rate']}")
        self.logger.info(f"  Batch Normalization: {trial_params['use_batch_norm']}")
        
        self.logger.info("-"*50)
        
        # Rest of the existing objective method code...
        trainer = PyTorchTrainer(
            model, criterion, optimizer,
            config=self.config,
            device=self.config['training']['device'],
            verbose=False  # Set to True for more detailed training logs
        )
        
        patience = self.config['optimization']['early_stopping']['patience']
        min_delta = self.config['optimization']['early_stopping']['min_delta']
        best_metric = float('-inf')
        patience_counter = 0
        last_metric = float('-inf')
        bn_issues_counter = 0
        max_bn_issues = 3  # Maximum allowed consecutive BatchNorm issues
        
        self.logger.info(f"\nStarting trial {trial.number}")
        self.logger.info(f"Parameters: {trial_params}")
        
        for epoch in range(self.config['training']['epochs']):
            trainer.train_epoch(train_loader)
            _, metrics = trainer.evaluate(val_loader)
            
            # Check BatchNorm stability if model uses it
            if hasattr(model, 'bn_manager'):
                bn_issues = model.bn_manager.validate_state(early_warning=True)
                if bn_issues:
                    bn_issues_counter += 1
                    self.logger.warning(f"Trial {trial.number}, Epoch {epoch}: BatchNorm issues detected")
                    if bn_issues_counter >= max_bn_issues:
                        self.logger.warning(f"Trial {trial.number} pruned due to persistent BatchNorm issues")
                        raise optuna.TrialPruned("Persistent BatchNorm instability")
                else:
                    bn_issues_counter = 0
            
            metric = metrics[self.config['training']['optimization_metric']]
            trial.report(metric, epoch)
            
            # Enhanced early stopping logic with detailed logging
            if metric > best_metric + min_delta:
                improvement = metric - best_metric if best_metric != float('-inf') else metric
                self.logger.info(f"Epoch {epoch}: Metric improved by {improvement:.4f}")
                best_metric = metric
                patience_counter = 0
                
                if metric > self.best_trial_value:
                    self.best_trial_value = metric
                    self.logger.info(f"New best trial metric: {metric:.4f}")
                    self.save_best_model(model, optimizer, metric, trial_params)
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{patience}")
            
            # Separate pruning logic with logging
            if metric < last_metric - 0.1:
                self.logger.info(f"Trial {trial.number} pruned due to metric deterioration")
                self.logger.info(f"Current: {metric:.4f}, Previous: {last_metric:.4f}")
                raise optuna.TrialPruned("Trial pruned due to metric deterioration")
            
            last_metric = metric
            
            # Early stopping check with logging
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"Best metric achieved: {best_metric:.4f}")
                break
            
            # Log iteration progress
            if epoch % 5 == 0 or epoch == self.config['training']['epochs'] - 1:
                self.logger.debug(
                    f"Epoch {epoch + 1}/{self.config['training']['epochs']}: "
                    f"Metric = {metric:.4f}, Best = {best_metric:.4f}"
                )
        
        self.logger.info(f"Trial {trial.number} finished. Final metric: {best_metric:.4f}\n")
        return best_metric

    def tune(self, train_loader, val_loader):
        """Run hyperparameter tuning with enhanced logging"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting Hyperparameter Tuning")
        self.logger.info(f"Number of trials: {self.config['optimization']['n_trials']}")
        self.logger.info(f"Early stopping patience: {self.config['optimization']['early_stopping']['patience']}")
        self.logger.info(f"Early stopping min delta: {self.config['optimization']['early_stopping']['min_delta']}")
        self.logger.info("="*50 + "\n")
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        try:
            study.optimize(
                lambda trial: self.objective(trial, train_loader, val_loader),
                n_trials=self.config['optimization']['n_trials']
            )
            
            # Check if any checkpoints were created
            checkpoint_files = list(CHECKPOINT_DIR.glob('checkpoint_v*'))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found after tuning")
            
            # Load best model to verify it's valid
            result = self.checkpoint_manager.load_checkpoint()
            if result is None:
                raise ValueError("Failed to load best model after tuning")
            
            self.logger.info("\n" + "="*50)
            self.logger.info("Tuning Completed Successfully!")
            self.logger.info(f"Best trial value: {study.best_trial.value:.4f}")
            self.logger.info("Best parameters:")
            for key, value in study.best_params.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("="*50 + "\n")
            
            return study.best_trial, study.best_params
            
        except Exception as e:
            self.logger.error(f"Error during tuning: {e}")
            raise

    # ...rest of existing code...

def restore_best_model(config):
    """Restore best model using CheckpointManager"""
    logger = logging.getLogger('MLPTrainer')
    checkpoint_manager = CheckpointManager(config, logger)
    
    try:
        result = checkpoint_manager.load_checkpoint()
        if result is None:
            return None
            
        checkpoint_meta, model = result
        
        # Create optimizer
        optimizer_type = checkpoint_meta['hyperparameters']['optimizer_type']
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=checkpoint_meta['hyperparameters']['learning_rate'],
                betas=checkpoint_meta['hyperparameters'].get('betas', (0.9, 0.999)),
                eps=checkpoint_meta['hyperparameters'].get('eps', 1e-8),
                weight_decay=checkpoint_meta['hyperparameters'].get('weight_decay', 0.0)
            )
        else:  # SGD
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=checkpoint_meta['hyperparameters']['learning_rate'],
                momentum=checkpoint_meta['hyperparameters'].get('momentum', 0.9),
                weight_decay=checkpoint_meta['hyperparameters'].get('weight_decay', 0.0)
            )
        
        optimizer.load_state_dict(checkpoint_meta['optimizer_state_dict'])
        
        return {
            'model': model,
            'optimizer': optimizer,
            'optimizer_type': optimizer_type,
            'metric_value': checkpoint_meta['metric_value'],
            'hyperparameters': checkpoint_meta['hyperparameters']
        }
        
    except Exception as e:
        logger.error(f"Error restoring model from checkpoint: {e}")
        return None

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create best_model section if it doesn't exist
    if 'best_model' not in config:
        config['best_model'] = {}
    
    # Format parameters for config
    hidden_layers = [best_params[f'hidden_layer_{i}'] for i in range(best_params['n_layers'])]
    
    # Get learning rate from config using the correct parameter name
    learning_rate = best_params.get('lr', config['training']['optimizer_params']['Adam']['lr'])
    
    config['best_model'].update({
        'hidden_layers': hidden_layers,
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': learning_rate,
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
        config=config,
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
    os.environ['PYTH longer ONHASHSEED'] = str(seed)

def log_cpu_optimizations(config, logger):
    """Log CPU optimization settings at startup"""
    if logger.isEnabledFor(logging.INFO):  # Only log if INFO is enabled
        logger.info("CPU Optimization Status:")
        logger.debug(f"CPU Architecture: {platform.processor()}")
        logger.debug(f"PyTorch Version: {torch.__version__}")
        logger.debug(f"Number of CPU cores: {multiprocessing.cpu_count()}")
        
        enabled_features = []
        # ...existing feature detection code...
        
        # Consolidate logging into fewer messages
        if enabled_features:
            logger.info("Enabled optimizations: " + ", ".join(enabled_features))

        # Move DataLoader settings to debug
        logger.debug("DataLoader settings:")
        logger.debug(f"Workers: {config['training']['dataloader'].get('num_workers', 'auto')}")
        logger.debug(f"Memory settings: pin_memory={config['training']['dataloader'].get('pin_memory', False)}, "
                    f"persistent_workers={config['training']['dataloader'].get('persistent_workers', False)}")

class ModelValidator:
    """Handles validation and testing of trained models"""
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('ModelValidator')
        self.validation_results = {}
        # Get figures directory from config or use default
        self.figures_dir = config.get('logging', {}).get('figures_dir', 'figures')
        # Create figures directory if it doesn't exist
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def validate_model(self, model, val_loader, device='cpu'):
        """Comprehensive model validation"""
        model.eval()
        self.validation_results = {
            'confidence_metrics': self._analyze_confidence(model, val_loader, device),
            'confusion_matrix': self._generate_confusion_matrix(model, val_loader, device),
            'classification_report': self._generate_classification_report(model, val_loader, device)
        }
        
        self._log_validation_results()
        return self.validation_results
    
    def _analyze_confidence(self, model, dataloader, device):
        """Analyze prediction confidences"""
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = torch.softmax(model(inputs), dim=1)
                all_probs.append(outputs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        probs = np.vstack(all_probs)
        labels = np.concatenate(all_labels)
        preds = np.argmax(probs, axis=1)
        
        confidences = np.max(probs, axis=1)
        correct_mask = preds == labels
        
        return {
            'avg_confidence_correct': confidences[correct_mask].mean(),
            'avg_confidence_incorrect': confidences[~correct_mask].mean(),
            'high_confidence_errors': np.sum((confidences > 0.9) & ~correct_mask),
            'low_confidence_correct': np.sum((confidences < 0.6) & correct_mask)
        }
    
    def _generate_confusion_matrix(self, model, dataloader, device):
        """Generate and plot confusion matrix"""
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'))
        plt.close()
        
        return cm
    
    def _generate_classification_report(self, model, dataloader, device):
        """Generate detailed classification report"""
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        return classification_report(y_true, y_pred, output_dict=True)
    
    def _log_validation_results(self):
        """Log validation results with reduced verbosity"""
        self.logger.info("\nValidation Summary:")
        
        # Consolidate metrics into a single message
        metrics = self.validation_results['confidence_metrics']
        self.logger.info(
            f"Confidence Metrics - "
            f"Correct: {metrics['avg_confidence_correct']:.3f}, "
            f"Incorrect: {metrics['avg_confidence_incorrect']:.3f}, "
            f"High Conf. Errors: {metrics['high_confidence_errors']}"
        )
        
        # Move detailed class metrics to debug level
        report = self.validation_results['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                self.logger.debug(f"Class {class_name}: "
                                f"F1={metrics['f1-score']:.3f}, "
                                f"Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}")
    
    def cross_validate(self, model_class, train_dataset, n_splits=5, **model_params):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
            self.logger.info(f"\nFold {fold+1}/{n_splits}")
            
            # Create train/val splits
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            # Create dataloaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=train_subsampler,
                num_workers=min(
                    self.config['training']['dataloader']['num_workers'],
                    multiprocessing.cpu_count()
                ),
                pin_memory=self.config['training']['dataloader']['pin_memory']
            )
            val_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=val_subsampler,
                num_workers=min(
                    self.config['training']['dataloader']['num_workers'],
                    multiprocessing.cpu_count()
                ),
                pin_memory=self.config['training']['dataloader']['pin_memory']
            )
            
            # Create and train model
            model = model_class(**model_params)
            
            # Get optimizer parameters
            optimizer_name = self.config['training']['optimizer_choice']
            optimizer_params = self.config['training']['optimizer_params'][optimizer_name].copy()
            
            # Handle special cases for optimizer parameters
            if optimizer_name == 'Adam':
                # Remove 'base_lr' and use it as 'lr'
                if 'base_lr' in optimizer_params:
                    optimizer_params['lr'] = optimizer_params.pop('base_lr')
            
            # Create optimizer with correct parameters
            optimizer = getattr(torch.optim, optimizer_name)(
                model.parameters(),
                **optimizer_params
            )
            
            criterion = getattr(nn, self.config['training']['loss_function'])()
            
            trainer = PyTorchTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                config=self.config,
                device=self.config['training']['device'],
                verbose=False
            )
            
            # Train for specified number of epochs
            _, _, _, _, fold_score = trainer.train(
                train_loader,
                val_loader,
                self.config['training']['epochs'],
                metric=self.config['training']['optimization_metric']
            )
            
            scores.append(fold_score)
            self.logger.info(f"Fold {fold+1} Score: {fold_score:.4f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.logger.info(f"\nCross-validation complete:")
        self.logger.info(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': scores
        }

class AdaptiveOptimizer:
    """Manages adaptive optimization strategies with population-based training insights"""
    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        self.history: List[Tuple[float, float]] = []  # (lr, metric) pairs
        # Remove verbose parameter here
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        self.warmup_scheduler = None
        self.cycle_scheduler = None
        self.current_strategy = 'warmup'
        
        # Get warmup steps from config, with fallback values
        self.warmup_steps = (
            self.config.get('best_model', {}).get('warmup_steps', 0) or
            self.config.get('optimization', {}).get('warmup', {}).get('max_steps', 0)
        )
        
        if self.config.get('optimization', {}).get('warmup', {}).get('enabled', False):
            self._setup_warmup()
        self._setup_cycle()
    
    def _setup_warmup(self):
        """Initialize warmup phase"""
        if self.warmup_steps > 0:
            self.warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
    
    def _setup_cycle(self):
        """Initialize cycling phase with dynamic parameters"""
        base_lr = self.optimizer.param_groups[0]['lr']
        try:
            max_lr_factor = self.config['training']['scheduler']['params']['max_lr_factor']
        except KeyError:
            max_lr_factor = 10.0  # default value
            
        total_steps = self.config.get('training', {}).get('epochs', 100)
        
        self.cycle_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=base_lr * max_lr_factor,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0
        )
    
    def step(self, metric: float, epoch: int):
        """Adaptive stepping based on metric performance"""
        # Get current LR before step
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.history.append((current_lr, metric))
        
        # Apply strategy
        if self.current_strategy == 'warmup' and epoch < self.warmup_steps:
            self.warmup_scheduler.step()
            new_lr = self.warmup_scheduler.get_last_lr()[0]
        elif self.current_strategy == 'cycle':
            self.cycle_scheduler.step()
            new_lr = self.cycle_scheduler.get_last_lr()[0]
        else:  # plateau
            old_lr = current_lr  # Store old LR before step
            self.plateau_scheduler.step(metric)
            new_lr = self.optimizer.param_groups[0]['lr']
            # Only log if plateau scheduler changed the LR
            if new_lr != old_lr:
                logging.getLogger("PyTorchTrainer").info(
                    f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}"
                )

    def _calculate_improvement(self) -> float:
        """Calculate recent improvement rate"""
        recent = self.history[-3:]
        improvements = [curr[1] - prev[1] for prev, curr in zip(recent, recent[1:])]
        return sum(improvements) / len(improvements)
    
    def _adjust_strategy(self):
        """Adjust optimization strategy based on performance"""
        if self.current_strategy == 'cycle':
            # Switch to plateau detection if cycling isn't helping
            self.current_strategy = 'plateau'
            # Reset optimizer to best learning rate from history
            best_lr = max(self.history, key=lambda x: x[1])[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = best_lr
        elif self.current_strategy == 'plateau':
            # If we've reduced LR significantly, try cycling again
            if self.optimizer.param_groups[0]['lr'] < self.history[0][0] / 10:
                self.current_strategy = 'cycle'
                self._setup_cycle()

class MetricsManager:
    """Manages multiple torchmetrics for multi-class classification"""
    def __init__(self, config: Dict[str, Any], num_classes: int, device: str = 'cpu'):
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.logger = logging.getLogger('MetricsManager')
        self.metrics: Dict[str, torchmetrics.Metric] = {}
        self._initialize_metrics()
        
    def _initialize_metrics(self) -> None:
        """Initialize metrics based on config"""
        available_metrics = self.config['training']['metrics']['available']
        tracked_metrics = self.config['training']['metrics']['tracked']
        
        for metric_key in tracked_metrics:
            if metric_key not in available_metrics:
                self.logger.warning(f"Metric {metric_key} not found in available metrics")
                continue
                
            metric_config = available_metrics[metric_key]
            metric_class = getattr(torchmetrics.classification, metric_config['name'])
            params = metric_config.get('params', {})
            
            # Add num_classes to params if the metric class accepts it
            if 'num_classes' in inspect.signature(metric_class).parameters:
                params['num_classes'] = self.num_classes
            
            try:
                metric = metric_class(**params).to(self.device)
                self.metrics[metric_key] = metric
                self.logger.debug(f"Initialized metric: {metric_key}")
            except Exception as e:
                self.logger.error(f"Failed to initialize metric {metric_key}: {e}")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics with predictions and targets"""
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute_all(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute and return all metrics"""
        results = {}
        for name, metric in self.metrics.items():
            try:
                value = metric.compute()
                results[name] = value.item() if torch.is_tensor(value) else value
            except Exception as e:
                self.logger.error(f"Error computing metric {name}: {e}")
                results[name] = float('nan')
        return results

    def get_primary_metric(self) -> float:
        """Get the primary metric value"""
        primary = self.config['training']['metrics']['primary']
        if primary not in self.metrics:
            raise ValueError(f"Primary metric {primary} not found in tracked metrics")
        return self.metrics[primary].compute().item()

    def reset(self) -> None:
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Set up logging with config
    logger = setup_logger('MLPTrainer', config)
    
    # Log CPU optimization settings at startup
    log_cpu_optimizations(config, logger)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create datasets and dataloaders
    train_df = pd.read_csv(config['data']['train_path'])
    val_df = pd.read_csv(config['data']['val_path'])
    train_dataset = CustomDataset(train_df, config['data']['target_column'])
    val_dataset = CustomDataset(val_df, config['data']['target_column'])
    
    # Use correct path to performance settings
    batch_size = config['training']['batch_size']
    if 'training' in config and 'performance' in config['training']:
        batch_size *= config['training']['performance'].get('batch_size_multiplier', 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(
            config['training']['dataloader']['num_workers'],
            multiprocessing.cpu_count()
        ),
        pin_memory=config['training']['dataloader']['pin_memory'],
        prefetch_factor=config['training']['dataloader']['prefetch_factor'],
        persistent_workers=config['training']['dataloader']['persistent_workers'],
        drop_last=config['training'].get('drop_last', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(
            config['training']['dataloader']['num_workers'],
            multiprocessing.cpu_count()
        ),
        pin_memory=config['training']['dataloader']['pin_memory'],
        prefetch_factor=config['training']['dataloader']['prefetch_factor'],
        persistent_workers=config['training']['dataloader']['persistent_workers']
    )
    
    # Ensure checkpoint directory exists
    CHECKPOINT_DIR.mkdir(exist_ok=True)  # Changed from config['model']['save_path']
    
    need_training = False
    # Try to restore model first
    restored = restore_best_model(config)
    
    if restored is None:
        logger.info("No valid checkpoint found. Starting hyperparameter tuning...")
        need_training = True
    elif 'best_model' not in config:
        logger.info("No best model configuration found. Will retrain with restored model as starting point...")
        need_training = True
    
    if need_training:
        # Run hyperparameter tuning
        tuner = HyperparameterTuner(config)
        best_trial, best_params = tuner.tune(train_loader, val_loader)
        save_best_params_to_config(config_path, best_trial, best_params)
        # Reload config and restore the newly trained model
        config = load_config(config_path)
        restored = restore_best_model(config)
        
        if restored is None:
            logger.error("Failed to create model even after tuning. Exiting.")
            sys.exit(1)
    
    # Log model parameters
    if 'best_model' in config:
        logger.info("\nBest model parameters from config:")
        for key, value in config['best_model'].items():
            logger.info(f"    {key}: {value}")
    else:
        logger.info("\nUsing restored model parameters:")
        for key, value in restored['hyperparameters'].items():
            logger.info(f"    {key}: {value}")
    
    model = restored['model']
    optimizer = restored['optimizer']
    
    # Create criterion for evaluation
    criterion = getattr(nn, config['training']['loss_function'])()
    
    # Create trainer for evaluation
    trainer = PyTorchTrainer(
        model, criterion, optimizer,
        config=config,
        device=config['training']['device'],
        verbose=True
    )
    
    # Initialize memory optimization with sample data
    sample_X, _ = next(iter(train_loader))
    sample_input = sample_X[0]
    
    optimal_batch_size = trainer.memory_manager.optimize_memory(model, sample_input)
    
    # Update batch size if needed
    if (optimal_batch_size != batch_size):
        train_loader = DataLoader(
            train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            num_workers=min(
                config['training']['dataloader']['num_workers'],
                multiprocessing.cpu_count()
            ),
            pin_memory=config['training']['dataloader']['pin_memory'],
            prefetch_factor=config['training']['dataloader']['prefetch_factor'],
            persistent_workers=config['training']['dataloader']['persistent_workers'],
            drop_last=config['training'].get('drop_last', False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=optimal_batch_size,
            shuffle=False,
            num_workers=min(
                config['training']['dataloader']['num_workers'],
                multiprocessing.cpu_count()
            ),
            pin_memory=config['training']['dataloader']['pin_memory'],
            prefetch_factor=config['training']['dataloader']['prefetch_factor'],
            persistent_workers=config['training']['dataloader']['persistent_workers']
        )
    
    # Validate the best model
    logger.info("\nRunning comprehensive model validation...")
    validator = ModelValidator(config, logger)
    print("\n" + "="*50)
    validation_results = validator.validate_model(
        model=restored['model'],
        val_loader=val_loader,
        device=config['training']['device']
    )
    
    # Evaluate restored model
    print("\nEvaluating restored model on validation set...")
    val_loss, val_metrics = trainer.evaluate(val_loader)
    
    metric_name = config['training']['optimization_metric']
    metric_value = val_metrics[metric_name]
    
    print(f"\nRestored model performance:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")  # Changed from 'f1' to 'f1_score'
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")
    print(f"Current {metric_name.upper()}: {metric_value:.4f}")
    
    # First run standard evaluation
    print("\nStandard Model Evaluation:")
    val_loss, val_metrics = trainer.evaluate(val_loader)
    
    print(f"\nBasic Performance Metrics:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")
    print(f"Current {metric_name.upper()}: {metric_value:.4f}")
    
    # Now run comprehensive validation
    print("\n" + "="*50)
    print("Running Comprehensive Model Analysis")
    print("="*50)
    
    validator = ModelValidator(config, logger)
    validation_results = validator.validate_model(
        model=restored['model'],
        val_loader=val_loader,
        device=config['training']['device']
    )
    
    # Print detailed validation results
    print("\nConfidence Analysis:")
    conf_metrics = validation_results['confidence_metrics']
    print(f"Average Confidence (Correct Predictions): {conf_metrics['avg_confidence_correct']:.4f}")
    print(f"Average Confidence (Incorrect Predictions): {conf_metrics['avg_confidence_incorrect']:.4f}")
    print(f"High Confidence Errors (>90%): {conf_metrics['high_confidence_errors']}")
    print(f"Low Confidence Correct (<60%): {conf_metrics['low_confidence_correct']}")
    
    print("\nClassification Report:")
    report = validation_results['classification_report']
    # Print per-class metrics
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"\nClass {class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    print("\nConfusion Matrix has been saved to 'confusion_matrix.png'")
    print("="*50)

    print("Performing Cross-Validation")
    print("="*50)
    
    # Create a fresh model with the best parameters for cross-validation
    cv_model_params = {
        'input_size': config['model']['input_size'],
        'hidden_layers': restored['hyperparameters']['hidden_layers'],
        'num_classes': config['model']['num_classes'],
        'dropout_rate': restored['hyperparameters']['dropout_rate'],
        'use_batch_norm': restored['hyperparameters']['use_batch_norm']
    }
    
    # Combine training and validation sets for cross-validation
    full_dataset = CustomDataset(
        pd.concat([train_df, val_df], axis=0).reset_index(drop=True),
        config['data']['target_column']
    )
    
    cv_results = validator.cross_validate(
        model_class=MLPClassifier,
        train_dataset=full_dataset,
        n_splits=config['training']['validation']['cross_validation']['n_splits'],
        **cv_model_params
    )
    
    print("\nCross-Validation Results:")
    print(f"Mean Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    print("Individual Fold Scores:")
    for i, score in enumerate(cv_results['scores'], 1):
        print(f"  Fold {i}: {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()