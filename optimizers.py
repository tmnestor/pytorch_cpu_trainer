import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import psutil
import cpuinfo
import intel_extension_for_pytorch as ipex
from .utils import setup_logger

class CPUOptimizer:
    """Handles CPU-specific optimizations for PyTorch training."""
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.logger = setup_logger(config, 'cpu_optimization')
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

def create_optimizer(model_params, config):
    """Create optimizer based on configuration."""
    optimizer_name = config['training']['optimizer_choice']
    optimizer_params = config['training']['optimizer_params'][optimizer_name]
    
    # Get optimizer class from torch.optim
    optimizer_class = getattr(optim, optimizer_name)
    
    return optimizer_class(model_params, **optimizer_params)

def create_scheduler(optimizer, train_loader, config):
    """Create learning rate scheduler based on configuration."""
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config['params'].copy()
    
    if scheduler_type == 'OneCycleLR':
        # Special handling for OneCycleLR
        max_lr = float(config['training']['optimizer_params']['Adam']['lr']) * \
                float(scheduler_params.pop('max_lr_factor', 10.0))
        
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=float(scheduler_params.get('pct_start', 0.3)),
            div_factor=float(scheduler_params.get('div_factor', 25.0)),
            final_div_factor=float(scheduler_params.get('final_div_factor', 1e4))
        )
    else:
        # Get scheduler class from torch.optim.lr_scheduler
        scheduler_class = getattr(lr_scheduler, scheduler_type)
        return scheduler_class(optimizer, **scheduler_params)

def create_warmup_scheduler(optimizer, config):
    """Create warmup scheduler if enabled in config."""
    if config['training'].get('warmup', {}).get('enabled', False):
        warmup_steps = config['training']['warmup'].get('max_steps', 1000)
        base_lr = config['training']['optimizer_params'][config['training']['optimizer_choice']]['lr']
        initial_lr = base_lr / 100  # Start with 1% of target learning rate
        
        return CPUWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            initial_lr=initial_lr,
            target_lr=base_lr
        )
    return None
