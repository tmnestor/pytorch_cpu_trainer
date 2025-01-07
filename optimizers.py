import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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
