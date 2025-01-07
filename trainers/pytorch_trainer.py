import gc
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.optim.swa_utils as swa_utils
import intel_extension_for_pytorch as ipex
from ..utils import setup_logger
from ..optimizers import CPUWarmupScheduler  # Update import path
import logging  # Add missing import

class PyTorchTrainer:
    """A generic PyTorch trainer class."""
    def __init__(self, model, criterion, optimizer, device='cpu', verbose=False, scheduler=None, warmup_scheduler=None, config=None):
        self.logger = setup_logger(config, 'pytorch_trainer') if config else logging.getLogger('pytorch_trainer')
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler  # Add explicit warmup scheduler
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
            model = model.to(memory_format=torch.channels_last)
            
        # Initialize mixed precision settings based on hardware support
        self.use_mixed_precision = hasattr(ipex, 'core') and ipex.core.onednn_has_bf16_support()
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
        
        # Initialize SWA if enabled
        swa_enabled = config.get('training', {}).get('swa', {}).get('enabled', False) if config else False
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
            
            # Update metrics
            total_loss += loss.item() * self.grad_accum_steps
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
            
            # Memory cleanup
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Calculate metrics
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
        """Plots the learning curves for loss and chosen metric."""
        os.makedirs('figures', exist_ok=True)
        plt.figure(figsize=(12, 5))
        
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
        
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'learning_curves.png'))
        plt.close()

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model for specified number of epochs."""
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        best_val_metric = float('-inf')
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Handle warmup and SWA scheduling
            if hasattr(self, 'warmup_scheduler') and self.warmup_scheduler and epoch < self.warmup_scheduler.warmup_steps:
                self.warmup_scheduler.step()
            elif hasattr(self, 'swa_model') and self.swa_model is not None and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step() if hasattr(self, 'swa_scheduler') else None
            else:
                self.scheduler.step() if self.scheduler is not None else None
            
            # Training and evaluation
            train_loss, train_accuracy, train_f1 = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # Select appropriate metric
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
            
            gc.collect()
        
        # Plot final learning curves
        self.plot_learning_curves(
            train_losses, val_losses,
            train_metrics, val_metrics,
            metric_name='F1-Score' if metric == 'f1' else 'Accuracy'
        )
        
        return train_losses, val_losses, train_metrics, val_metrics, best_val_metric
