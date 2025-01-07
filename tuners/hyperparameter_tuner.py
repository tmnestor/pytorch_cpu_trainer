import os
import torch
import torch.nn as nn
import optuna
from ..models import MLPClassifier  # Update import path
from ..trainers.pytorch_trainer import PyTorchTrainer
from ..utils import setup_logger
from ..model_history import ModelHistory

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.best_trial_value = float('-inf')
        self.best_model_state = None 
        self.best_optimizer_state = None
        self.best_params = None
        self.logger = setup_logger(config, 'hyperparameter_tuning')
        os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
        
        if 'config_path' not in config:
            raise ValueError("config_path must be set in config dictionary")
            
        self.config_path = config['config_path']
        self.history = ModelHistory(self.config_path)

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
        """Create model and optimizer with trial parameters."""
        # Extract hyperparameters from trial
        n_layers = trial.suggest_int('n_layers', 1, 4)
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
        
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.config['training']['device'],
            config=self.config
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
                
                # Pruning checks
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
        """Run hyperparameter optimization."""
        self.logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader),
            n_trials=self.config['optimization']['n_trials']
        )
        
        # Save the best trial to database
        self.logger.info(f"Best trial value: {study.best_trial.value:.4f}")
        self.history.save_experiment(
            self.config_path,
            study.best_trial.value,
            self.config['training']['optimization_metric']
        )
        
        return study.best_trial, study.best_params
