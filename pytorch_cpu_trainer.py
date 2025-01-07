import gc
import datetime
import random
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse  # Add missing import
from sklearn.metrics import f1_score  # Add missing import

# Use absolute imports for package
from pytorch_cpu_trainer.models import MLPClassifier, CustomDataset, LabelSmoothingLoss, restore_best_model
from pytorch_cpu_trainer.trainers import PyTorchTrainer
from pytorch_cpu_trainer.optimizers import create_optimizer, create_scheduler, create_warmup_scheduler, CPUOptimizer
from pytorch_cpu_trainer.tuners import HyperparameterTuner
from pytorch_cpu_trainer.model_history import ModelHistory, update_default_config
from pytorch_cpu_trainer.utils import get_path, ensure_path_exists, setup_logger

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'best_model' not in config:
        config['best_model'] = {}
    
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

# Remove the class definitions that were moved
# Keep only the utility functions and main()

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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
    
    # Load configuration and add config path
    config = load_config(args.config)
    config['config_path'] = args.config

    # Create all required directories first
    root_path = config['paths']['root']
    for subdir_name, subdir_path in config['paths']['subdirs'].items():
        full_path = os.path.join(root_path, subdir_path)
        os.makedirs(full_path, exist_ok=True)
    
    # Create necessary directories using resolved paths
    ensure_path_exists(get_path(config, 'model.save_path'))
    ensure_path_exists(get_path(config, 'logging.directory'))
    ensure_path_exists(get_path(config, 'paths.subdirs.data'))
    ensure_path_exists(get_path(config, 'paths.subdirs.figures'))
    
    # Update load paths
    train_path = get_path(config, 'data.train_path')
    val_path = get_path(config, 'data.val_path')
    
    # Set up logger AFTER adding config_path
    logger = setup_logger(config, 'MLPTrainer')
    logger.info(f"Starting in {args.mode} mode...")
    
    # Update config defaults
    logger.info("Checking for historical best configurations")
    update_default_config(args.config)
    
    # Reload config but preserve config_path
    config_path = config['config_path']  # Save config_path
    config = load_config(args.config)
    config['config_path'] = config_path  # Restore config_path after reload
    
    # Initialize CPU optimization early
    cpu_optimizer = CPUOptimizer(config)
    cpu_optimizer.configure_thread_settings()
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Load and validate data files
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
            tuner = HyperparameterTuner(config)  # Now has access to config path
            best_trial, best_params = tuner.tune(train_loader, val_loader)
            save_best_params_to_config(args.config, best_trial, best_params)
            config = load_config(args.config)
            
            # Remove this part since we now save during tuning
            # history = ModelHistory(args.config)
            # history.save_experiment(...)
        
        # Rest of the training code remains the same
        restored = restore_best_model(config)
        model = restored['model']
        
        # Use factory functions to create optimizer and schedulers
        optimizer = create_optimizer(model.parameters(), config)
        scheduler = create_scheduler(optimizer, train_loader, config)
        warmup_scheduler = create_warmup_scheduler(optimizer, config)
        
        # Create criterion with label smoothing if enabled
        if config['training'].get('label_smoothing', {}).get('enabled', False):
            criterion = LabelSmoothingLoss(
                num_classes=config['model']['num_classes'],
                smoothing=config['training']['label_smoothing']['factor']
            )
        else:
            criterion = getattr(nn, config['training']['loss_function'])()
            
        # Create trainer with new schedulers
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=config['training']['device'],
            verbose=True,
            scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,  # Add warmup scheduler
            config=config
        )
        
        # Train the model
        logger.info("\nFinal Model Performance:")
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