import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
from .utils import setup_logger  # Fix import to be relative

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

def restore_best_model(config):
    """Utility function to restore the best model and its optimizer."""
    logger = setup_logger(config, 'MLPTrainer')
    checkpoint_path = config['model']['save_path']
    
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model = MLPClassifier(
                input_size=config['model']['input_size'],
                hidden_layers=checkpoint['hyperparameters']['hidden_layers'],
                num_classes=config['model']['num_classes'],
                dropout_rate=checkpoint['hyperparameters']['dropout_rate'],
                use_batch_norm=checkpoint['hyperparameters']['use_batch_norm']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = getattr(torch.optim, checkpoint['optimizer_name'])(
                model.parameters(),
                lr=checkpoint['hyperparameters']['lr'],
                weight_decay=checkpoint['hyperparameters'].get('weight_decay', 0.0)
            )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return {
                'model': model,
                'optimizer': optimizer,
                'metric_name': checkpoint['metric_name'],
                'metric_value': checkpoint['metric_value'],
                'hyperparameters': checkpoint['hyperparameters']
            }
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}. Using default model.")
    
    # Use default model configuration
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
