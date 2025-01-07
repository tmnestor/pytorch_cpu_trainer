import os
import logging

def setup_logger(config, name='MLPTrainer'):
    """Set up logging with both file and console handlers."""
    # Create logging directory if it doesn't exist
    log_dir = config['logging']['directory']
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the logger - use lowercase name for consistency
    logger_name = name.lower()
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler - use component specific config if available
    if logger_name in config['logging']['handlers']:
        handler_config = config['logging']['handlers'][logger_name]
        log_path = os.path.join(log_dir, handler_config['filename'])
        fh = logging.FileHandler(log_path)
        fh.setLevel(getattr(logging, handler_config['level']))
    else:
        # Default file handler - always use lowercase for filename
        log_path = os.path.join(log_dir, f'{logger_name}.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(getattr(logging, config['logging']['file_level']))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config['logging']['console_level']))
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_path(config, path_key):
    """
    Resolves a path from config using the root directory.
    
    Args:
        config: The configuration dictionary
        path_key: String like 'model.history_db' or 'logging.directory'
    
    Returns:
        Absolute path with root directory prefixed
    """
    # Get the template path
    keys = path_key.split('.')
    template = config
    for key in keys:
        template = template[key]
    
    # Get root and subdirs
    root_path = config['paths']['root']
    subdirs = config['paths']['subdirs']
    
    # First create the real subdirectories if they don't exist
    for subdir_name, subdir_path in subdirs.items():
        full_subdir_path = os.path.join(root_path, subdir_path)
        os.makedirs(full_subdir_path, exist_ok=True)
    
    # Now replace placeholders with actual paths
    for subdir_name, subdir_path in subdirs.items():
        template = template.replace(f'{{{subdir_name}}}', subdir_path)
    
    return os.path.join(root_path, template)

def ensure_path_exists(path, is_file=True):
    """Create directory structure for a path."""
    directory = os.path.dirname(path) if is_file else path
    if directory:
        os.makedirs(directory, exist_ok=True)
