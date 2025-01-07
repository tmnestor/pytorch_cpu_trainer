import os

def get_path(config, path_key):
    """
    Resolves a path from config using the root directory and replaces placeholders.
    
    Args:
        config: The configuration dictionary
        path_key: String like 'model.history_db' or 'logging.directory'
    
    Returns:
        Absolute path with resolved placeholders
    """
    # Get the path template
    keys = path_key.split('.')
    template = config
    for key in keys:
        template = template[key]
        
    # Replace all placeholders
    root_path = config['paths']['root']
    subdirs = config['paths']['subdirs']
    
    # Replace each placeholder with its actual path
    for subdir_name, subdir_path in subdirs.items():
        placeholder = f'{{{subdir_name}}}'
        if placeholder in template:
            template = template.replace(placeholder, subdir_path)
    
    # Join with root path
    return os.path.join(root_path, template)

def ensure_path_exists(path, is_file=True):
    """Create directory structure for a path."""
    directory = os.path.dirname(path) if is_file else path
    if directory:
        os.makedirs(directory, exist_ok=True)
