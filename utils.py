import os

def get_path(config, path_key):
    """
    Resolves a path from config using the root directory.
    
    Args:
        config: The configuration dictionary
        path_key: String like 'model.save_path' or 'logging.directory'
    
    Returns:
        Absolute path with root directory prefixed
    """
    # Get the path template
    keys = path_key.split('.')
    template = config
    for key in keys:
        template = template[key]
        
    # Replace directory placeholders
    for subdir, value in config['paths']['subdirs'].items():
        template = template.replace(f'{{{subdir}}}', value)
    
    # Join with root path
    return os.path.join(config['paths']['root'], template)

def ensure_path_exists(path, is_file=True):
    """Create directory structure for a path."""
    directory = os.path.dirname(path) if is_file else path
    if directory:
        os.makedirs(directory, exist_ok=True)
