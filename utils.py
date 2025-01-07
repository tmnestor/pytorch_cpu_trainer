import os

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
