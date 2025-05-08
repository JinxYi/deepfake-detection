
import os

def make_directory(base_dir, new_dir):
    """Creates a directory given a base directory. No error is raised if the directory already exists."""
    new_path = os.path.join(base_dir, new_dir)
    os.makedirs(new_path, exist_ok=True)
    return new_path
