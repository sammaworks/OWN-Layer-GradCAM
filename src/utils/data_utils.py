import os
from pathlib import Path

def list_images(directory):
    """Lists all image files in the given directory."""
    return [str(file) for file in Path(directory).rglob("*.jpg") + Path(directory).rglob("*.png")]

def create_directory(directory):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)