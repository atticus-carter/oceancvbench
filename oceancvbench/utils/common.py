"""Common utility functions for OceanCVBench."""

import os
import logging
from typing import Optional, Union, Dict, List, Any
from pathlib import Path

# Global debug mode flag
_DEBUG_MODE = False

def set_debug_mode(enabled: bool) -> None:
    """
    Set the debug mode for the application.
    
    Args:
        enabled: True to enable debug mode, False to disable
    """
    global _DEBUG_MODE
    _DEBUG_MODE = enabled
    
    # Configure logging level based on debug mode
    if enabled:
        logging.getLogger('oceancvbench').setLevel(logging.DEBUG)
    else:
        logging.getLogger('oceancvbench').setLevel(logging.INFO)


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        True if debug mode is enabled, otherwise False
    """
    return _DEBUG_MODE


def ensure_dir(path: Union[str, Path]) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create if it doesn't exist
        
    Returns:
        The absolute path to the directory
    """
    dir_path = os.path.abspath(path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_package_root() -> str:
    """
    Get the root directory of the package.
    
    Returns:
        Absolute path to the package root
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def is_valid_image(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid image file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a valid image, otherwise False
    """
    if not os.path.exists(file_path):
        return False
    
    # Check extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    if not any(str(file_path).lower().endswith(ext) for ext in valid_extensions):
        return False
    
    # Additional validation could be added here
    return True
