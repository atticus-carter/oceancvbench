"""Common utilities for the oceancvbench package."""

import os
import sys
import logging
from typing import Optional, Union, List, Dict, Any, Tuple

logger = logging.getLogger('oceancvbench.utils')

# Global debug flag
_DEBUG_MODE = False

def set_debug_mode(enable: bool) -> None:
    """
    Set the global debug mode.
    
    Args:
        enable: True to enable debug mode, False to disable
    """
    global _DEBUG_MODE
    _DEBUG_MODE = enable
    
    # Configure logging level based on debug mode
    if enable:
        logging.getLogger('oceancvbench').setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.getLogger('oceancvbench').setLevel(logging.INFO)

def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        True if debug mode is enabled, False otherwise
    """
    return _DEBUG_MODE

def get_file_extensions(file_type: str) -> List[str]:
    """
    Get common file extensions for a given file type.
    
    Args:
        file_type: Type of file ('image', 'video', 'annotation', etc.)
        
    Returns:
        List of file extensions
    """
    extension_map = {
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        'annotation': ['.txt', '.xml', '.json', '.yaml', '.yml'],
        'model': ['.pt', '.pth', '.weights', '.onnx', '.tflite', '.h5']
    }
    
    return extension_map.get(file_type.lower(), [])
