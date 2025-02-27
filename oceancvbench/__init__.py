"""
OceanCVBench: A benchmarking toolkit for marine computer vision models.

This package provides tools for evaluating computer vision models on marine imagery,
with a focus on standardized benchmarks across various underwater conditions.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('oceancvbench')

# Import version
from .version import __version__

# Package info
__author__ = 'OceanCVBench Team'

# Export common utilities
from .utils.common import is_debug_mode, set_debug_mode

# Initialize debug mode from environment variable if present
debug_env = os.environ.get('OCEANCV_DEBUG', '').lower() in ('true', '1', 't')
set_debug_mode(debug_env)

# Import submodules for easy access
from . import analytics
from . import data_preprocessing
from . import evaluation
from . import inference
from . import training

__all__ = [
    'analytics',
    'data_preprocessing',
    'evaluation',
    'inference',
    'training',
    '__version__',
    'is_debug_mode',
    'set_debug_mode',
]
