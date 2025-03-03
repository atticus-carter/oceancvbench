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

# Package version
__version__ = '0.1.0'

# Package info
__author__ = 'OceanCV Team'

# Export common utilities
from .utils.common import is_debug_mode, set_debug_mode

# Initialize debug mode from environment variable if present
debug_env = os.environ.get('OCEANCV_DEBUG', '').lower() in ('true', '1', 't')
set_debug_mode(debug_env)
