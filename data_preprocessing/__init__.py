"""
Data preprocessing module for OceanCVBench.

This module provides utilities for preparing and augmenting datasets.
"""

from .augmentation import (
    augment_biofouling, augment_shear_perspective, 
    augment_camera_distance
)
from .underrepresented_classes import (
    find_underrepresented_classes,
    generate_synthetic_data,
    visualize_class_distribution
)
from .outlier_detection import (
    detect_whole_image_outliers,
    detect_bounding_box_outliers
)
from .dataset_stats import (
    extract_image_stats,
    analyze_dataset_images,
    generate_dataset_report
)

__all__ = [
    'augment_biofouling', 'augment_shear_perspective', 'augment_camera_distance',
    'find_underrepresented_classes', 'generate_synthetic_data', 'visualize_class_distribution',
    'detect_whole_image_outliers', 'detect_bounding_box_outliers',
    'extract_image_stats', 'analyze_dataset_images', 'generate_dataset_report'
]
