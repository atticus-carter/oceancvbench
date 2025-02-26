"""
Evaluation module for OceanCVBench.

This module provides utilities for evaluating model performance.
"""

from .metrics import (
    calculate_iou, calculate_precision_recall, calculate_map,
    create_confusion_matrix, plot_precision_recall_curve,
    plot_confusion_matrix
)

__all__ = [
    'calculate_iou', 'calculate_precision_recall', 'calculate_map',
    'create_confusion_matrix', 'plot_precision_recall_curve',
    'plot_confusion_matrix'
]
