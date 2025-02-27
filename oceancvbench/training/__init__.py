"""
Training module for OceanCVBench.

This module provides utilities for training and fine-tuning computer vision models
on marine imagery datasets.
"""

from .training_pipeline import train_model, fine_tune_model

__all__ = ['train_model', 'fine_tune_model']
