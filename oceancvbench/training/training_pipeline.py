"""
Training pipeline module for OceanCVBench.

This module handles fine-tuning YOLO models on the benchmark dataset.
"""

import os
import logging
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Optional

# Set up logging
logger = logging.getLogger("oceancvbench.training_pipeline")

def ensure_ultralytics():
    """Ensure that the ultralytics package is installed."""
    try:
        import ultralytics
    except ImportError:
        logger.warning("Ultralytics package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        logger.info("Ultralytics installed successfully.")

def fine_tune_model(
    weights_path: str,
    data_yaml_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    workers: int = 4,
    device: str = ""
) -> str:
    """
    Fine-tune a YOLO model on the benchmark dataset.
    
    Args:
        weights_path: Path to the pre-trained YOLO weights (.pt file)
        data_yaml_path: Path to the data.yaml file
        output_path: Path where to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Image size for training
        workers: Number of worker threads
        device: Device to use (e.g., "0" for GPU 0, "" for CPU)
        
    Returns:
        Path to the fine-tuned model
    """
    logger.info(f"Fine-tuning model {os.path.basename(weights_path)} for {epochs} epochs")
    
    # Ensure ultralytics is installed
    ensure_ultralytics()
    
    from ultralytics import YOLO
    
    try:
        # Load the model
        model = YOLO(weights_path)
        
        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'workers': workers,
            'device': device,
            'name': 'oceancvbench_finetune',
            'project': os.path.dirname(output_path),
            'exist_ok': True,
            'patience': 20,  # Early stopping patience
            'save': True,    # Save checkpoints
            'verbose': True  # Print verbose output
        }
        
        # Start training
        logger.info(f"Starting fine-tuning with args: {train_args}")
        model.train(**train_args)
        
        # Get the best model path
        run_dir = os.path.join(os.path.dirname(output_path), 'oceancvbench_finetune')
        best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
        
        # Copy the best model to the output path
        import shutil
        shutil.copyfile(best_model_path, output_path)
        
        logger.info(f"Fine-tuning completed. Best model saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        raise
