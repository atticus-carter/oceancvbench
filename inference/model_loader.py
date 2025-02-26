"""
Model loader module for OceanCVBench.

This module provides utilities for loading different types of models
with appropriate configurations.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
import torch

# Set up logging
logger = logging.getLogger("oceancvbench.inference.model_loader")

def load_yolo_model(model_path: str, conf_thresh: float = 0.4, 
                   iou_thresh: float = 0.5, device: str = '') -> Any:
    """
    Loads a YOLO model with specified thresholds.
    
    Args:
        model_path: Path to the YOLO .pt model file
        conf_thresh: Confidence threshold for detections
        iou_thresh: IoU threshold for non-maximum suppression
        device: Device to run inference on ('cpu', '0', '0,1', etc.)
        
    Returns:
        Loaded YOLO model ready for inference
    """
    try:
        # Import here to avoid dependency if not needed
        from ultralytics import YOLO
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Load the model
        model = YOLO(model_path)
        
        # Set the device if specified
        if device:
            model.to(device)
        
        # Set the inference parameters
        model.conf = conf_thresh
        model.iou = iou_thresh
        
        logger.info(f"Loaded YOLO model from {model_path}")
        logger.info(f"Configuration: conf_thresh={conf_thresh}, iou_thresh={iou_thresh}, device={device or 'default'}")
        
        return model
    
    except ImportError:
        logger.error("Error: ultralytics package not found. Please install it: pip install ultralytics")
        return None
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get information about a loaded model.
    
    Args:
        model: A loaded model
        
    Returns:
        Dictionary with model information
    """
    if model is None:
        return {"error": "Model is None"}
    
    try:
        # For YOLO models
        if hasattr(model, 'names'):
            return {
                "type": "YOLO",
                "class_names": model.names,
                "num_classes": len(model.names),
                "conf_threshold": getattr(model, 'conf', None),
                "iou_threshold": getattr(model, 'iou', None),
            }
        else:
            # Generic model info
            return {
                "type": model.__class__.__name__,
                "device": next(model.parameters()).device.type if hasattr(model, 'parameters') else "unknown"
            }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e)}


def get_available_devices() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    devices = {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": []
    }
    
    if devices["cuda_available"]:
        for i in range(devices["cuda_device_count"]):
            try:
                devices["cuda_devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                })
            except Exception as e:
                logger.error(f"Error getting CUDA device info for device {i}: {e}")
    
    return devices


def auto_select_device() -> str:
    """
    Automatically select the best available device for inference.
    
    Returns:
        Device string (e.g., '0' for CUDA device 0, 'cpu' for CPU)
    """
    devices = get_available_devices()
    
    if devices["cuda_available"] and devices["cuda_device_count"] > 0:
        # Get device with most memory
        if devices["cuda_devices"]:
            best_device = max(devices["cuda_devices"], key=lambda x: x.get("memory_total", 0))
            logger.info(f"Auto-selected CUDA device: {best_device['index']} ({best_device['name']})")
            return str(best_device["index"])
    
    logger.info("Auto-selected device: CPU (no CUDA devices available)")
    return "cpu"
