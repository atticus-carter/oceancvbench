"""
Evaluation module for OceanCVBench.

This module handles running inference on the benchmark test set and collecting metrics.
"""

import os
import json
import time
import logging
import glob
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple
import pandas as pd

# Set up logging
logger = logging.getLogger("oceancvbench.evaluation")

def find_label_files(image_path: str) -> list:
    """
    Find the label files corresponding to an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of label file paths
    """
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Check in the 'labels' directory
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)
    
    possible_paths = [
        # Label in the same directory
        os.path.join(image_dir, f"{base_name}.txt"),
        # Label in a 'labels' directory at the same level as 'images'
        os.path.join(parent_dir, "labels", f"{base_name}.txt"),
        # Label in a 'labels' directory within the image directory
        os.path.join(image_dir, "labels", f"{base_name}.txt")
    ]
    
    # Return all existing paths
    return [path for path in possible_paths if os.path.exists(path)]

def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Dict[str, float]]:
    """
    Parse a YOLO label file and convert normalized coordinates to pixel values.
    
    Args:
        label_path: Path to the label file
        img_width: Width of the image in pixels
        img_height: Height of the image in pixels
        
    Returns:
        List of bounding boxes in format [class_id, x1, y1, x2, y2, conf]
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to top-left and bottom-right coordinates
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                
                boxes.append({
                    'class_id': class_id,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'confidence': 1.0  # Ground truth always has confidence 1.0
                })
                
    return boxes

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (width, height)
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}, using default dimensions")
            return (640, 640)
        
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        return (640, 640)  # Default fallback

def evaluate_model_on_benchmark(
    model_path: str,
    benchmark_dir: str,
    img_size: int = 640,
    device: str = '',
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate a model on the benchmark test set.
    
    Args:
        model_path: Path to the YOLO model
        benchmark_dir: Path to the benchmark test set directory
        img_size: Image size for inference
        device: Device to use for inference
        conf_thresh: Confidence threshold for detections
        iou_thresh: IoU threshold for NMS
        
    Returns:
        Dictionary containing evaluation results
    """
    from inference.yolo_integration import load_yolo_model
    from inference.localize import localize_images
    
    logger.info(f"Evaluating model {os.path.basename(model_path)} on benchmark test set")
    
    # Load the model
    model = load_yolo_model(model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    
    if model is None:
        logger.error("Failed to load model")
        raise RuntimeError("Failed to load YOLO model")
    
    # Find all challenge folders
    challenge_folders = [d for d in os.listdir(benchmark_dir) 
                         if os.path.isdir(os.path.join(benchmark_dir, d))]
    
    results = {}
    
    # Process each challenge folder
    for folder in challenge_folders:
        folder_path = os.path.join(benchmark_dir, folder)
        logger.info(f"Running inference on challenge: {folder}")
        
        # Run inference on this folder
        start_time = time.time()
        detections_df = localize_images(
            folder=folder_path,
            model_path=model_path,  # Using the model path directly
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            csv=False,
            save_images=False
        )
        inference_time = time.time() - start_time
        
        # Collect ground truth for this folder
        gt_boxes = []
        image_files = []
        
        # Find all images in this folder
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            # Also check in images subfolder if exists
            subfolder = os.path.join(folder_path, 'images')
            if os.path.exists(subfolder):
                image_files.extend(glob.glob(os.path.join(subfolder, ext)))
        
        # Process each image to get ground truth
        for img_path in image_files:
            img_width, img_height = get_image_dimensions(img_path)
            
            # Find and parse label files
            label_files = find_label_files(img_path)
            
            for label_path in label_files:
                boxes = parse_yolo_label(label_path, img_width, img_height)
                for box in boxes:
                    box['filename'] = os.path.basename(img_path)
                    box['image_path'] = img_path
                    box['img_width'] = img_width
                    box['img_height'] = img_height
                    gt_boxes.append(box)
        
        # Convert ground truth to DataFrame
        gt_df = pd.DataFrame(gt_boxes) if gt_boxes else pd.DataFrame()
        
        # Store results for this folder
        results[folder] = {
            'detections': detections_df,
            'ground_truth': gt_df,
            'inference_time': inference_time,
            'image_count': len(image_files)
        }
    
    logger.info("Evaluation completed on all benchmark folders")
    return results
