"""
Inference module for OceanCVBench.

This module provides functions for running inference with loaded models.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import torch

# Set up logging
logger = logging.getLogger("oceancvbench.inference.inference")

def run_inference_on_image(model: Any, image: np.ndarray, conf_thresh: float = None,
                         iou_thresh: float = None) -> Dict[str, Any]:
    """
    Run inference on a single image.
    
    Args:
        model: Loaded model
        image: Input image as numpy array
        conf_thresh: Optional confidence threshold override
        iou_thresh: Optional IoU threshold override
        
    Returns:
        Dictionary with inference results
    """
    try:
        # Record start time
        start_time = time.time()
        
        # Run inference
        if hasattr(model, 'predict'):  # YOLO model
            # Override thresholds if provided
            if conf_thresh is not None:
                original_conf = model.conf
                model.conf = conf_thresh
            
            if iou_thresh is not None:
                original_iou = model.iou
                model.iou = iou_thresh
                
            # Run prediction
            results = model.predict(image, verbose=False)
            
            # Restore original thresholds if they were changed
            if conf_thresh is not None:
                model.conf = original_conf
            
            if iou_thresh is not None:
                model.iou = original_iou
                
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract the first result (single image)
            result = results[0] if results else None
            
            return {
                'raw_output': result,
                'processing_time': processing_time,
                'success': True
            }
        else:
            logger.error("Unsupported model type for inference")
            return {
                'success': False,
                'error': "Unsupported model type"
            }
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def run_batch_inference(model: Any, images: List[np.ndarray], conf_thresh: float = None,
                      iou_thresh: float = None) -> List[Dict[str, Any]]:
    """
    Run inference on a batch of images.
    
    Args:
        model: Loaded model
        images: List of input images
        conf_thresh: Optional confidence threshold override
        iou_thresh: Optional IoU threshold override
        
    Returns:
        List of dictionaries with inference results
    """
    try:
        # Record start time
        start_time = time.time()
        
        # Run inference
        if hasattr(model, 'predict'):  # YOLO model
            # Override thresholds if provided
            if conf_thresh is not None:
                original_conf = model.conf
                model.conf = conf_thresh
            
            if iou_thresh is not None:
                original_iou = model.iou
                model.iou = iou_thresh
                
            # Run prediction
            results = model.predict(images, verbose=False)
            
            # Restore original thresholds if they were changed
            if conf_thresh is not None:
                model.conf = original_conf
            
            if iou_thresh is not None:
                model.iou = original_iou
                
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Prepare results
            return [
                {
                    'raw_output': result,
                    'processing_time': total_processing_time / len(images),  # Estimate time per image
                    'success': True
                }
                for result in results
            ]
        else:
            logger.error("Unsupported model type for batch inference")
            return [
                {
                    'success': False,
                    'error': "Unsupported model type"
                }
                for _ in images
            ]
            
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        return [
            {
                'success': False,
                'error': str(e)
            }
            for _ in images
        ]


def run_inference_on_tiles(model: Any, tiles: List[Dict[str, Any]], 
                         conf_thresh: float = None, iou_thresh: float = None) -> List[Dict[str, Any]]:
    """
    Run inference on image tiles and track their coordinates.
    
    Args:
        model: Loaded model
        tiles: List of tile dictionaries with 'tile', 'x', 'y', 'width', 'height' keys
        conf_thresh: Optional confidence threshold override
        iou_thresh: Optional IoU threshold override
        
    Returns:
        List of dictionaries with inference results and tile coordinates
    """
    # Extract just the tile images for batch processing
    tile_images = [t['tile'] for t in tiles]
    
    # Run inference on all tiles
    tile_results = run_batch_inference(model, tile_images, conf_thresh, iou_thresh)
    
    # Combine results with tile coordinates
    for i, (tile, result) in enumerate(zip(tiles, tile_results)):
        # Add tile coordinates to result
        result['x'] = tile['x']
        result['y'] = tile['y']
        result['width'] = tile['width']
        result['height'] = tile['height']
    
    return tile_results


def time_inference(model: Any, image: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
    """
    Time inference performance on a single image.
    
    Args:
        model: Loaded model
        image: Input image
        num_runs: Number of runs for timing
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    # Warm-up run
    _ = run_inference_on_image(model, image)
    
    # Timed runs
    for _ in range(num_runs):
        start_time = time.time()
        _ = run_inference_on_image(model, image)
        times.append(time.time() - start_time)
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
        'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
    }


def extract_detection_boxes(inference_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detection boxes from inference results.
    
    Args:
        inference_result: Dictionary with inference results
        
    Returns:
        List of dictionaries containing detection boxes
    """
    boxes = []
    
    if not inference_result.get('success', False):
        return boxes
    
    raw_output = inference_result.get('raw_output')
    if raw_output is None:
        return boxes
    
    try:
        # For YOLO model results
        if hasattr(raw_output, 'boxes'):
            raw_boxes = raw_output.boxes
            
            # Get the original image dimensions
            orig_shape = raw_output.orig_shape
            
            # Process each box
            for i in range(len(raw_boxes)):
                # Get box data (xyxy format)
                box = raw_boxes.xyxy[i].cpu().numpy()
                conf = float(raw_boxes.conf[i])
                class_id = int(raw_boxes.cls[i])
                
                # Get label if available
                label = raw_output.names.get(class_id, f"class_{class_id}") if hasattr(raw_output, 'names') else f"class_{class_id}"
                
                # Create detection dictionary
                detection = {
                    'xmin': float(box[0]),
                    'ymin': float(box[1]),
                    'xmax': float(box[2]),
                    'ymax': float(box[3]),
                    'width': float(box[2] - box[0]),
                    'height': float(box[3] - box[1]),
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': label,
                    'area': float((box[2] - box[0]) * (box[3] - box[1])),
                    'img_width': orig_shape[1],
                    'img_height': orig_shape[0]
                }
                
                boxes.append(detection)
        
        return boxes
        
    except Exception as e:
        logger.error(f"Error extracting detection boxes: {e}")
        return []


def extract_batch_detections(batch_results: List[Dict[str, Any]], 
                           image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract detections from a batch of inference results.
    
    Args:
        batch_results: List of inference result dictionaries
        image_paths: Optional list of image paths to associate with results
        
    Returns:
        Dictionary with combined results
    """
    all_detections = []
    processing_times = []
    failed_images = []
    
    for i, result in enumerate(batch_results):
        # Get image path if available
        image_path = image_paths[i] if image_paths and i < len(image_paths) else f"image_{i}"
        
        # Extract processing time
        if 'processing_time' in result:
            processing_times.append(result['processing_time'])
        
        # Check if inference was successful
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            failed_images.append({'image_path': image_path, 'error': error_msg})
            continue
        
        # Extract detections
        detections = extract_detection_boxes(result)
        
        # Add image path to each detection
        for det in detections:
            det['image_path'] = image_path
            all_detections.append(det)
    
    return {
        'all_detections': all_detections,
        'mean_processing_time': np.mean(processing_times) if processing_times else 0,
        'total_detections': len(all_detections),
        'failed_images': failed_images,
        'success_rate': 1.0 - len(failed_images) / len(batch_results) if batch_results else 0
    }


def run_inference_pipeline(model: Any, image_paths: List[str], 
                         conf_thresh: float = 0.25, iou_thresh: float = 0.45,
                         img_size: int = 640, batch_size: int = 16,
                         device: str = None) -> Dict[str, Any]:
    """
    Run the complete inference pipeline on a list of images.
    
    Args:
        model: Loaded model
        image_paths: List of paths to images
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold
        img_size: Size to resize images to
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        Dictionary with detection results
    """
    from .preprocessing import prepare_batch_from_images
    from .postprocessing import apply_confidence_filter, non_max_suppression
    
    # Set device if provided
    if device:
        if hasattr(model, 'to'):
            model.to(device)
    
    # Prepare batches
    batches = prepare_batch_from_images(
        image_paths=image_paths,
        target_size=(img_size, img_size),
        normalize=False,  # YOLO handles normalization internally
        max_batch_size=batch_size
    )
    
    all_detections = []
    total_time = 0.0
    
    # Process each batch
    for batch in batches:
        # Extract preprocessed images
        batch_images = [item['processed_image'] for item in batch]
        batch_paths = [item['image_path'] for item in batch]
        
        # Run inference
        batch_results = run_batch_inference(model, batch_images, conf_thresh, iou_thresh)
        
        # Extract detections
        batch_detections = extract_batch_detections(batch_results, batch_paths)
        all_detections.extend(batch_detections['all_detections'])
        total_time += batch_detections['mean_processing_time'] * len(batch)
    
    # Apply postprocessing
    filtered_detections = apply_confidence_filter(all_detections, threshold=conf_thresh)
    final_detections = non_max_suppression(filtered_detections, iou_threshold=iou_thresh)
    
    # Format results
    results = {
        'detections': final_detections,
        'total_images': len(image_paths),
        'total_detections': len(final_detections),
        'processing_time_seconds': total_time,
        'fps': len(image_paths) / total_time if total_time > 0 else 0
    }
    
    return results
