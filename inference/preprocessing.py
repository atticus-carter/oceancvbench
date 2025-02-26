"""
Image preprocessing module for OceanCVBench.

This module provides utilities for preparing images before inference,
including resizing, normalization, and tiling.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger("oceancvbench.inference.preprocessing")

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target size, optionally preserving aspect ratio.
    
    Args:
        image: Input image (numpy array)
        target_size: Target size as (width, height)
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    
    if keep_aspect_ratio:
        # Calculate the scale factor to maintain aspect ratio
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas of the target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding to center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas, scale, (x_offset, y_offset)
    else:
        # Simply resize to target dimensions
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized, target_width / original_width, (0, 0)


def normalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406], 
                  std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        image: Input image (numpy array)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized image
    """
    # Convert to float32 if not already
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Scale to [0, 1]
    image = image / 255.0
    
    # Normalize with mean and std
    for i in range(3):  # For each channel
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
    return image


def create_image_tiles(image: np.ndarray, tile_size: Tuple[int, int], 
                     overlap: float = 0.2) -> List[Dict[str, Any]]:
    """
    Split a large image into overlapping tiles for processing.
    
    Args:
        image: Input image (numpy array)
        tile_size: Size of each tile as (width, height)
        overlap: Overlap between tiles (0.0 to 1.0)
        
    Returns:
        List of dictionaries containing tiles and their coordinates
    """
    height, width = image.shape[:2]
    tile_width, tile_height = tile_size
    
    # Calculate step size with overlap
    stride_x = int(tile_width * (1 - overlap))
    stride_y = int(tile_height * (1 - overlap))
    
    # Ensure stride is at least 1
    stride_x = max(1, stride_x)
    stride_y = max(1, stride_y)
    
    tiles = []
    
    # Generate tiles
    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            # Adjust coordinates if they exceed image dimensions
            right = min(x + tile_width, width)
            bottom = min(y + tile_height, height)
            
            # Handle edge case for last column/row
            if right == width and x > 0:
                x = max(0, width - tile_width)
            if bottom == height and y > 0:
                y = max(0, height - tile_height)
            
            # Extract tile
            tile = image[y:bottom, x:right]
            
            # Add to list only if the tile is of sufficient size
            if tile.shape[0] > 0 and tile.shape[1] > 0:
                tiles.append({
                    'tile': tile,
                    'x': x,
                    'y': y,
                    'width': right - x,
                    'height': bottom - y
                })
    
    logger.debug(f"Created {len(tiles)} tiles from image of size {width}x{height}")
    return tiles


def preprocess_for_model(image: np.ndarray, target_size: Tuple[int, int],
                       normalize: bool = True) -> Dict[str, Any]:
    """
    Preprocess an image for model input with all necessary steps.
    
    Args:
        image: Input image (numpy array)
        target_size: Target size for the model input
        normalize: Whether to normalize the image
        
    Returns:
        Dictionary with preprocessed image and metadata
    """
    # Check if image is valid
    if image is None or image.size == 0:
        logger.error("Invalid image for preprocessing")
        return None
    
    # Store original image shape
    original_shape = image.shape
    
    # Resize the image
    resized_image, scale_factor, padding = resize_image(image, target_size, keep_aspect_ratio=True)
    
    # Normalize if requested
    if normalize:
        resized_image = normalize_image(resized_image)
    
    # Prepare result
    result = {
        'processed_image': resized_image,
        'original_shape': original_shape,
        'scale_factor': scale_factor,
        'padding': padding,
        'target_size': target_size
    }
    
    return result


def load_and_preprocess_image(image_path: Union[str, Path], target_size: Tuple[int, int],
                            normalize: bool = False) -> Dict[str, Any]:
    """
    Load an image from disk and preprocess it.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the model input
        normalize: Whether to normalize the image
        
    Returns:
        Dictionary with preprocessed image and metadata
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        result = preprocess_for_model(image, target_size, normalize)
        
        # Add the image path
        result['image_path'] = str(image_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None


def prepare_batch_from_images(image_paths: List[Union[str, Path]], target_size: Tuple[int, int],
                           normalize: bool = False, max_batch_size: int = 16) -> List[Dict[str, Any]]:
    """
    Prepare batches of preprocessed images for inference.
    
    Args:
        image_paths: List of paths to image files
        target_size: Target size for the model input
        normalize: Whether to normalize the images
        max_batch_size: Maximum batch size
        
    Returns:
        List of batches, where each batch is a dictionary with preprocessed images and metadata
    """
    preprocessed_images = []
    batches = []
    
    # Preprocess all images
    for path in image_paths:
        processed = load_and_preprocess_image(path, target_size, normalize)
        if processed:
            preprocessed_images.append(processed)
    
    # Create batches
    for i in range(0, len(preprocessed_images), max_batch_size):
        batch = preprocessed_images[i:i+max_batch_size]
        batches.append(batch)
    
    logger.info(f"Prepared {len(batches)} batches from {len(preprocessed_images)} images")
    return batches
