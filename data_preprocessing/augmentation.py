"""Image augmentation functions for underwater imagery."""

import cv2
import numpy as np
import random


def augment_biofouling(image, synthetic_fouling_path):
    """
    Overlays a synthetic "biofouling" texture on an image to simulate underwater growths.
    
    Args:
        image: Input image (numpy array)
        synthetic_fouling_path: Path to biofouling texture image
        
    Returns:
        Augmented image with biofouling effect
    """
    # Load biofouling texture
    try:
        fouling = cv2.imread(synthetic_fouling_path, cv2.IMREAD_UNCHANGED)
        if fouling is None:
            raise FileNotFoundError(f"Could not load biofouling texture from {synthetic_fouling_path}")
    except Exception as e:
        print(f"Error loading biofouling texture: {e}")
        return image
    
    # Resize fouling texture to match input image
    fouling = cv2.resize(fouling, (image.shape[1], image.shape[0]))
    
    # If fouling has alpha channel, use it, otherwise create random opacity
    if fouling.shape[2] == 4:
        alpha = fouling[:, :, 3] / 255.0
        fouling = fouling[:, :, :3]
    else:
        # Create random opacity mask
        alpha = np.random.uniform(0.1, 0.4, size=(image.shape[0], image.shape[1]))
    
    # Apply the fouling with alpha blending
    alpha = np.expand_dims(alpha, axis=2)
    augmented = image * (1 - alpha) + fouling * alpha
    
    return augmented.astype(np.uint8)


def augment_shear_perspective(image, shear_factor=0.2):
    """
    Applies a shear or perspective shift to simulate camera movement.
    
    Args:
        image: Input image (numpy array)
        shear_factor: Strength of the shear transformation
        
    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    
    # Create random shear matrix
    shear_amount_x = random.uniform(-shear_factor, shear_factor)
    shear_amount_y = random.uniform(-shear_factor, shear_factor)
    
    # Create transformation matrix
    M = np.float32([
        [1, shear_amount_x, 0],
        [shear_amount_y, 1, 0]
    ])
    
    # Apply the transformation
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed


def augment_camera_distance(image, scale=1.2):
    """
    Rescales the image to simulate objects at different distances.
    
    Args:
        image: Input image (numpy array)
        scale: Scale factor (>1 for zoom in, <1 for zoom out)
        
    Returns:
        Scaled image
    """
    h, w = image.shape[:2]
    
    # If scale > 1, we're zooming in (simulating closer distance)
    if scale > 1:
        # Calculate the center
        center_x, center_y = w // 2, h // 2
        # Calculate new dimensions
        new_w, new_h = int(w/scale), int(h/scale)
        
        # Calculate the crop coordinates
        start_x = center_x - new_w // 2
        start_y = center_y - new_h // 2
        end_x = start_x + new_w
        end_y = start_y + new_h
        
        # Ensure within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)
        
        # Crop and resize
        cropped = image[start_y:end_y, start_x:end_x]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
        
    # If scale < 1, we're zooming out (simulating farther distance)
    elif scale < 1:
        # Resize to smaller
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Create a canvas of the original size
        canvas = np.zeros_like(image)
        
        # Calculate where to paste the resized image
        paste_x = (w - resized.shape[1]) // 2
        paste_y = (h - resized.shape[0]) // 2
        
        # Paste the resized image
        canvas[paste_y:paste_y+resized.shape[0], paste_x:paste_x+resized.shape[1]] = resized
        return canvas
        
    else:
        return image.copy()
