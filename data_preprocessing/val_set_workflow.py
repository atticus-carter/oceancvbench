"""Functions for validation set preparation and augmentation."""

import os
import shutil
from pathlib import Path
import cv2
from .augmentation import augment_biofouling, augment_shear_perspective, augment_camera_distance


def prepare_val_set(val_dir, output_dir, triple=True):
    """
    Copies validation images to a new folder, optionally tripling them for augmentation.
    
    Args:
        val_dir: Source directory with validation images
        output_dir: Destination directory for the processed validation set
        triple: If True, create three copies of each image for different augmentations
        
    Returns:
        List of paths to the copied images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(val_dir) 
                  if os.path.isfile(os.path.join(val_dir, f)) and 
                  os.path.splitext(f.lower())[1] in valid_extensions]
    
    copied_paths = []
    
    for img_file in image_files:
        src_path = os.path.join(val_dir, img_file)
        
        if triple:
            # Create three copies with different suffixes
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            
            for suffix in ['_orig', '_aug1', '_aug2']:
                dst_filename = f"{base_name}{suffix}{ext}"
                dst_path = os.path.join(output_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
                copied_paths.append(dst_path)
        else:
            # Just copy the file once
            dst_path = os.path.join(output_dir, img_file)
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)
    
    print(f"Copied {len(image_files)} images to {output_dir}")
    if triple:
        print(f"Created {len(copied_paths)} files (3 copies of each image)")
    
    return copied_paths


def run_augmentations_on_val(val_dir, aug_output_dir, fouling_path="biofouling_texture.jpg"):
    """
    Applies augmentations to validation images.
    
    Args:
        val_dir: Directory containing validation images
        aug_output_dir: Directory where augmented images will be saved
        fouling_path: Path to biofouling texture image
        
    Returns:
        Dictionary mapping original filenames to lists of augmented file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(aug_output_dir, exist_ok=True)
    
    # Check if fouling texture exists
    if not os.path.exists(fouling_path):
        print(f"Warning: Biofouling texture not found at {fouling_path}. Using fallback.")
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(val_dir) 
                  if os.path.isfile(os.path.join(val_dir, f)) and 
                  os.path.splitext(f.lower())[1] in valid_extensions]
    
    results = {}
    
    for img_file in image_files:
        src_path = os.path.join(val_dir, img_file)
        image = cv2.imread(src_path)
        
        if image is None:
            print(f"Warning: Could not read {src_path}, skipping.")
            continue
            
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        augmented_paths = []
        
        # 1. Biofouling augmentation
        try:
            biofouling_img = augment_biofouling(image.copy(), fouling_path)
            biofouling_path = os.path.join(aug_output_dir, f"{base_name}_biofouling{ext}")
            cv2.imwrite(biofouling_path, biofouling_img)
            augmented_paths.append(biofouling_path)
        except Exception as e:
            print(f"Error applying biofouling to {img_file}: {e}")
        
        # 2. Shear/perspective augmentation
        try:
            shear_img = augment_shear_perspective(image.copy())
            shear_path = os.path.join(aug_output_dir, f"{base_name}_shear{ext}")
            cv2.imwrite(shear_path, shear_img)
            augmented_paths.append(shear_path)
        except Exception as e:
            print(f"Error applying shear to {img_file}: {e}")
            
        # 3. Camera distance augmentation
        try:
            # Zoom in (closer)
            zoom_in_img = augment_camera_distance(image.copy(), scale=1.2)
            zoom_in_path = os.path.join(aug_output_dir, f"{base_name}_closer{ext}")
            cv2.imwrite(zoom_in_path, zoom_in_img)
            augmented_paths.append(zoom_in_path)
            
            # Zoom out (farther)
            zoom_out_img = augment_camera_distance(image.copy(), scale=0.8)
            zoom_out_path = os.path.join(aug_output_dir, f"{base_name}_farther{ext}")
            cv2.imwrite(zoom_out_path, zoom_out_img)
            augmented_paths.append(zoom_out_path)
        except Exception as e:
            print(f"Error applying distance augmentation to {img_file}: {e}")
        
        results[img_file] = augmented_paths
    
    print(f"Applied augmentations to {len(image_files)} images")
    print(f"Created {sum(len(paths) for paths in results.values())} augmented images")
    
    return results
