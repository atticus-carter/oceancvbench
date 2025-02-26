"""File operation utilities."""

import os
import yaml
import shutil
import zipfile
import random
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger('oceancvbench.utils.file_ops')

def create_data_yaml(train_dir: str, val_dir: str, names: Union[List[str], Dict[int, str]], 
                     out_path: str = "data.yaml") -> str:
    """
    Builds a minimal YOLO-friendly config file (data.yaml).
    
    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory
        names: List or dictionary of class names
        out_path: Output path for the YAML file
        
    Returns:
        Path to the created YAML file
    """
    # If names is a dictionary, ensure it's properly formatted
    if isinstance(names, dict):
        # Make sure keys are integers and values are strings
        names = {int(k): str(v) for k, v in names.items()}
    
    yaml_data = {
        'train': os.path.abspath(train_dir),
        'val': os.path.abspath(val_dir),
        'nc': len(names),  # number of classes
        'names': names
    }
    
    # Ensure directories exist in yaml_data paths
    for key in ['train', 'val']:
        if not os.path.exists(yaml_data[key]):
            logger.warning(f"{key} directory does not exist: {yaml_data[key]}")
    
    # Create parent directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Write the YAML file
    with open(out_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"Created data YAML file at {out_path}")
    return out_path


def create_balanced_splits(input_folder: str, output_folder: str, 
                          val_ratio: float = 0.2, test_ratio: float = 0.0) -> Tuple[str, str, str]:
    """
    Split images into train/val/test folders with balanced class distribution.
    
    Args:
        input_folder: Source folder with images and annotations
        output_folder: Destination for split datasets
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (train_dir, val_dir, test_dir) paths
    """
    # Create output directory structure
    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')
    
    for directory in [train_dir, val_dir]:
        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'labels'), exist_ok=True)
    
    if test_ratio > 0:
        os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
    
    # Check if input has images/labels structure
    input_has_structure = os.path.exists(os.path.join(input_folder, 'images')) and \
                         os.path.exists(os.path.join(input_folder, 'labels'))
    
    # Find all image files
    if input_has_structure:
        image_dir = os.path.join(input_folder, 'images')
        label_dir = os.path.join(input_folder, 'labels')
    else:
        image_dir = input_folder
        label_dir = input_folder
    
    # Get image files
    from .common import get_file_extensions
    image_exts = get_file_extensions('image')
    
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in image_exts]
    
    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return train_dir, val_dir, test_dir
    
    # Build a class distribution map if label files exist
    class_images = {}  # Map class_id to list of image files
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            # Read the label file to identify classes
            with open(label_path, 'r') as f:
                classes = set()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes.add(class_id)
            
            # Add image to each class it contains
            for class_id in classes:
                if class_id not in class_images:
                    class_images[class_id] = []
                class_images[class_id].append(img_file)
        else:
            # No label file - add to a default class
            if -1 not in class_images:
                class_images[-1] = []
            class_images[-1].append(img_file)
    
    # Determine split ratios
    train_ratio = 1.0 - val_ratio - test_ratio
    
    # Split each class maintaining proportions
    all_train, all_val, all_test = [], [], []
    
    for class_id, imgs in class_images.items():
        random.shuffle(imgs)
        
        n_total = len(imgs)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test
        
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train+n_val]
        test_imgs = imgs[n_train+n_val:] if test_ratio > 0 else []
        
        all_train.extend(train_imgs)
        all_val.extend(val_imgs)
        all_test.extend(test_imgs)
    
    # Copy files to destination folders
    def copy_files(image_list, dest_subdir):
        for img_file in image_list:
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(dest_subdir, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label if exists
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            src_label = os.path.join(label_dir, label_file)
            
            if os.path.exists(src_label):
                dst_label = os.path.join(dest_subdir, 'labels', label_file)
                shutil.copy2(src_label, dst_label)
    
    copy_files(all_train, train_dir)
    copy_files(all_val, val_dir)
    if test_ratio > 0:
        copy_files(all_test, test_dir)
    
    # Log results
    logger.info(f"Split dataset: {len(all_train)} train, {len(all_val)} val, {len(all_test)} test images")
    
    return train_dir, val_dir, test_dir


def zip_folder(folder_path: str, output_zip_name: str = "export") -> str:
    """
    Recursively zips a folder and all its contents.
    
    Args:
        folder_path: Path to the folder to zip
        output_zip_name: Name of the output zip file (without extension)
        
    Returns:
        Path to the created zip file
    """
    # If output_zip_name doesn't end with .zip, add it
    if not output_zip_name.lower().endswith('.zip'):
        output_zip_name += '.zip'
    
    # Create the output path
    if os.path.dirname(output_zip_name):
        # If a path is specified in output_zip_name, use it
        output_path = output_zip_name
    else:
        # Otherwise, create the zip next to the folder
        output_path = os.path.join(os.path.dirname(folder_path), output_zip_name)
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    folder_path = os.path.abspath(folder_path)
    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    
    # Create the zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Create arcname (path within the zip file)
                arcname = os.path.join(folder_name, os.path.relpath(file_path, folder_path))
                zipf.write(file_path, arcname=arcname)
    
    logger.info(f"Created zip archive: {output_path}")
    return output_path
