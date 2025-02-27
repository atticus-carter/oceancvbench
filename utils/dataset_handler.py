"""
Dataset handling module for OceanCVBench.

This module provides utilities for standardizing, validating, and augmenting
datasets for YOLO model training and evaluation.
"""

import os
import yaml
import shutil
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import random
import glob
import json

# Import albumentations for image augmentation
try:
    import albumentations as A
except ImportError:
    logging.warning("Albumentations not installed. Some augmentation features will be unavailable. "
                   "Install with: pip install albumentations")

# Set up logging
logger = logging.getLogger("oceancvbench.dataset_handler")

# Supported formats
SUPPORTED_FORMATS = {
    'yolo': '.txt',
    'coco': '.json',
    'voc': '.xml',
    'labelme': '.json'
}

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


class DatasetHandler:
    """
    Handler for processing, validating, and augmenting datasets.
    """
    
    def __init__(self, dataset_dir: str, format: str = 'yolo', class_names: Dict[int, str] = None):
        """
        Initialize the dataset handler.
        
        Args:
            dataset_dir: Path to the dataset directory
            format: Format of annotations (yolo, coco, voc, labelme)
            class_names: Dictionary mapping class IDs to class names
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.format = format.lower()
        self.class_names = class_names or {}
        
        # Validate format
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported formats: {list(SUPPORTED_FORMATS.keys())}")
        
        # Check if directory exists
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Initialize annotation extension
        self.annotation_ext = SUPPORTED_FORMATS[self.format]
    
    def validate_dataset_structure(self) -> Dict[str, Any]:
        """
        Validate the dataset structure based on the format.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # YOLO validation
        if self.format == 'yolo':
            # Check for required directories
            expected_dirs = ['images', 'labels']
            missing_dirs = []
            
            for dir_name in expected_dirs:
                dir_path = os.path.join(self.dataset_dir, dir_name)
                if not os.path.exists(dir_path):
                    # Also check if they might be one level up (common YOLO structure)
                    for split in ['train', 'val', 'test']:
                        split_dir = os.path.join(self.dataset_dir, split, dir_name)
                        if os.path.exists(split_dir):
                            break
                    else:  # No break occurred
                        missing_dirs.append(dir_name)
            
            if missing_dirs:
                results['warnings'].append(f"Missing standard directories: {', '.join(missing_dirs)}")
            
            # Look for images
            image_paths = []
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(glob.glob(os.path.join(self.dataset_dir, '**', f'*{ext}'), recursive=True))
            
            if not image_paths:
                results['errors'].append(f"No images found in {self.dataset_dir}")
                return results
            
            # Count annotations
            annotation_paths = glob.glob(os.path.join(self.dataset_dir, '**', f'*{self.annotation_ext}'), recursive=True)
            
            # Group by directory to understand structure
            directories = {}
            for path in image_paths:
                dir_name = os.path.dirname(path)
                if dir_name not in directories:
                    directories[dir_name] = {'images': 0, 'annotations': 0}
                directories[dir_name]['images'] += 1
            
            for path in annotation_paths:
                dir_name = os.path.dirname(path)
                if dir_name not in directories:
                    directories[dir_name] = {'images': 0, 'annotations': 0}
                directories[dir_name]['annotations'] += 1
            
            # Check for annotation-image correspondence
            for dir_name, counts in directories.items():
                if counts['images'] > 0 and counts['annotations'] == 0:
                    results['warnings'].append(f"Directory {dir_name} has images but no annotations")
            
            # Collect statistics
            results['stats'] = {
                'total_images': len(image_paths),
                'total_annotations': len(annotation_paths),
                'directories': len(directories)
            }
            
            # Check if data.yaml exists
            yaml_paths = glob.glob(os.path.join(self.dataset_dir, '*.yaml')) + glob.glob(os.path.join(self.dataset_dir, '*.yml'))
            if not yaml_paths:
                results['warnings'].append("No YAML configuration file found for class definitions")
            else:
                results['stats']['yaml_path'] = yaml_paths[0]
        
        # COCO validation
        elif self.format == 'coco':
            # Look for JSON annotations
            json_files = glob.glob(os.path.join(self.dataset_dir, '**', '*.json'), recursive=True)
            
            if not json_files:
                results['errors'].append(f"No JSON annotation files found in {self.dataset_dir}")
                return results
            
            # Try to parse and validate COCO format
            valid_coco_files = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                    # Check for required COCO keys
                    required_keys = ['images', 'annotations', 'categories']
                    if not all(key in data for key in required_keys):
                        continue
                        
                    valid_coco_files.append(json_file)
                except Exception as e:
                    logger.debug(f"Error parsing {json_file}: {e}")
            
            if not valid_coco_files:
                results['errors'].append("No valid COCO format JSON files found")
                return results
                
            # Look for referenced images
            image_paths = []
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(glob.glob(os.path.join(self.dataset_dir, '**', f'*{ext}'), recursive=True))
                
            results['stats'] = {
                'total_images': len(image_paths),
                'valid_coco_files': len(valid_coco_files),
                'coco_files': valid_coco_files
            }
        
        # Check if we have enough data
        if ('total_images' in results['stats'] and results['stats']['total_images'] == 0) or \
           ('total_annotations' in results['stats'] and results['stats']['total_annotations'] == 0):
            results['errors'].append("Dataset appears to be empty or has no annotations")
        
        # Set valid flag if no errors
        results['valid'] = len(results['errors']) == 0
        
        return results

    def validate_annotations(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate a sample of annotations to check for common issues.
        
        Args:
            sample_size: Number of annotations to check
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Find all annotation files
        annotation_paths = glob.glob(os.path.join(self.dataset_dir, '**', f'*{self.annotation_ext}'), recursive=True)
        
        if not annotation_paths:
            results['errors'].append(f"No annotation files found with extension {self.annotation_ext}")
            results['valid'] = False
            return results
        
        # Sample annotation files
        if sample_size < len(annotation_paths):
            annotation_sample = random.sample(annotation_paths, sample_size)
        else:
            annotation_sample = annotation_paths
        
        # Counters for statistics
        stats = {
            'empty_annotations': 0,
            'invalid_annotations': 0,
            'out_of_bounds': 0,
            'negative_dimensions': 0,
            'total_boxes': 0,
            'class_counts': {}
        }
        
        # Check each annotation
        for ann_path in annotation_sample:
            if self.format == 'yolo':
                # Get corresponding image path
                img_name = os.path.splitext(os.path.basename(ann_path))[0]
                img_path = None
                
                # Look for image with various extensions
                for ext in IMAGE_EXTENSIONS:
                    possible_img = glob.glob(os.path.join(self.dataset_dir, '**', f"{img_name}{ext}"), recursive=True)
                    if possible_img:
                        img_path = possible_img[0]
                        break
                
                if not img_path:
                    results['warnings'].append(f"Could not find image for annotation: {ann_path}")
                    continue
                
                # Get image dimensions
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        results['warnings'].append(f"Could not read image: {img_path}")
                        continue
                    img_height, img_width = img.shape[:2]
                except Exception as e:
                    results['warnings'].append(f"Error reading image {img_path}: {e}")
                    continue
                
                # Read annotation file
                try:
                    with open(ann_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        stats['empty_annotations'] += 1
                        continue
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            stats['invalid_annotations'] += 1
                            continue
                            
                        # Parse YOLO format
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Update class counts
                        stats['class_counts'][class_id] = stats['class_counts'].get(class_id, 0) + 1
                        stats['total_boxes'] += 1
                        
                        # Check for out of bounds
                        if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                            stats['out_of_bounds'] += 1
                            
                        # Check for negative dimensions
                        if width <= 0 or height <= 0:
                            stats['negative_dimensions'] += 1
                            
                except Exception as e:
                    results['warnings'].append(f"Error parsing annotation {ann_path}: {e}")
                    stats['invalid_annotations'] += 1
        
        # Add statistics to results
        results['stats'] = stats
        
        # Add warnings for issues
        if stats['empty_annotations'] > 0:
            results['warnings'].append(f"Found {stats['empty_annotations']} empty annotation files")
            
        if stats['invalid_annotations'] > 0:
            results['warnings'].append(f"Found {stats['invalid_annotations']} invalid annotation files")
            
        if stats['out_of_bounds'] > 0:
            results['errors'].append(f"Found {stats['out_of_bounds']} bounding boxes with coordinates out of bounds (not in [0,1])")
            results['valid'] = False
            
        if stats['negative_dimensions'] > 0:
            results['errors'].append(f"Found {stats['negative_dimensions']} bounding boxes with negative or zero dimensions")
            results['valid'] = False
        
        return results

    def convert_to_yolo_format(self, output_dir: str, splits: Dict[str, float] = None) -> Dict[str, str]:
        """
        Convert the dataset to YOLO format.
        
        Args:
            output_dir: Directory to save the YOLO format dataset
            splits: Dictionary defining train/val/test splits
            
        Returns:
            Dictionary with paths to the output YOLO directories
        """
        # Set default splits if not provided
        if splits is None:
            splits = {'train': 0.8, 'val': 0.2}
        
        # Normalize splits to ensure they sum to 1
        total = sum(splits.values())
        splits = {k: v/total for k, v in splits.items()}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories for each split
        split_dirs = {}
        for split in splits:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
            split_dirs[split] = split_dir
        
        # Find all images
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(self.dataset_dir, '**', f'*{ext}'), recursive=True))
        
        # Shuffle images
        random.shuffle(image_paths)
        
        # Calculate split indices
        split_indices = {}
        start_idx = 0
        for split, ratio in splits.items():
            count = int(len(image_paths) * ratio)
            split_indices[split] = (start_idx, start_idx + count)
            start_idx += count
        
        # Handle any rounding errors by adding remaining to last split
        last_split = list(splits.keys())[-1]
        split_indices[last_split] = (split_indices[last_split][0], len(image_paths))
        
        # Process each image according to its split
        results = {
            'splits': {},
            'class_counts': {},
            'total_images': len(image_paths),
            'total_boxes': 0,
            'errors': []
        }
        
        for split, (start, end) in split_indices.items():
            split_images = image_paths[start:end]
            results['splits'][split] = {'images': 0, 'annotations': 0, 'boxes': 0}
            
            for img_path in split_images:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Copy image to destination
                dest_img_path = os.path.join(split_dirs[split], 'images', f"{img_name}{os.path.splitext(img_path)[1]}")
                shutil.copy2(img_path, dest_img_path)
                results['splits'][split]['images'] += 1
                
                # Find annotations based on format
                annotations = []
                
                if self.format == 'yolo':
                    # Find YOLO annotation file
                    ann_filename = f"{img_name}{self.annotation_ext}"
                    # Look in common locations
                    possible_locations = [
                        os.path.join(os.path.dirname(img_path).replace('images', 'labels'), ann_filename),
                        os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', ann_filename)
                    ]
                    
                    ann_path = None
                    for loc in possible_locations:
                        if os.path.exists(loc):
                            ann_path = loc
                            break
                    
                    if ann_path:
                        # Since it's already in YOLO format, just copy it
                        dest_ann_path = os.path.join(split_dirs[split], 'labels', ann_filename)
                        shutil.copy2(ann_path, dest_ann_path)
                        
                        # Read and count boxes
                        try:
                            with open(ann_path, 'r') as f:
                                lines = f.readlines()
                                
                            results['splits'][split]['boxes'] += len(lines)
                            results['total_boxes'] += len(lines)
                            results['splits'][split]['annotations'] += 1
                            
                            # Count classes
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    results['class_counts'][class_id] = results['class_counts'].get(class_id, 0) + 1
                        except Exception as e:
                            results['errors'].append(f"Error processing {ann_path}: {e}")
                
                elif self.format == 'coco':
                    # COCO conversion would go here
                    # This is more complex and would require parsing the COCO JSON file,
                    # finding the annotations for this specific image, and converting to YOLO format
                    pass
                
                elif self.format == 'voc':
                    # VOC conversion would go here
                    pass
                
                elif self.format == 'labelme':
                    # LabelMe conversion would go here
                    pass
        
        # Create data.yaml
        yaml_path = os.path.join(output_dir, 'data.yaml')
        
        # Prepare class names
        class_names = {}
        if self.class_names:
            class_names = self.class_names
        else:
            for class_id in results['class_counts'].keys():
                class_names[class_id] = f"class_{class_id}"
        
        # Write data.yaml
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'names': class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        results['yaml_path'] = yaml_path
        
        return results

    def create_augmentation_pipeline(self, augmentation_level: str = 'medium') -> Any:
        """
        Create an augmentation pipeline using albumentations.
        
        Args:
            augmentation_level: Level of augmentation (light, medium, heavy)
            
        Returns:
            Albumentations transformation pipeline
        """
        if 'albumentations' not in globals():
            logger.error("Albumentations is not installed. Please install it with: pip install albumentations")
            return None
        
        # Define transformation levels
        if augmentation_level == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        elif augmentation_level == 'medium':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.2)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        elif augmentation_level == 'heavy':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        # For marine-specific augmentations (underwater effects)
        elif augmentation_level == 'marine':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # Simulate underwater light absorption
                    A.ToSepia(p=1.0),  # Simulate greenish/brownish water tint
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0)  # Color shift common underwater
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        else:
            logger.error(f"Unknown augmentation level: {augmentation_level}")
            return None
            
        return transform

    def augment_dataset(self, output_dir: str, augmentation_level: str = 'medium', 
                       augmentation_factor: float = 1.0) -> Dict[str, Any]:
        """
        Create an augmented copy of the dataset.
        
        Args:
            output_dir: Directory to save augmented dataset
            augmentation_level: Level of augmentation (light, medium, heavy, marine)
            augmentation_factor: How many augmented copies per original image
            
        Returns:
            Dictionary with augmentation statistics
        """
        # Create transformation pipeline
        transform = self.create_augmentation_pipeline(augmentation_level)
        
        if transform is None:
            return {
                'error': "Failed to create augmentation pipeline",
                'success': False
            }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(self.dataset_dir, '**', f'*{ext}'), recursive=True))
        
        # Calculate number of augmented copies
        augmentation_copies = max(1, int(augmentation_factor))
        
        results = {
            'original_images': len(image_paths),
            'augmented_images': 0,
            'augmentation_level': augmentation_level,
            'augmentation_factor': augmentation_factor,
            'success': True,
            'errors': []
        }
        
        # Process each image
        for img_path in image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Find annotation file
            ann_filename = f"{img_name}{SUPPORTED_FORMATS['yolo']}"
            
            # Look in common locations
            ann_path = None
            possible_locations = [
                os.path.join(os.path.dirname(img_path).replace('images', 'labels'), ann_filename),
                os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', ann_filename)
            ]
            
            for loc in possible_locations:
                if os.path.exists(loc):
                    ann_path = loc
                    break
            
            if not ann_path:
                results['errors'].append(f"Could not find annotation for {img_path}")
                continue
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    results['errors'].append(f"Could not read image: {img_path}")
                    continue
                
                # Convert to RGB for transformations
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Read annotations
                bboxes = []
                class_labels = []
                
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
                
                # Copy original image and annotation
                shutil.copy2(img_path, os.path.join(output_dir, 'images', os.path.basename(img_path)))
                shutil.copy2(ann_path, os.path.join(output_dir, 'labels', os.path.basename(ann_path)))
                results['augmented_images'] += 1
                
                # Create augmented copies
                for i in range(augmentation_copies):
                    # Apply transformations
                    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    
                    # Save augmented image
                    aug_img_name = f"{img_name}_aug{i+1}{os.path.splitext(img_path)[1]}"
                    aug_img_path = os.path.join(output_dir, 'images', aug_img_name)
                    
                    # Convert back to BGR for saving
                    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Save augmented annotations
                    aug_ann_name = f"{img_name}_aug{i+1}.txt"
                    aug_ann_path = os.path.join(output_dir, 'labels', aug_ann_name)
                    
                    with open(aug_ann_path, 'w') as f:
                        for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    
                    results['augmented_images'] += 1
            
            except Exception as e:
                results['errors'].append(f"Error augmenting {img_path}: {e}")
        
        # Create data.yaml
        yaml_path = os.path.join(output_dir, 'data.yaml')
        
        # Check if we have class names from input dataset
        class_names = {}
        if self.class_names:
            class_names = self.class_names
        else:
            # Try to load from existing yaml
            existing_yaml = glob.glob(os.path.join(self.dataset_dir, '*.yaml'))
            if existing_yaml:
                try:
                    with open(existing_yaml[0], 'r') as f:
                        yaml_data = yaml.safe_load(f)
                        if 'names' in yaml_data:
                            class_names = yaml_data['names']
                except Exception as e:
                    logger.warning(f"Error loading class names from YAML: {e}")
        
        # If still no class names, create generic ones
        if not class_names:
            # Find all class IDs
            class_ids = set()
            label_files = glob.glob(os.path.join(output_dir, 'labels', '*.txt'))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    pass
            
            for class_id in class_ids:
                class_names[class_id] = f"class_{class_id}"
        
        # Write data.yaml
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.join('images'),
            'val': os.path.join('images'),
            'names': class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        results['yaml_path'] = yaml_path
        
        return results

    def create_balanced_subset(self, output_dir: str, target_counts: Dict[int, int] = None, 
                             max_imbalance_ratio: float = 3.0) -> Dict[str, Any]:
        """
        Create a more balanced subset of the dataset.
        
        Args:
            output_dir: Directory to save the balanced dataset
            target_counts: Target count for each class (if None, will balance automatically)
            max_imbalance_ratio: Maximum allowed ratio between most and least common class
            
        Returns:
            Dictionary with statistics about the balanced dataset
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        # First, collect statistics about class distribution
        class_stats = {}
        image_boxes = {}  # Maps image paths to list of (class_id, box) tuples
        
        # Find all annotation files
        annotation_paths = glob.glob(os.path.join(self.dataset_dir, '**', f'*{SUPPORTED_FORMATS["yolo"]}'), recursive=True)
        
        # Process each annotation file
        for ann_path in annotation_paths:
            img_name = os.path.splitext(os.path.basename(ann_path))[0]
            
            # Find corresponding image
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                possible_paths = glob.glob(os.path.join(self.dataset_dir, '**', f"{img_name}{ext}"), recursive=True)
                if possible_paths:
                    img_path = possible_paths[0]
                    break
            
            if not img_path:
                logger.warning(f"No image found for annotation: {ann_path}")
                continue
            
            # Parse annotations
            boxes = []
            try:
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            box = [x_center, y_center, width, height]
                            boxes.append((class_id, box))
                            
                            # Update class statistics
                            if class_id not in class_stats:
                                class_stats[class_id] = 0
                            class_stats[class_id] += 1
            except Exception as e:
                logger.error(f"Error parsing annotation {ann_path}: {e}")
                continue
            
            # Add to image_boxes dict
            if boxes:
                image_boxes[img_path] = {
                    'boxes': boxes,
                    'annotation_path': ann_path
                }
        
        if not class_stats:
            return {
                'error': "No valid annotations found in dataset",
                'success': False
            }
        
        # Calculate target counts if not provided
        if target_counts is None:
            # Find minimum count that ensures max_imbalance_ratio
            min_count = min(class_stats.values())
            max_count = max(class_stats.values())
            
            if max_count / min_count > max_imbalance_ratio:
                # Determine target count based on imbalance ratio
                target = int(min_count * max_imbalance_ratio)
                target_counts = {class_id: min(count, target) for class_id, count in class_stats.items()}
            else:
                # Already balanced enough
                target_counts = class_stats
        
        # Select images to include in the balanced subset
        selected_images = set()
        current_counts = {class_id: 0 for class_id in class_stats}
        
        # Prioritize images with rare classes
        class_priority = sorted(class_stats.keys(), key=lambda c: class_stats[c])
        
        for class_id in class_priority:
            target = target_counts[class_id]
            
            # Skip if we already have enough of this class from other images
            if current_counts[class_id] >= target:
                continue
            
            # Find images containing this class
            class_images = []
            for img_path, data in image_boxes.items():
                if img_path in selected_images:
                    continue
                
                class_count = sum(1 for cid, _ in data['boxes'] if cid == class_id)
                if class_count > 0:
                    class_images.append((img_path, class_count))
            
            # Sort by count (most instances of this class first)
            class_images.sort(key=lambda x: x[1], reverse=True)
            
            # Add images until we reach the target
            for img_path, count in class_images:
                if current_counts[class_id] >= target:
                    break
                
                # Add this image
                selected_images.add(img_path)
                
                # Update counts for all classes in this image
                for cid, _ in image_boxes[img_path]['boxes']:
                    current_counts[cid] = current_counts.get(cid, 0) + 1
        
        # Copy selected images and annotations to output directory
        results = {
            'original_class_counts': class_stats,
            'target_class_counts': target_counts,
            'actual_class_counts': {class_id: 0 for class_id in class_stats},
            'selected_images': len(selected_images),
            'success': True
        }
        
        for img_path in selected_images:
            # Copy image
            dest_img_path = os.path.join(output_dir, 'images', os.path.basename(img_path))
            shutil.copy2(img_path, dest_img_path)
            
            # Copy annotation
            ann_path = image_boxes[img_path]['annotation_path']
            dest_ann_path = os.path.join(output_dir, 'labels', os.path.basename(ann_path))
            shutil.copy2(ann_path, dest_ann_path)
            
            # Update actual class counts
            for class_id, _ in image_boxes[img_path]['boxes']:
                results['actual_class_counts'][class_id] += 1
        
        # Create data.yaml
        yaml_path = os.path.join(output_dir, 'data.yaml')
        
        # Use class names if available
        class_names = {}
        if self.class_names:
            class_names = self.class_names
        else:
            # Try to find existing yaml
            existing_yaml = glob.glob(os.path.join(self.dataset_dir, '*.yaml'))
            if existing_yaml:
                try:
                    with open(existing_yaml[0], 'r') as f:
                        yaml_data = yaml.safe_load(f)
                        if 'names' in yaml_data:
                            class_names = yaml_data['names']
                except Exception as e:
                    logger.warning(f"Error loading class names from YAML: {e}")
        
        # If still no class names, create generic ones
        if not class_names:
            for class_id in class_stats:
                class_names[class_id] = f"class_{class_id}"
        
        # Write data.yaml
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.join('images'),
            'val': os.path.join('images'),
            'names': class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        results['yaml_path'] = yaml_path
        results['class_names'] = class_names
        
        return results