"""Object detection and localization functions."""

import os
import pandas as pd
import numpy as np
import glob
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
from .yolo_integration import load_yolo_model

# Set up logging
logger = logging.getLogger('oceancvbench.inference.localize')


def localize_images(folder: str, model_path: str, iou_thresh: float = 0.5, 
                   conf_thresh: float = 0.4, csv: bool = False,
                   save_images: bool = False) -> pd.DataFrame:
    """
    Run object detection on all images in a folder.
    
    Args:
        folder: Path to folder containing images
        model_path: Path to YOLO model file
        iou_thresh: IoU threshold for non-maximum suppression
        conf_thresh: Confidence threshold for detections
        csv: If True, save results to a CSV file
        save_images: If True, save images with detection overlays
        
    Returns:
        DataFrame containing detection results
    """
    # Load the model
    model = load_yolo_model(model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    if model is None:
        logger.error("Failed to load model")
        return pd.DataFrame()
    
    # Find all images in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
        # Also check in images subfolder if exists
        subfolder = os.path.join(folder, 'images')
        if os.path.exists(subfolder):
            image_paths.extend(glob.glob(os.path.join(subfolder, ext)))
    
    if not image_paths:
        logger.warning(f"No images found in {folder}")
        return pd.DataFrame()
    
    # Prepare the results DataFrame
    results_data = []
    
    # Create output directory for visualizations if needed
    if save_images:
        vis_dir = os.path.join(folder, 'detections')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Localizing objects"):
        try:
            # Run inference
            detections = model(img_path)
            
            # Get the filename
            filename = os.path.basename(img_path)
            
            # Check if the model returns predictions
            boxes_detected = False
            
            # Process the results
            for det in detections:
                boxes = det.boxes
                # Get original image dimensions
                img_width, img_height = det.orig_shape[1], det.orig_shape[0]
                
                if len(boxes) > 0:
                    boxes_detected = True
                
                for i in range(len(boxes)):
                    # Get box data
                    box = boxes[i].xyxy[0].cpu().numpy()  # xyxy format (x1, y1, x2, y2)
                    xmin, ymin, xmax, ymax = box
                    conf = float(boxes[i].conf)
                    class_id = int(boxes[i].cls)
                    
                    # Add to results
                    results_data.append({
                        'filename': filename,
                        'class_id': class_id,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'width': xmax - xmin,
                        'height': ymax - ymin,
                        'confidence': conf,
                        'img_width': img_width,
                        'img_height': img_height,
                        'image_path': img_path
                    })
            
            # Save visualization if requested
            if save_images and boxes_detected:
                # Load the original image
                img = cv2.imread(img_path)
                
                # Draw boxes on the image
                for result in results_data:
                    if result['filename'] == filename:
                        # Draw rectangle
                        cv2.rectangle(img, 
                                     (int(result['xmin']), int(result['ymin'])), 
                                     (int(result['xmax']), int(result['ymax'])), 
                                     (0, 255, 0), 2)
                        
                        # Add label with class and confidence
                        label = f"Class {result['class_id']}: {result['confidence']:.2f}"
                        cv2.putText(img, label, 
                                   (int(result['xmin']), int(result['ymin'] - 10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save the image
                output_path = os.path.join(vis_dir, f"det_{filename}")
                cv2.imwrite(output_path, img)
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results_data)
    
    # Save to CSV if requested
    if csv and not df_results.empty:
        csv_path = os.path.join(folder, "localize_results.csv")
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
    
    return df_results


def batch_localize(image_folders: List[str], model_path: str, 
                  output_dir: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Run localization on multiple folders of images.
    
    Args:
        image_folders: List of folder paths containing images
        model_path: Path to YOLO model file
        output_dir: Directory to save all results (if None, save in each folder)
        **kwargs: Additional arguments to pass to localize_images
        
    Returns:
        Dictionary mapping folder names to result DataFrames
    """
    results = {}
    
    for folder in image_folders:
        folder_name = os.path.basename(folder)
        logger.info(f"Processing folder: {folder_name}")
        
        # Run localization
        df = localize_images(folder, model_path, **kwargs)
        
        # Store results
        results[folder_name] = df
        
        # Save to output directory if specified
        if output_dir and not df.empty:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{folder_name}_results.csv")
            df.to_csv(output_path, index=False)
    
    return results


def export_coco_format(df_results: pd.DataFrame, output_path: str,
                      class_names: Dict[int, str] = None) -> str:
    """
    Export detection results to COCO format JSON.
    
    Args:
        df_results: DataFrame with detection results
        output_path: Path to save the JSON file
        class_names: Optional mapping of class IDs to names
        
    Returns:
        Path to the saved JSON file
    """
    import json
    from datetime import datetime
    
    if df_results.empty:
        logger.warning("Empty results DataFrame, cannot export to COCO format")
        return None
    
    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "Exported from OceanCVBench",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "OceanCVBench",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    unique_classes = df_results['class_id'].unique()
    for i, class_id in enumerate(unique_classes):
        name = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
        coco_output["categories"].append({
            "id": int(class_id),
            "name": name,
            "supercategory": "none"
        })
    
    # Process each unique image
    image_id_map = {}  # Maps filename to COCO image ID
    annotation_id = 1
    
    for image_index, (filename, group) in enumerate(df_results.groupby('filename')):
        # Get image info from the first detection in this image
        first_det = group.iloc[0]
        img_width = first_det.get('img_width', 640)
        img_height = first_det.get('img_height', 480)
        
        # Add image entry
        image_id = image_index + 1  # COCO image IDs start from 1
        image_id_map[filename] = image_id
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": int(img_width),
            "height": int(img_height),
            "license": 1,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Add annotations for this image
        for _, det in group.iterrows():
            x = float(det['xmin'])
            y = float(det['ymin'])
            w = float(det['width'])
            h = float(det['height'])
            
            # COCO uses [x,y,width,height] format
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(det['class_id']),
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0,
                "score": float(det['confidence'])
            })
            
            annotation_id += 1
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    logger.info(f"Exported {len(coco_output['annotations'])} annotations to {output_path}")
    return output_path
