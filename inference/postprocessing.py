"""
Postprocessing module for OceanCVBench.

This module provides functions for processing inference results,
including filtering detections, adjusting coordinates, and visualization.
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logger = logging.getLogger("oceancvbench.inference.postprocessing")

def apply_confidence_filter(detections: List[Dict[str, Any]], 
                          threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Filter detections by confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        threshold: Confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.get('confidence', 0) >= threshold]


def calculate_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates
    x1_min, y1_min = box1.get('xmin', 0), box1.get('ymin', 0)
    x1_max, y1_max = box1.get('xmax', 0), box1.get('ymax', 0)
    
    x2_min, y2_min = box2.get('xmin', 0), box2.get('ymin', 0)
    x2_max, y2_max = box2.get('xmax', 0), box2.get('ymax', 0)
    
    # Calculate intersection area
    x_intersection_min = max(x1_min, x2_min)
    y_intersection_min = max(y1_min, y2_min)
    x_intersection_max = min(x1_max, x2_max)
    y_intersection_max = min(y1_max, y2_max)
    
    # Check if there is an intersection
    if x_intersection_max <= x_intersection_min or y_intersection_max <= y_intersection_min:
        return 0.0
    
    intersection_area = (x_intersection_max - x_intersection_min) * (y_intersection_max - y_intersection_min)
    
    # Calculate areas of both boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def non_max_suppression(detections: List[Dict[str, Any]], iou_threshold: float = 0.5,
                      class_specific: bool = True) -> List[Dict[str, Any]]:
    """
    Apply non-maximum suppression to remove duplicate detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering boxes as duplicates
        class_specific: Whether to apply NMS per class or across all classes
        
    Returns:
        List of detections after NMS
    """
    if not detections:
        return []
    
    # Sort detections by confidence (descending)
    sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    selected_detections = []
    
    # If class-specific, group detections by class
    if class_specific:
        # Group by class
        class_detections = {}
        for det in sorted_detections:
            class_id = det.get('class_id', -1)
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)
        
        # Apply NMS for each class separately
        for class_id, class_dets in class_detections.items():
            class_selected = []
            
            while class_dets:
                # Select detection with highest confidence
                best_det = class_dets.pop(0)
                class_selected.append(best_det)
                
                # Filter out overlapping detections
                class_dets = [det for det in class_dets if calculate_iou(best_det, det) < iou_threshold]
                
            # Add selected detections for this class
            selected_detections.extend(class_selected)
    else:
        # Apply NMS across all classes
        while sorted_detections:
            # Select detection with highest confidence
            best_det = sorted_detections.pop(0)
            selected_detections.append(best_det)
            
            # Filter out overlapping detections
            sorted_detections = [det for det in sorted_detections if calculate_iou(best_det, det) < iou_threshold]
    
    return selected_detections


def adjust_tile_coordinates(tile_detections: List[Dict[str, Any]], 
                          tile_x: int, tile_y: int) -> List[Dict[str, Any]]:
    """
    Adjust detection coordinates from a tile to the original image.
    
    Args:
        tile_detections: List of detections in tile coordinates
        tile_x: X-coordinate of the tile in the original image
        tile_y: Y-coordinate of the tile in the original image
        
    Returns:
        List of detections with adjusted coordinates
    """
    adjusted_detections = []
    
    for det in tile_detections:
        # Create a copy to avoid modifying the original
        adjusted = det.copy()
        
        # Adjust coordinates
        adjusted['xmin'] += tile_x
        adjusted['ymin'] += tile_y
        adjusted['xmax'] += tile_x
        adjusted['ymax'] += tile_y
        
        # Add tile info
        adjusted['tile_x'] = tile_x
        adjusted['tile_y'] = tile_y
        
        adjusted_detections.append(adjusted)
    
    return adjusted_detections


def merge_tile_detections(all_tile_detections: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge detections from multiple tiles and remove duplicates.
    
    Args:
        all_tile_detections: List of detection lists from each tile
        
    Returns:
        Merged list of detections after NMS
    """
    # Flatten the list of lists
    flattened_detections = []
    for tile_detections in all_tile_detections:
        flattened_detections.extend(tile_detections)
    
    # Apply NMS to remove duplicates
    merged_detections = non_max_suppression(flattened_detections, iou_threshold=0.5, class_specific=True)
    
    return merged_detections


def scale_back_coordinates(detections: List[Dict[str, Any]], scale_factor: float,
                          padding: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Scale detection coordinates back to the original image dimensions.
    
    Args:
        detections: List of detection dictionaries
        scale_factor: Scale factor used during preprocessing
        padding: Padding used during preprocessing (x_offset, y_offset)
        
    Returns:
        List of detections with scaled coordinates
    """
    if scale_factor == 0:
        logger.error("Invalid scale factor (0)")
        return detections
    
    scaled_detections = []
    
    for det in detections:
        # Create a copy to avoid modifying the original
        scaled = det.copy()
        
        # Adjust for padding
        x_offset, y_offset = padding
        
        # Scale back coordinates
        scaled['xmin'] = (scaled['xmin'] - x_offset) / scale_factor
        scaled['ymin'] = (scaled['ymin'] - y_offset) / scale_factor
        scaled['xmax'] = (scaled['xmax'] - x_offset) / scale_factor
        scaled['ymax'] = (scaled['ymax'] - y_offset) / scale_factor
        
        # Update width and height
        scaled['width'] = scaled['xmax'] - scaled['xmin']
        scaled['height'] = scaled['ymax'] - scaled['ymin']
        
        # Update area
        scaled['area'] = scaled['width'] * scaled['height']
        
        scaled_detections.append(scaled)
    
    return scaled_detections


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert detection results to a pandas DataFrame.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        DataFrame with detection results
    """
    return pd.DataFrame(detections)


def save_detections_to_csv(detections: List[Dict[str, Any]], output_path: str) -> str:
    """
    Save detection results to a CSV file.
    
    Args:
        detections: List of detection dictionaries
        output_path: Path to save the CSV file
        
    Returns:
        Path to the saved file
    """
    df = detections_to_dataframe(detections)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(detections)} detections to {output_path}")
    
    return output_path


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], 
                  score_threshold: float = 0.0, thickness: int = 2, 
                  show_labels: bool = True) -> np.ndarray:
    """
    Draw detection boxes and labels on an image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries
        score_threshold: Minimum confidence score to display
        thickness: Line thickness for boxes
        show_labels: Whether to display labels
        
    Returns:
        Image with drawn detections
    """
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Filter by threshold
    detections = [d for d in detections if d.get('confidence', 0) >= score_threshold]
    
    # Generate distinct colors for different classes
    class_ids = set(d.get('class_id', 0) for d in detections)
    colors = {}
    
    for i, class_id in enumerate(class_ids):
        # Use different hue values with high saturation and value
        hue = (i * 30) % 180  # Different hues, spaced by 30 degrees
        colors[class_id] = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
    
    # Draw each detection
    for det in detections:
        # Get coordinates
        xmin = int(det.get('xmin', 0))
        ymin = int(det.get('ymin', 0))
        xmax = int(det.get('xmax', 0))
        ymax = int(det.get('ymax', 0))
        
        # Get class info
        class_id = det.get('class_id', 0)
        class_name = det.get('class_name', f"Class {class_id}")
        conf = det.get('confidence', 0)
        
        # Get color for this class
        color = colors.get(class_id, (0, 255, 0))  # Default to green
        
        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Draw label if requested
        if show_labels:
            label = f"{class_name}: {conf:.2f}"
            
            # Get size of text for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(img, (xmin, ymin - text_height - 10), (xmin + text_width + 10, ymin), color, -1)
            
            # Draw text
            cv2.putText(img, label, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def save_visualization(image: np.ndarray, detections: List[Dict[str, Any]], 
                     output_path: str, score_threshold: float = 0.0) -> str:
    """
    Draw detections on image and save to file.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries
        output_path: Path to save the visualization
        score_threshold: Minimum confidence score to display
        
    Returns:
        Path to the saved image
    """
    # Draw detections
    vis_img = draw_detections(image, detections, score_threshold)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, vis_img)
    logger.info(f"Saved visualization to {output_path}")
    
    return output_path


def convert_to_yolo_format(detections: List[Dict[str, Any]], 
                         output_dir: str, 
                         create_missing_images: bool = False) -> Dict[str, List[str]]:
    """
    Convert detections to YOLO format and save as txt files.
    
    Args:
        detections: List of detection dictionaries
        output_dir: Directory to save the YOLO format files
        create_missing_images: Whether to copy images if not found in output dir
        
    Returns:
        Dictionary with paths to created label files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group detections by image
    image_detections = {}
    for det in detections:
        image_path = det.get('image_path')
        if not image_path:
            continue
            
        if image_path not in image_detections:
            image_detections[image_path] = []
            
        image_detections[image_path].append(det)
    
    # Create labels directory
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # If copying images, create images directory
    if create_missing_images:
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
    
    # Process each image
    created_files = {
        'labels': [],
        'images': []
    }
    
    for image_path, dets in image_detections.items():
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create YOLO format label file
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Get image dimensions (from first detection)
        if dets:
            img_width = dets[0].get('img_width', 640)
            img_height = dets[0].get('img_height', 480)
        else:
            # If no detections, try to get dimensions from image
            try:
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]
            except:
                img_width, img_height = 640, 480  # Default fallback
        
        # Convert all detections to YOLO format and write to file
        with open(label_path, 'w') as f:
            for det in dets:
                class_id = det.get('class_id', 0)
                
                # Extract coordinates
                xmin = det.get('xmin', 0)
                ymin = det.get('ymin', 0)
                width = det.get('width', 0)
                height = det.get('height', 0)
                
                # Convert to normalized YOLO format (center_x, center_y, width, height)
                center_x = (xmin + width/2) / img_width
                center_y = (ymin + height/2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Clip values to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # Write to file
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        created_files['labels'].append(label_path)
        
        # Copy image if requested
        if create_missing_images:
            img_output_path = os.path.join(images_dir, os.path.basename(image_path))
            
            if not os.path.exists(img_output_path):
                try:
                    # Copy the image
                    import shutil
                    shutil.copy2(image_path, img_output_path)
                    created_files['images'].append(img_output_path)
                except Exception as e:
                    logger.error(f"Error copying image {image_path}: {e}")
    
    return created_files


def visualize_batch_results(detections: List[Dict[str, Any]], 
                          output_dir: str, 
                          conf_threshold: float = 0.25) -> Dict[str, str]:
    """
    Create visualization for a batch of detection results.
    
    Args:
        detections: List of detection dictionaries
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold for displaying detections
        
    Returns:
        Dictionary mapping image paths to visualization paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group detections by image
    image_detections = {}
    for det in detections:
        image_path = det.get('image_path')
        if not image_path:
            continue
            
        if image_path not in image_detections:
            image_detections[image_path] = []
            
        image_detections[image_path].append(det)
    
    # Create visualizations for each image
    visualization_paths = {}
    
    for image_path, dets in image_detections.items():
        try:
            # Load the image
            img = cv2.imread(image_path)
            
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                continue
                
            # Create output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
            
            # Draw detections and save
            vis_path = save_visualization(img, dets, vis_path, conf_threshold)
            
            visualization_paths[image_path] = vis_path
            
        except Exception as e:
            logger.error(f"Error creating visualization for {image_path}: {e}")
    
    return visualization_paths


def create_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of detection results.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {
            'total_detections': 0,
            'unique_images': 0,
            'classes': {},
            'confidence': {'min': 0, 'max': 0, 'mean': 0}
        }
    
    # Convert to DataFrame for easier analysis
    df = detections_to_dataframe(detections)
    
    # Count by class
    class_counts = df['class_id'].value_counts().to_dict()
    
    # Get class names if available
    class_names = {}
    for det in detections:
        class_id = det.get('class_id')
        class_name = det.get('class_name')
        if class_id is not None and class_name is not None:
            class_names[class_id] = class_name
    
    # Create class info with names
    class_info = {}
    for class_id, count in class_counts.items():
        name = class_names.get(class_id, f"Class {class_id}")
        class_info[int(class_id)] = {
            'name': name,
            'count': count,
            'percentage': count / len(df) * 100
        }
    
    # Compute confidence statistics
    confidence_stats = {
        'min': float(df['confidence'].min()),
        'max': float(df['confidence'].max()),
        'mean': float(df['confidence'].mean()),
        'median': float(df['confidence'].median())
    }
    
    # Count unique images
    unique_images = df['image_path'].nunique() if 'image_path' in df.columns else 0
    
    # Size statistics
    if 'width' in df.columns and 'height' in df.columns:
        size_stats = {
            'width': {
                'min': float(df['width'].min()),
                'max': float(df['width'].max()),
                'mean': float(df['width'].mean())
            },
            'height': {
                'min': float(df['height'].min()),
                'max': float(df['height'].max()),
                'mean': float(df['height'].mean())
            },
            'area': {
                'min': float(df['width'] * df['height']).min(),
                'max': float(df['width'] * df['height']).max(),
                'mean': float(df['width'] * df['height']).mean()
            }
        }
    else:
        size_stats = {}
    
    return {
        'total_detections': len(df),
        'unique_images': unique_images,
        'classes': class_info,
        'confidence': confidence_stats,
        'size': size_stats
    }


def create_detection_report(detections: List[Dict[str, Any]], 
                          output_dir: str,
                          include_visualizations: bool = True,
                          conf_threshold: float = 0.25) -> str:
    """
    Create a comprehensive detection report with statistics and visualizations.
    
    Args:
        detections: List of detection dictionaries
        output_dir: Directory to save report files
        include_visualizations: Whether to include visualizations
        conf_threshold: Confidence threshold for displaying detections
        
    Returns:
        Path to the HTML report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics
    summary = create_detection_summary(detections)
    
    # Save detections to CSV
    csv_path = os.path.join(output_dir, "detections.csv")
    save_detections_to_csv(detections, csv_path)
    
    # Create visualizations if requested
    vis_paths = {}
    if include_visualizations:
        vis_dir = os.path.join(output_dir, "visualizations")
        vis_paths = visualize_batch_results(detections, vis_dir, conf_threshold)
    
    # Create class distribution plot
    plt.figure(figsize=(10, 6))
    
    # Sort classes by count
    sorted_classes = sorted(
        summary['classes'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Extract data for plotting
    class_names = [f"{class_info['name']} (ID:{class_id})" for class_id, class_info in sorted_classes]
    class_counts = [class_info['count'] for _, class_info in sorted_classes]
    
    # Create bar chart
    bars = plt.bar(range(len(class_names)), class_counts, color='skyblue')
    
    # Add value labels on top of bars
    for i, count in enumerate(class_counts):
        plt.text(i, count + 1, str(count), ha='center')
    
    # Set labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Detection Counts by Class')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Create confidence distribution plot
    if detections:
        plt.figure(figsize=(10, 6))
        
        # Get confidence values
        confidences = [d.get('confidence', 0) for d in detections]
        
        # Create histogram
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue')
        
        # Set labels and title
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Distribution of Detection Confidence Scores')
        plt.axvline(x=conf_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({conf_threshold})')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        conf_plot_path = os.path.join(output_dir, "confidence_distribution.png")
        plt.savefig(conf_plot_path)
        plt.close()
    
    # Create HTML report
    html_path = os.path.join(output_dir, "report.html")
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
                .metric { margin: 5px 0; }
                .vis-container { display: flex; flex-wrap: wrap; gap: 10px; }
                .vis-item { margin-bottom: 20px; }
                img { max-width: 100%; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Object Detection Report</h1>
        """)
        
        # Summary section
        f.write("<div class='summary'>")
        f.write("<h2>Summary</h2>")
        f.write(f"<div class='metric'><b>Total Detections:</b> {summary['total_detections']}</div>")
        f.write(f"<div class='metric'><b>Unique Images:</b> {summary['unique_images']}</div>")
        f.write(f"<div class='metric'><b>Confidence Threshold:</b> {conf_threshold}</div>")
        f.write(f"<div class='metric'><b>Average Confidence:</b> {summary['confidence']['mean']:.3f}</div>")
        f.write(f"<div class='metric'><b>Number of Classes:</b> {len(summary['classes'])}</div>")
        f.write("</div>")
        
        # Class distribution plot
        f.write("<h2>Class Distribution</h2>")
        f.write(f"<img src='{os.path.relpath(plot_path, output_dir)}' alt='Class Distribution'>")
        
        # Confidence distribution plot
        if detections:
            f.write("<h2>Confidence Distribution</h2>")
            f.write(f"<img src='{os.path.relpath(conf_plot_path, output_dir)}' alt='Confidence Distribution'>")
        
        # Class details table
        f.write("<h2>Class Details</h2>")
        f.write("<table>")
        f.write("<tr><th>Class ID</th><th>Class Name</th><th>Count</th><th>Percentage</th></tr>")
        
        for class_id, info in sorted(summary['classes'].items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"<tr>")
            f.write(f"<td>{class_id}</td>")
            f.write(f"<td>{info['name']}</td>")
            f.write(f"<td>{info['count']}</td>")
            f.write(f"<td>{info['percentage']:.1f}%</td>")
            f.write(f"</tr>")
            
        f.write("</table>")
        
        # Visualizations section
        if include_visualizations and vis_paths:
            f.write("<h2>Sample Visualizations</h2>")
            f.write("<div class='vis-container'>")
            
            # Limit to a reasonable number
            sample_vis = list(vis_paths.items())[:20]
            
            for img_path, vis_path in sample_vis:
                img_name = os.path.basename(img_path)
                rel_vis_path = os.path.relpath(vis_path, output_dir)
                
                f.write("<div class='vis-item'>")
                f.write(f"<h4>{img_name}</h4>")
                f.write(f"<img src='{rel_vis_path}' alt='Detection visualization'>")
                f.write("</div>")
                
            f.write("</div>")
            
            if len(vis_paths) > 20:
                f.write(f"<p>Showing 20 of {len(vis_paths)} visualizations.</p>")
        
        # Footer with metadata
        f.write("<hr>")
        f.write("<footer>")
        f.write(f"<p>Report generated with OceanCVBench on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>")
        f.write("</footer>")
        
        f.write("</body></html>")
    
    logger.info(f"Created detection report at {html_path}")
    return html_path
