"""Plotting utilities for visualizing data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger('oceancvbench.utils.plotting')

def plot_bbox_distribution(df_bboxes: pd.DataFrame, class_names: Optional[Dict[int, str]] = None,
                          figsize: Tuple[int, int] = (12, 6), 
                          output_path: Optional[str] = None) -> None:
    """
    Creates a bar plot showing the distribution of bounding boxes per class.
    
    Args:
        df_bboxes: DataFrame with bounding box information containing 'class_id' column
        class_names: Optional mapping of class IDs to human-readable names
        figsize: Figure size as (width, height) in inches
        output_path: If provided, save the figure to this path
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
    
    # Count bounding boxes per class
    class_counts = df_bboxes['class_id'].value_counts().sort_index()
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Prepare category labels
    if class_names:
        labels = [class_names.get(int(i), f"Class {i}") for i in class_counts.index]
    else:
        labels = [f"Class {i}" for i in class_counts.index]
    
    # Create the bar plot with a color gradient based on count
    bars = plt.bar(range(len(labels)), class_counts.values, 
                  color=plt.cm.viridis(class_counts.values/max(class_counts.values)))
    
    # Add value labels on top of each bar
    for i, (count, bar) in enumerate(zip(class_counts.values, bars)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Calculate statistics for annotations
    mean_count = class_counts.mean()
    median_count = class_counts.median()
    
    # Add horizontal lines for mean and median
    plt.axhline(y=mean_count, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_count:.1f}')
    plt.axhline(y=median_count, color='g', linestyle='-.', alpha=0.7,
                label=f'Median: {median_count:.1f}')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Bounding Boxes')
    plt.title('Distribution of Bounding Boxes by Class')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bbox distribution plot to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_bbox_size_distribution(df_bboxes: pd.DataFrame, 
                               figsize: Tuple[int, int] = (14, 10),
                               output_path: Optional[str] = None) -> None:
    """
    Creates plots showing the distribution of bounding box dimensions.
    
    Args:
        df_bboxes: DataFrame with bounding box information
        figsize: Figure size as (width, height) in inches
        output_path: If provided, save the figure to this path
    """
    # Check required columns
    for col in ['width', 'height', 'class_id']:
        if col not in df_bboxes.columns:
            # Try to calculate width/height if not present
            if col == 'width' and 'xmin' in df_bboxes.columns and 'xmax' in df_bboxes.columns:
                df_bboxes['width'] = df_bboxes['xmax'] - df_bboxes['xmin']
            elif col == 'height' and 'ymin' in df_bboxes.columns and 'ymax' in df_bboxes.columns:
                df_bboxes['height'] = df_bboxes['ymax'] - df_bboxes['ymin']
            else:
                raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Add area and aspect ratio
    df_bboxes = df_bboxes.copy()
    df_bboxes['area'] = df_bboxes['width'] * df_bboxes['height']
    df_bboxes['aspect_ratio'] = df_bboxes['width'] / df_bboxes['height']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Distribution of widths
    sns.histplot(data=df_bboxes, x='width', hue='class_id', 
                kde=True, element='step', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Bounding Box Widths')
    
    # 2. Distribution of heights
    sns.histplot(data=df_bboxes, x='height', hue='class_id', 
                kde=True, element='step', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Bounding Box Heights')
    
    # 3. Distribution of areas
    sns.histplot(data=df_bboxes, x='area', hue='class_id', 
                kde=True, element='step', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Bounding Box Areas')
    
    # 4. Distribution of aspect ratios
    sns.histplot(data=df_bboxes, x='aspect_ratio', hue='class_id', 
                kde=True, element='step', ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Aspect Ratios')
    axes[1, 1].set_xlim(0, 3)  # Limit to reasonable aspect ratios
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bbox size distribution plots to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_detection_visualization(image: np.ndarray, 
                               boxes: List[Tuple[int, int, int, int]],
                               classes: List[int],
                               scores: Optional[List[float]] = None,
                               class_names: Optional[Dict[int, str]] = None,
                               figsize: Tuple[int, int] = (12, 8),
                               output_path: Optional[str] = None) -> None:
    """
    Visualizes object detections on an image.
    
    Args:
        image: Input image as numpy array (BGR or RGB format)
        boxes: List of bounding boxes as (xmin, ymin, xmax, ymax)
        classes: List of class IDs corresponding to each box
        scores: Optional list of confidence scores
        class_names: Optional mapping of class IDs to human-readable names
        figsize: Figure size as (width, height) in inches
        output_path: If provided, save the figure to this path
    """
    import cv2
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Check if image is BGR (OpenCV default) and convert to RGB for matplotlib
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Generate random colors for each class
    np.random.seed(42)  # for reproducibility
    colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in set(classes)}
    
    # Draw each bounding box
    for i, (box, class_id) in enumerate(zip(boxes, classes)):
        xmin, ymin, xmax, ymax = box
        
        # Get color for this class
        color = colors[class_id]
        # BGR for OpenCV functions
        color_bgr = (color[2], color[1], color[0])
        
        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bgr, 2)
        
        # Prepare label text
        if class_names and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
            
        if scores:
            label = f"{label}: {scores[i]:.2f}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (xmin, ymin - text_size[1] - 10), (xmin + text_size[0], ymin), color_bgr, -1)
        
        # Draw label text
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Detection Results: {len(boxes)} objects")
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved detection visualization to {output_path}")
    else:
        plt.show()
    plt.close()
