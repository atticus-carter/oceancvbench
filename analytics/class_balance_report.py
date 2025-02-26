"""Tools for analyzing class balance in datasets."""

import os
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from pathlib import Path
import cv2

# Set up logging
logger = logging.getLogger('oceancvbench.analytics.class_balance')

def _parse_annotation_file(annotation_path: str) -> List[Dict[str, float]]:
    """Parse YOLO annotation file and extract bounding boxes."""
    try:
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
            
        boxes = []
        for ann in annotations:
            parts = ann.strip().split()
            
            if len(parts) >= 5:
                # YOLO format: class_id center_x center_y width height
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                boxes.append({
                    'class_id': class_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
                
        return boxes
    except Exception as e:
        logger.error(f"Error parsing annotation file {annotation_path}: {e}")
        return []


def _get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get the dimensions of an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 640, 480  # Default if can't read image
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        logger.error(f"Error getting dimensions for {image_path}: {e}")
        return 640, 480  # Default fallback size


def _generate_spatial_heatmap(boxes_df: pd.DataFrame, img_width: int = 640, img_height: int = 480, 
                            by_class: bool = True) -> Dict[str, np.ndarray]:
    """Generate heatmaps showing the spatial distribution of objects."""
    
    # Create empty heatmaps
    resolution = (img_height // 10, img_width // 10)  # Lower resolution for heatmap
    
    if by_class:
        unique_classes = boxes_df['class_id'].unique()
        heatmaps = {
            f"class_{class_id}": np.zeros(resolution)
            for class_id in unique_classes
        }
        heatmaps["all_classes"] = np.zeros(resolution)
    else:
        heatmaps = {"all_classes": np.zeros(resolution)}
    
    # For each box, update the heatmaps
    for _, box in boxes_df.iterrows():
        # Convert normalized coordinates to pixel values
        center_x_pixel = int(box['center_x'] * img_width)
        center_y_pixel = int(box['center_y'] * img_height)
        
        # Map to heatmap resolution
        hm_x = min(int(center_x_pixel / img_width * resolution[1]), resolution[1] - 1)
        hm_y = min(int(center_y_pixel / img_height * resolution[0]), resolution[0] - 1)
        
        # Update the "all classes" heatmap
        heatmaps["all_classes"][hm_y, hm_x] += 1
        
        # Update the class-specific heatmap
        if by_class:
            class_key = f"class_{box['class_id']}"
            heatmaps[class_key][hm_y, hm_x] += 1
    
    return heatmaps


def _plot_spatial_heatmaps(heatmaps: Dict[str, np.ndarray], class_names: Dict[int, str] = None,
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """Create plots from spatial heatmaps."""
    num_plots = len(heatmaps)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Make it indexable
    
    axes = axes.flatten()
    
    for i, (name, heatmap) in enumerate(heatmaps.items()):
        if i >= len(axes):
            break
            
        sns.heatmap(heatmap, ax=axes[i], cmap='viridis')
        
        title = name
        if class_names and name.startswith("class_"):
            class_id = int(name.split("_")[1])
            if class_id in class_names:
                title = class_names[class_id]
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(heatmaps), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def _format_class_name(class_id: int, class_names: Optional[Dict[int, str]]) -> str:
    """Format class name using lookup dictionary if available."""
    if class_names and class_id in class_names:
        return f"{class_id}: {class_names[class_id]}"
    return f"Class {class_id}"


def CBR(data_yaml_path: str, over_under_threshold: float = 0.05, 
        show_heatmap: bool = True) -> Dict[str, Any]:
    """
    Generate a comprehensive Class Balance Report.
    
    Args:
        data_yaml_path: Path to YOLO data.yaml file
        over_under_threshold: Threshold for identifying over/under-represented classes
        show_heatmap: Whether to display a heatmap of object positions
        
    Returns:
        Dictionary containing comprehensive class balance statistics
    """
    # Load the YAML file
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file {data_yaml_path}: {e}")
        return {}
    
    # Extract paths and class names
    train_dir = data_config.get('train', '')
    val_dir = data_config.get('val', '')
    class_names = data_config.get('names', {})
    
    # Convert class names to int keys if needed
    class_names = {int(k): v for k, v in class_names.items()} if class_names else {}
    
    # Check if paths exist
    if not os.path.exists(train_dir):
        logger.error(f"Train directory not found: {train_dir}")
        return {}
    
    # Initialize results
    results = {
        'class_counts': {},
        'class_sizes': {},
        'spatial_distribution': {},
        'dataset_stats': {},
        'recommendations': []
    }
    
    # Process datasets
    all_bboxes = []
    
    # Function to process a dataset directory
    def process_directory(directory, dataset_name):
        if not os.path.exists(directory):
            logger.warning(f"{dataset_name} directory not found: {directory}")
            return []
            
        # Check directory structure
        labels_dir = os.path.join(directory, 'labels')
        images_dir = os.path.join(directory, 'images')
        
        if not os.path.exists(labels_dir):
            # Check if labels are directly in the directory
            labels_dir = directory
            images_dir = directory
            
            if not any(f.endswith('.txt') for f in os.listdir(directory)):
                logger.warning(f"No label files found in {directory}")
                return []
        
        # Find all label files
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        logger.info(f"Found {len(label_files)} label files in {dataset_name} dataset")
        
        bboxes_data = []
        
        # Process each label file
        for label_file in tqdm(label_files, desc=f"Processing {dataset_name} labels"):
            # Get corresponding image file
            base_name = os.path.basename(label_file).rsplit('.', 1)[0]
            
            # Look for image with common extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    image_path = img_path
                    break
            
            # Get boxes from the annotation file
            boxes = _parse_annotation_file(label_file)
            
            # If we found boxes and an image, add to results
            if boxes and image_path:
                img_width, img_height = _get_image_dimensions(image_path)
                
                # Add image info to each box
                for box in boxes:
                    box_data = box.copy()
                    box_data['filename'] = os.path.basename(image_path)
                    box_data['dataset'] = dataset_name
                    box_data['img_width'] = img_width
                    box_data['img_height'] = img_height
                    
                    # Convert normalized YOLO coordinates to pixel values
                    box_data['width_px'] = box['width'] * img_width
                    box_data['height_px'] = box['height'] * img_height
                    box_data['xmin'] = (box['center_x'] - box['width'] / 2) * img_width
                    box_data['ymin'] = (box['center_y'] - box['height'] / 2) * img_height
                    box_data['xmax'] = (box['center_x'] + box['width'] / 2) * img_width
                    box_data['ymax'] = (box['center_y'] + box['height'] / 2) * img_height
                    
                    # Calculate area (both normalized and pixels)
                    box_data['area_norm'] = box['width'] * box['height']
                    box_data['area_px'] = box_data['width_px'] * box_data['height_px']
                    
                    # Calculate aspect ratio
                    if box['height'] > 0:
                        box_data['aspect_ratio'] = box['width'] / box['height']
                    else:
                        box_data['aspect_ratio'] = 1.0
                        
                    bboxes_data.append(box_data)
                    
        return bboxes_data
    
    # Process train dataset
    train_bboxes = process_directory(train_dir, 'train')
    all_bboxes.extend(train_bboxes)
    
    # Process val dataset if provided
    if val_dir:
        val_bboxes = process_directory(val_dir, 'val')
        all_bboxes.extend(val_bboxes)
    
    # Convert to DataFrame for easier analysis
    df_bboxes = pd.DataFrame(all_bboxes)
    
    # If no bounding boxes found, return empty results
    if df_bboxes.empty:
        logger.error("No valid bounding boxes found in the dataset")
        return results
    
    # 1. Class counts
    class_counts = df_bboxes['class_id'].value_counts().sort_index()
    total_classes = len(class_counts)
    total_boxes = len(df_bboxes)
    
    # Calculate class percentages
    class_percentages = (class_counts / total_boxes * 100).round(2)
    
    # Store in results
    results['class_counts'] = class_counts.to_dict()
    results['class_percentages'] = class_percentages.to_dict()
    
    # 2. Class size statistics
    class_size_stats = {}
    for class_id, group in df_bboxes.groupby('class_id'):
        stats = {
            'count': len(group),
            'width_px': {
                'mean': group['width_px'].mean(),
                'median': group['width_px'].median(),
                'min': group['width_px'].min(),
                'max': group['width_px'].max(),
                'std': group['width_px'].std()
            },
            'height_px': {
                'mean': group['height_px'].mean(),
                'median': group['height_px'].median(),
                'min': group['height_px'].min(),
                'max': group['height_px'].max(),
                'std': group['height_px'].std()
            },
            'area_px': {
                'mean': group['area_px'].mean(),
                'median': group['area_px'].median(),
                'min': group['area_px'].min(),
                'max': group['area_px'].max(),
                'std': group['area_px'].std()
            },
            'aspect_ratio': {
                'mean': group['aspect_ratio'].mean(),
                'median': group['aspect_ratio'].median(),
                'min': group['aspect_ratio'].min(),
                'max': group['aspect_ratio'].max(),
                'std': group['aspect_ratio'].std()
            }
        }
        class_size_stats[int(class_id)] = stats
    
    results['class_sizes'] = class_size_stats
    
    # 3. Identify class balance issues
    avg_count = total_boxes / total_classes
    
    # Calculate Gini coefficient (measure of inequality)
    sorted_counts = sorted(class_counts)
    cumulative = np.cumsum(sorted_counts)
    lorenz_curve = cumulative / cumulative[-1]
    n = len(sorted_counts)
    gini = 1 - 2 * np.trapz(lorenz_curve, dx=1/n)
    
    # Calculate coefficient of variation (std / mean)
    cv = class_counts.std() / class_counts.mean()
    
    # Identify over/under-represented classes
    over_threshold = avg_count * (1 + over_under_threshold)
    under_threshold = avg_count * (1 - over_under_threshold)
    
    overrepresented = class_counts[class_counts > over_threshold].index.tolist()
    underrepresented = class_counts[class_counts < under_threshold].index.tolist()
    
    results['dataset_stats'] = {
        'total_images': df_bboxes['filename'].nunique(),
        'total_bounding_boxes': total_boxes,
        'total_classes': total_classes,
        'average_boxes_per_class': avg_count,
        'coefficient_of_variation': cv,
        'gini_coefficient': gini,
        'max_to_min_ratio': class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf'),
        'overrepresented_classes': overrepresented,
        'underrepresented_classes': underrepresented
    }
    
    # 4. Spatial distribution analysis
    if show_heatmap:
        try:
            heatmaps = _generate_spatial_heatmap(df_bboxes)
            fig = _plot_spatial_heatmaps(heatmaps, class_names)
            
            # Save the figure to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
                results['spatial_heatmap_path'] = tmp.name
                
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating spatial heatmap: {e}")
    
    # 5. Generate recommendations
    recommendations = []
    
    # Check for severe class imbalance
    if gini > 0.4:
        recommendations.append("The dataset has significant class imbalance (high Gini coefficient). "
                              "Consider balancing techniques.")
    
    # Check for underrepresented classes
    if underrepresented:
        classes_str = ", ".join([_format_class_name(c, class_names) for c in underrepresented[:5]])
        if len(underrepresented) > 5:
            classes_str += f" and {len(underrepresented) - 5} more"
        
        recommendations.append(f"These classes are underrepresented: {classes_str}. "
                              "Consider collecting more data or using augmentation techniques.")
    
    # Check for overrepresented classes
    if overrepresented and gini > 0.3:
        classes_str = ", ".join([_format_class_name(c, class_names) for c in overrepresented[:5]])
        if len(overrepresented) > 5:
            classes_str += f" and {len(overrepresented) - 5} more"
        
        recommendations.append(f"These classes are overrepresented: {classes_str}. "
                              "Consider downsampling or adjusting sampling weights during training.")
    
    # Check for small bounding boxes
    small_boxes_threshold = 32 * 32  # 32x32 pixels
    small_box_classes = []
    
    for class_id, stats in class_size_stats.items():
        if stats['area_px']['mean'] < small_boxes_threshold:
            small_box_classes.append(class_id)
    
    if small_box_classes:
        classes_str = ", ".join([_format_class_name(c, class_names) for c in small_box_classes])
        recommendations.append(f"These classes have very small bounding boxes: {classes_str}. "
                              "Consider using models optimized for small objects.")
    
    # Box density recommendations
    avg_boxes_per_img = total_boxes / df_bboxes['filename'].nunique()
    if avg_boxes_per_img > 15:
        recommendations.append(f"High object density detected ({avg_boxes_per_img:.1f} objects per image). "
                              "Consider using a detection model that handles dense predictions well.")
    
    results['recommendations'] = recommendations
    
    # 6. Generate visualizations
    try:
        # Class distribution plot
        plt.figure(figsize=(12, 6))
        
        # Prepare category labels
        labels = [_format_class_name(i, class_names) for i in class_counts.index]
        
        # Create the bar chart 
        bars = plt.bar(range(len(labels)), class_counts.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(class_counts))))
        
        # Add value labels on top of bars
        for i, (count, bar) in enumerate(zip(class_counts.values, bars)):
            plt.text(bar.get_x() + bar.get_width()/2., count + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Add a horizontal line for the average
        plt.axhline(y=avg_count, color='r', linestyle='--', 
                   label=f'Average: {avg_count:.1f}')
        
        # Set labels and title
        plt.xlabel('Class')
        plt.ylabel('Number of Bounding Boxes')
        plt.title('Distribution of Classes in Dataset')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        
        # Save the figure to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
            results['class_distribution_plot'] = tmp.name
            
        plt.close()
        
        # Box size distribution plot
        plt.figure(figsize=(10, 6))
        
        # Box plot of bounding box areas by class
        sns.boxplot(x='class_id', y='area_px', data=df_bboxes)
        
        # Set labels and title
        plt.xlabel('Class')
        plt.ylabel('Bounding Box Area (pixels)')
        plt.title('Distribution of Bounding Box Areas by Class')
        plt.xticks(range(len(class_names)), 
                  [_format_class_name(i, class_names) for i in range(len(class_names))], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
            results['box_size_plot'] = tmp.name
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    logger.info(f"Class Balance Report completed: {len(class_counts)} classes analyzed")
    return results


def print_cbr_summary(cbr_results: Dict[str, Any], verbose: bool = False):
    """
    Pretty-print a summary of the Class Balance Report.
    
    Args:
        cbr_results: Results dictionary from CBR function
        verbose: Whether to print detailed statistics
    """
    if not cbr_results:
        print("No valid Class Balance Report results to display.")
        return
        
    print("\n========== CLASS BALANCE REPORT SUMMARY ==========\n")
    
    # Dataset stats
    stats = cbr_results.get('dataset_stats', {})
    print(f"Total Images: {stats.get('total_images', 0):,}")
    print(f"Total Bounding Boxes: {stats.get('total_bounding_boxes', 0):,}")
    print(f"Total Classes: {stats.get('total_classes', 0)}")
    print(f"Average Boxes per Class: {stats.get('average_boxes_per_class', 0):.1f}")
    print(f"Gini Coefficient: {stats.get('gini_coefficient', 0):.3f} (0=perfectly balanced, 1=completely imbalanced)")
    print(f"Coefficient of Variation: {stats.get('coefficient_of_variation', 0):.3f}")
    print(f"Max-to-Min Ratio: {stats.get('max_to_min_ratio', 0):.1f}\n")
    
    # Class Counts
    class_counts = cbr_results.get('class_counts', {})
    class_percentages = cbr_results.get('class_percentages', {})
    
    print("Class Distribution:")
    print("-------------------")
    
    # Get class names if available
    class_names = {}
    for cls in sorted(class_counts.keys()):
        class_name = _format_class_name(int(cls), class_names)
        count = class_counts.get(cls, 0)
        percentage = class_percentages.get(cls, 0)
        print(f"{class_name}: {count:,} ({percentage:.1f}%)")
    
    print("\nClass Balance Issues:")
    print("--------------------")
    
    # Print overrepresented classes
    over = stats.get('overrepresented_classes', [])
    if over:
        print("Overrepresented classes:")
        for cls in over[:5]:
            class_name = _format_class_name(int(cls), class_names)
            count = class_counts.get(str(cls), 0) 
            print(f"  - {class_name}: {count:,} instances")
        if len(over) > 5:
            print(f"  - ... and {len(over) - 5} more classes")
    
    # Print underrepresented classes
    under = stats.get('underrepresented_classes', [])
    if under:
        print("Underrepresented classes:")
        for cls in under[:5]:
            class_name = _format_class_name(int(cls), class_names)
            count = class_counts.get(str(cls), 0)
            print(f"  - {class_name}: {count:,} instances")
        if len(under) > 5:
            print(f"  - ... and {len(under) - 5} more classes")
    
    # Print recommendations
    recommendations = cbr_results.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        print("--------------")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    if verbose:
        print("\nDetailed Class Statistics:")
        print("------------------------")
        
        class_sizes = cbr_results.get('class_sizes', {})
        for class_id, stats in sorted(class_sizes.items()):
            class_name = _format_class_name(int(class_id), class_names)
            print(f"\n{class_name}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Average Size: {stats['width_px']['mean']:.1f} × {stats['height_px']['mean']:.1f} pixels")
            print(f"  Average Area: {stats['area_px']['mean']:.1f} pixels²")
            print(f"  Average Aspect Ratio: {stats['aspect_ratio']['mean']:.2f}")
            
    print("\nVisualization files:")
    print("------------------")
    if 'class_distribution_plot' in cbr_results:
        print(f"Class Distribution Plot: {cbr_results['class_distribution_plot']}")
    if 'box_size_plot' in cbr_results:
        print(f"Box Size Plot: {cbr_results['box_size_plot']}")
    if 'spatial_heatmap_path' in cbr_results:
        print(f"Spatial Heatmap Plot: {cbr_results['spatial_heatmap_path']}")
