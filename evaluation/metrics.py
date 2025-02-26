"""
Metrics for evaluating object detection models.

This module provides comprehensive functionality for calculating and visualizing
object detection performance metrics including IoU, precision, recall, mAP,
F1 scores, and confusion matrices.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import json
from tqdm import tqdm
from scipy.interpolate import interp1d

logger = logging.getLogger('oceancvbench.evaluation.metrics')


def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box coordinates (xmin, ymin, xmax, ymax)
        box2: Second box coordinates (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU score between 0 and 1
    """
    # Get the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # If the boxes don't overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of the boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_precision_recall(predictions: pd.DataFrame, 
                              ground_truth: pd.DataFrame,
                              iou_threshold: float = 0.5,
                              conf_threshold: Optional[float] = None,
                              class_specific: bool = True) -> Dict[str, Any]:
    """
    Calculate precision and recall metrics for object detection.
    
    Args:
        predictions: DataFrame with prediction boxes (must include: filename, class_id, xmin, ymin, xmax, ymax, confidence)
        ground_truth: DataFrame with ground truth boxes (must include: filename, class_id, xmin, ymin, xmax, ymax)
        iou_threshold: Minimum IoU for a true positive detection
        conf_threshold: Optional confidence threshold for predictions
        class_specific: Whether to calculate metrics per class
        
    Returns:
        Dictionary containing precision, recall, F1 scores, and other metrics
    """
    # Validate input dataframes
    required_pred_cols = ['filename', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence']
    required_gt_cols = ['filename', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax']
    
    for col in required_pred_cols:
        if col not in predictions.columns:
            raise ValueError(f"Predictions DataFrame is missing required column: {col}")
    
    for col in required_gt_cols:
        if col not in ground_truth.columns:
            raise ValueError(f"Ground truth DataFrame is missing required column: {col}")
    
    # Filter by confidence threshold if specified
    if conf_threshold is not None:
        predictions = predictions[predictions['confidence'] >= conf_threshold].copy()
    
    # Sort predictions by confidence (descending)
    predictions = predictions.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    # Initialize counters
    tp = 0  # True positives
    fp = 0  # False positives
    
    # For class-specific metrics
    class_metrics = {}
    unique_classes = set(ground_truth['class_id'].unique())
    unique_classes.update(predictions['class_id'].unique())
    
    # Track assigned ground truth boxes to avoid double-counting
    assigned_gt = set()  # Set of (filename, gt_idx) tuples
    
    # For per-class tracking
    if class_specific:
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_gt_count = Counter(ground_truth['class_id'])
        
        # Initialize precision-recall curve data
        pr_curve_data = {
            cls: {
                'precision': [],
                'recall': [],
                'threshold': []
            } for cls in unique_classes
        }
    
    # Process each prediction
    for idx, pred in tqdm(predictions.iterrows(), total=len(predictions), desc="Evaluating detections"):
        # Get relevant prediction data
        pred_filename = pred['filename']
        pred_class = pred['class_id']
        pred_box = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
        pred_conf = pred['confidence']
        
        # Find all ground truth boxes in the same image with the same class
        gt_matches = ground_truth[
            (ground_truth['filename'] == pred_filename) & 
            (ground_truth['class_id'] == pred_class)
        ]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find the best matching ground truth box
        for gt_idx, gt_row in gt_matches.iterrows():
            gt_box = (gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax'])
            
            # Calculate IoU
            iou = calculate_iou(pred_box, gt_box)
            
            # Keep track of the best match
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if the best match is good enough and hasn't been assigned yet
        match_key = (pred_filename, best_gt_idx)
        if best_iou >= iou_threshold and best_gt_idx != -1 and match_key not in assigned_gt:
            tp += 1
            assigned_gt.add(match_key)
            
            if class_specific:
                class_tp[pred_class] += 1
        else:
            fp += 1
            
            if class_specific:
                class_fp[pred_class] += 1
        
        # Update precision-recall curve data points
        if class_specific:
            # Calculate cumulative precision and recall for this class
            current_precision = class_tp[pred_class] / (class_tp[pred_class] + class_fp[pred_class]) if (class_tp[pred_class] + class_fp[pred_class]) > 0 else 0
            current_recall = class_tp[pred_class] / class_gt_count[pred_class] if pred_class in class_gt_count and class_gt_count[pred_class] > 0 else 0
            
            pr_curve_data[pred_class]['precision'].append(current_precision)
            pr_curve_data[pred_class]['recall'].append(current_recall)
            pr_curve_data[pred_class]['threshold'].append(pred_conf)
    
    # Count total ground truth boxes
    total_gt = len(ground_truth)
    
    # Calculate overall precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compile results
    results = {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'ground_truth_count': total_gt,
            'iou_threshold': iou_threshold
        }
    }
    
    # Add class-specific metrics
    if class_specific:
        results['class_metrics'] = {}
        
        for cls in unique_classes:
            class_precision = class_tp[cls] / (class_tp[cls] + class_fp[cls]) if (class_tp[cls] + class_fp[cls]) > 0 else 0
            class_recall = class_tp[cls] / class_gt_count[cls] if cls in class_gt_count and class_gt_count[cls] > 0 else 0
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            results['class_metrics'][int(cls)] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'true_positives': class_tp[cls],
                'false_positives': class_fp[cls],
                'ground_truth_count': class_gt_count.get(cls, 0),
                'pr_curve': {
                    'precision': pr_curve_data[cls]['precision'],
                    'recall': pr_curve_data[cls]['recall'],
                    'threshold': pr_curve_data[cls]['threshold']
                }
            }
    
    return results


def calculate_map(predictions: pd.DataFrame, 
                 ground_truth: pd.DataFrame, 
                 iou_thresholds: Union[List[float], np.ndarray] = np.linspace(0.5, 0.95, 10),
                 class_names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    Calculate mean Average Precision (mAP) across classes and IoU thresholds.
    
    Args:
        predictions: DataFrame with prediction boxes
        ground_truth: DataFrame with ground truth boxes
        iou_thresholds: List of IoU thresholds to evaluate
        class_names: Optional dictionary mapping class IDs to names
        
    Returns:
        Dictionary containing mAP scores and per-class AP values
    """
    if isinstance(iou_thresholds, (list, tuple)):
        iou_thresholds = np.array(iou_thresholds)
    
    # Get unique classes
    unique_classes = set(ground_truth['class_id'].unique())
    unique_classes.update(predictions['class_id'].unique())
    
    # Store AP values for each class and threshold
    ap_values = {}
    ap_per_class = {cls: [] for cls in unique_classes}
    
    # Calculate AP for each IoU threshold
    for iou_threshold in tqdm(iou_thresholds, desc="Calculating mAP"):
        pr_results = calculate_precision_recall(
            predictions=predictions,
            ground_truth=ground_truth,
            iou_threshold=iou_threshold,
            class_specific=True
        )
        
        # For each class, calculate AP using the precision-recall curve
        class_metrics = pr_results.get('class_metrics', {})
        
        for cls in unique_classes:
            if cls not in class_metrics:
                ap_per_class[cls].append(0.0)
                continue
                
            # Get precision-recall curve data for this class
            pr_curve = class_metrics[cls]['pr_curve']
            precisions = np.array(pr_curve['precision'])
            recalls = np.array(pr_curve['recall'])
            
            # If we have no detections for this class
            if len(precisions) == 0:
                ap_per_class[cls].append(0.0)
                continue
            
            # Add sentinel values for proper interpolation
            if recalls[0] != 0:
                recalls = np.concatenate(([0], recalls))
                precisions = np.concatenate(([precisions[0]], precisions))
            if recalls[-1] != 1:
                recalls = np.concatenate((recalls, [1]))
                precisions = np.concatenate((precisions, [0]))
            
            # Compute the precision envelope (maximum precision for each recall level)
            for i in range(precisions.size - 1, 0, -1):
                precisions[i-1] = max(precisions[i-1], precisions[i])
            
            # Find indices where recall changes
            i = np.where(recalls[1:] != recalls[:-1])[0]
            
            # Calculate AP using the precision envelope
            ap = np.sum((recalls[i+1] - recalls[i]) * precisions[i+1])
            ap_per_class[cls].append(ap)
            
        # Store AP values for this threshold
        ap_values[iou_threshold] = {cls: ap_per_class[cls][-1] for cls in unique_classes}
    
    # Calculate mAP across classes and thresholds
    ap_array = np.array([np.mean(ap_per_class[cls]) for cls in unique_classes])
    mAP = np.mean(ap_array)
    
    # Calculate mAP@.50 (standard COCO metric)
    mAP50_index = np.argmin(np.abs(iou_thresholds - 0.5))
    mAP50 = np.mean([ap_values[iou_thresholds[mAP50_index]].get(cls, 0) for cls in unique_classes])
    
    # Calculate mAP@.75 (COCO metric)
    mAP75_index = np.argmin(np.abs(iou_thresholds - 0.75))
    mAP75 = np.mean([ap_values[iou_thresholds[mAP75_index]].get(cls, 0) for cls in unique_classes])
    
    # Prepare per-class results
    class_results = {}
    for cls in unique_classes:
        class_name = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        class_results[int(cls)] = {
            'name': class_name,
            'ap': np.mean(ap_per_class[cls]),
            'ap50': ap_values[iou_thresholds[mAP50_index]].get(cls, 0),
            'ap75': ap_values[iou_thresholds[mAP75_index]].get(cls, 0)
        }
    
    # Compile results
    results = {
        'mAP': mAP,
        'mAP50': mAP50,
        'mAP75': mAP75,
        'class_ap': class_results,
        'iou_thresholds': iou_thresholds.tolist()
    }
    
    return results


def create_confusion_matrix(predictions: pd.DataFrame, 
                           ground_truth: pd.DataFrame,
                           iou_threshold: float = 0.5,
                           conf_threshold: Optional[float] = None) -> Tuple[np.ndarray, List[int]]:
    """
    Create a confusion matrix for object detection results.
    
    Args:
        predictions: DataFrame with prediction boxes
        ground_truth: DataFrame with ground truth boxes
        iou_threshold: IoU threshold for matching boxes
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Tuple of (confusion_matrix, class_ids) where class_ids are the class indices in order
    """
    # Filter by confidence threshold if specified
    if conf_threshold is not None:
        predictions = predictions[predictions['confidence'] >= conf_threshold].copy()
    
    # Get all unique class IDs
    all_classes = set(ground_truth['class_id'].unique())
    all_classes.update(predictions['class_id'].unique())
    class_ids = sorted(all_classes)
    
    # Create mapping from class_id to index
    class_to_idx = {cls: i for i, cls in enumerate(class_ids)}
    
    # Initialize confusion matrix (with an extra column and row for background/missed detections)
    n_classes = len(class_ids)
    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    
    # Group by filename for efficient processing
    gt_by_file = ground_truth.groupby('filename')
    pred_by_file = predictions.groupby('filename')
    
    # Get all unique filenames
    all_files = set(ground_truth['filename'].unique())
    all_files.update(predictions['filename'].unique())
    
    # Process each file
    for filename in all_files:
        # Get ground truth boxes for this file
        gt_boxes = gt_by_file.get_group(filename) if filename in gt_by_file.groups else pd.DataFrame()
        # Get predicted boxes for this file
        pred_boxes = pred_by_file.get_group(filename) if filename in pred_by_file.groups else pd.DataFrame()
        
        # Skip if either is empty
        if gt_boxes.empty or pred_boxes.empty:
            # If we have GT but no predictions, count as missed detections
            if not gt_boxes.empty:
                for _, gt_row in gt_boxes.iterrows():
                    gt_class = gt_row['class_id']
                    gt_idx = class_to_idx[gt_class]
                    # Last column is for missed detections (true class but no prediction)
                    confusion_matrix[gt_idx, -1] += 1
            
            # If we have predictions but no GT, count as false positives
            if not pred_boxes.empty:
                for _, pred_row in pred_boxes.iterrows():
                    pred_class = pred_row['class_id']
                    pred_idx = class_to_idx[pred_class]
                    # Last row is for false positives (prediction with no matching GT)
                    confusion_matrix[-1, pred_idx] += 1
                    
            continue
        
        # Calculate IoU matrix between all GT and prediction boxes
        ious = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        for i, (_, gt_row) in enumerate(gt_boxes.iterrows()):
            gt_box = (gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax'])
            
            for j, (_, pred_row) in enumerate(pred_boxes.iterrows()):
                pred_box = (pred_row['xmin'], pred_row['ymin'], pred_row['xmax'], pred_row['ymax'])
                ious[i, j] = calculate_iou(gt_box, pred_box)
        
        # Find matches using the Hungarian algorithm for optimal assignment
        # This approach ensures each ground truth box is matched with at most one prediction
        # and vice versa, maximizing the total IoU.
        from scipy.optimize import linear_sum_assignment
        
        # Set unmatched pairs (IoU < threshold) to 0
        ious_masked = ious.copy()
        ious_masked[ious_masked < iou_threshold] = 0
        
        # Find optimal matching
        gt_indices, pred_indices = linear_sum_assignment(-ious_masked)  # Negative because we want to maximize
        
        # Track which GT and predictions were matched
        gt_matched = set()
        pred_matched = set()
        
        # Update confusion matrix based on matches
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            # Skip if IoU is below threshold
            if ious[gt_idx, pred_idx] < iou_threshold:
                continue
                
            gt_class = gt_boxes.iloc[gt_idx]['class_id']
            pred_class = pred_boxes.iloc[pred_idx]['class_id']
            
            gt_matrix_idx = class_to_idx[gt_class]
            pred_matrix_idx = class_to_idx[pred_class]
            
            # Increment cell in confusion matrix
            confusion_matrix[gt_matrix_idx, pred_matrix_idx] += 1
            
            # Mark as matched
            gt_matched.add(gt_idx)
            pred_matched.add(pred_idx)
        
        # Handle unmatched ground truth (missed detections)
        for i in range(len(gt_boxes)):
            if i not in gt_matched:
                gt_class = gt_boxes.iloc[i]['class_id']
                gt_matrix_idx = class_to_idx[gt_class]
                # Last column is for missed detections
                confusion_matrix[gt_matrix_idx, -1] += 1
        
        # Handle unmatched predictions (false positives)
        for i in range(len(pred_boxes)):
            if i not in pred_matched:
                pred_class = pred_boxes.iloc[i]['class_id']
                pred_matrix_idx = class_to_idx[pred_class]
                # Last row is for false positives
                confusion_matrix[-1, pred_matrix_idx] += 1
    
    # Add the background class ID (we use -1 as a convention)
    class_ids_with_bg = class_ids + [-1]
    
    return confusion_matrix, class_ids_with_bg


def plot_precision_recall_curve(pr_results: Dict[str, Any], 
                               class_names: Optional[Dict[int, str]] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curves for each class.
    
    Args:
        pr_results: Results from calculate_precision_recall function
        class_names: Optional dictionary mapping class IDs to names
        figsize: Figure size
        output_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib Figure object
    """
    class_metrics = pr_results.get('class_metrics', {})
    if not class_metrics:
        logger.warning("No class metrics available to plot precision-recall curves")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot a curve for each class
    for class_id, metrics in class_metrics.items():
        pr_curve = metrics.get('pr_curve', {})
        precision = pr_curve.get('precision', [])
        recall = pr_curve.get('recall', [])
        
        if not precision or not recall:
            continue
        
        # Get class name
        class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
        
        # Plot the curve
        ax.plot(recall, precision, '-', linewidth=2, label=f"{class_name} (AP={metrics.get('f1', 0):.3f})")
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_title(f"Precision-Recall Curves (IoU={pr_results.get('overall', {}).get('iou_threshold', 0.5)})")
    ax.legend(loc='lower left', fontsize='small')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curves to {output_path}")
    
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_ids: List[int],
                         class_names: Optional[Dict[int, str]] = None,
                         normalize: bool = True,
                         figsize: Tuple[int, int] = (12, 10),
                         output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a confusion matrix for object detection results.
    
    Args:
        confusion_matrix: The confusion matrix (output from create_confusion_matrix)
        class_ids: List of class IDs corresponding to the matrix indices
        class_names: Optional dictionary mapping class IDs to names
        normalize: Whether to normalize the matrix by row (true class)
        figsize: Figure size
        output_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib Figure object
    """
    # Convert class IDs to labels
    labels = []
    for class_id in class_ids:
        if class_id == -1:
            labels.append("Background")
        elif class_names and class_id in class_names:
            labels.append(class_names[class_id])
        else:
            labels.append(f"Class {class_id}")
    
    # Create a copy of the matrix for normalization
    cm = confusion_matrix.copy().astype(float)
    
    # Normalize if requested
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums[:, np.newaxis]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Ratio" if normalize else "Count", rotation=-90, va="bottom")
    
    # Add labels and ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add cell values as text
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if normalize:
                text_value = f"{value:.2f}" if value > 0.01 else ""
            else:
                text_value = f"{int(value)}" if value > 0 else ""
                
            ax.text(j, i, text_value,
                   ha="center", va="center", 
                   color="white" if value > threshold else "black")
    
    # Add labels and title
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title)
    
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    return fig

