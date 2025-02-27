"""
Scoring module for OceanCVBench.

This module handles calculating benchmark scores and saving results.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import csv

# Import metrics module for scoring calculations
from evaluation.metrics import calculate_precision_recall, calculate_map

# Set up logging
logger = logging.getLogger("oceancvbench.scoring")

def calculate_metrics_for_folder(detections_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics for a single benchmark folder.
    
    Args:
        detections_df: DataFrame with detection results
        ground_truth_df: DataFrame with ground truth boxes
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # If either DataFrame is empty, return empty metrics
    if detections_df.empty or ground_truth_df.empty:
        logger.warning("Empty detection or ground truth DataFrame, returning zeroes")
        return {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-score': 0.0,
            'mAP': 0.0,
            'FP_rate': 0.0,
            'FN_rate': 0.0
        }
    
    # Calculate precision-recall at IoU 0.5
    pr_results = calculate_precision_recall(
        predictions=detections_df,
        ground_truth=ground_truth_df,
        iou_threshold=0.5,
        class_specific=False
    )
    
    # Extract overall metrics
    overall = pr_results.get('overall', {})
    metrics['Precision'] = overall.get('precision', 0.0)
    metrics['Recall'] = overall.get('recall', 0.0)
    metrics['F1-score'] = overall.get('f1', 0.0)
    
    # Calculate mAP
    map_results = calculate_map(
        predictions=detections_df,
        ground_truth=ground_truth_df,
        iou_thresholds=[0.5],  # Use mAP@0.5 for simplicity
    )
    
    metrics['mAP'] = map_results.get('mAP', 0.0)
    
    # Calculate FP and FN rates
    total_gt = len(ground_truth_df)
    tp = overall.get('true_positives', 0)
    fp = overall.get('false_positives', 0)
    fn = total_gt - tp
    
    metrics['FP_rate'] = fp / total_gt if total_gt > 0 else 0.0
    metrics['FN_rate'] = fn / total_gt if total_gt > 0 else 0.0
    
    # Add processing time (will be populated if available)
    metrics['processing_time_ms'] = 0.0
    
    return metrics


def calculate_benchmark_scores(evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculate benchmark scores across all folders.
    
    Args:
        evaluation_results: Dictionary mapping folder names to evaluation results
        
    Returns:
        Dictionary with metrics per folder and overall
    """
    benchmark_scores = {}
    
    # Process each folder
    folder_weights = {}
    total_images = 0
    
    for folder_name, folder_data in evaluation_results.items():
        logger.info(f"Calculating metrics for folder: {folder_name}")
        
        detections_df = folder_data.get('detections', pd.DataFrame())
        ground_truth_df = folder_data.get('ground_truth', pd.DataFrame())
        inference_time = folder_data.get('inference_time', 0.0)
        image_count = folder_data.get('image_count', 0)
        
        # Calculate metrics for this folder
        metrics = calculate_metrics_for_folder(detections_df, ground_truth_df)
        
        # Add processing time if available
        if inference_time > 0 and image_count > 0:
            metrics['processing_time_ms'] = (inference_time * 1000) / image_count
        
        # Store metrics
        benchmark_scores[folder_name] = metrics
        
        # Track folder weight for weighted average
        folder_weights[folder_name] = image_count
        total_images += image_count
    
    # Calculate overall weighted average
    if total_images > 0 and benchmark_scores:
        overall_metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-score': 0.0,
            'mAP': 0.0,
            'FP_rate': 0.0,
            'FN_rate': 0.0,
            'processing_time_ms': 0.0
        }
        
        for folder_name, metrics in benchmark_scores.items():
            weight = folder_weights[folder_name] / total_images
            
            for metric_name, metric_value in metrics.items():
                overall_metrics[metric_name] += metric_value * weight
        
        benchmark_scores["Overall Score"] = overall_metrics
    
    logger.info(f"Calculated benchmark scores for {len(benchmark_scores) - 1} folders")
    return benchmark_scores


def save_benchmark_results(benchmark_scores: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Save benchmark results to a JSON file.
    
    Args:
        benchmark_scores: Dictionary with benchmark scores
        output_path: Path where to save the results
    """
    logger.info(f"Saving benchmark results to {output_path}")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(benchmark_scores, f, indent=2)
        
        # Also save as CSV for easier viewing
        csv_path = os.path.splitext(output_path)[0] + '.csv'
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Folder', 'Precision', 'Recall', 'F1-score', 'mAP', 'FP_rate', 'FN_rate', 'processing_time_ms']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for folder, metrics in benchmark_scores.items():
                row = {'Folder': folder}
                row.update(metrics)
                writer.writerow(row)
        
        logger.info(f"Results also saved as CSV to {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to save benchmark results: {e}")
        raise


def generate_leaderboard_entry(benchmark_scores: Dict[str, Dict[str, float]], model_name: str) -> Dict[str, Any]:
    """
    Generate a standardized entry for the leaderboard.
    
    Args:
        benchmark_scores: Dictionary with benchmark scores
        model_name: Name to identify the model
        
    Returns:
        Dictionary with formatted leaderboard entry
    """
    # Get overall metrics
    overall = benchmark_scores.get("Overall Score", {})
    
    # Create the entry
    entry = {
        "model_name": model_name,
        "submission_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "overall_metrics": {
            "mAP": round(overall.get("mAP", 0.0), 4),
            "F1_score": round(overall.get("F1-score", 0.0), 4),
            "precision": round(overall.get("Precision", 0.0), 4),
            "recall": round(overall.get("Recall", 0.0), 4)
        },
        "environmental_conditions": {}
    }
    
    # Add per-condition results
    for folder, metrics in benchmark_scores.items():
        if folder == "Overall Score":
            continue
            
        entry["environmental_conditions"][folder] = {
            "mAP": round(metrics.get("mAP", 0.0), 4),
            "F1_score": round(metrics.get("F1-score", 0.0), 4)
        }
    
    return entry


def compare_with_previous_benchmark(current_scores: Dict[str, Dict[str, float]], 
                                   previous_path: str) -> Dict[str, Any]:
    """
    Compare current benchmark with previous results.
    
    Args:
        current_scores: Current benchmark scores
        previous_path: Path to previous benchmark results
        
    Returns:
        Dictionary with comparison results
    """
    # Initialize comparison result
    comparison = {
        "improvement": False,
        "overall_change": 0.0,
        "detail": {}
    }
    
    if not os.path.exists(previous_path):
        logger.warning(f"Previous benchmark results not found at {previous_path}")
        return comparison
    
    try:
        # Load previous results
        with open(previous_path, 'r') as f:
            previous_scores = json.load(f)
        
        # Get overall scores
        current_overall = current_scores.get("Overall Score", {})
        previous_overall = previous_scores.get("Overall Score", {})
        
        # Compare mAP (main metric)
        current_map = current_overall.get("mAP", 0.0)
        previous_map = previous_overall.get("mAP", 0.0)
        map_change = current_map - previous_map
        
        comparison["improvement"] = map_change > 0
        comparison["overall_change"] = round(map_change, 4)
        
        # Compare each condition
        for folder in current_scores:
            if folder == "Overall Score":
                continue
                
            if folder in previous_scores:
                current_metrics = current_scores[folder]
                previous_metrics = previous_scores[folder]
                
                folder_comparison = {}
                for metric in ["mAP", "F1-score", "Precision", "Recall"]:
                    current_value = current_metrics.get(metric, 0.0)
                    previous_value = previous_metrics.get(metric, 0.0)
                    change = current_value - previous_value
                    
                    folder_comparison[metric] = {
                        "current": round(current_value, 4),
                        "previous": round(previous_value, 4),
                        "change": round(change, 4),
                        "improvement": change > 0
                    }
                
                comparison["detail"][folder] = folder_comparison
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing with previous benchmark: {e}")
        return comparison