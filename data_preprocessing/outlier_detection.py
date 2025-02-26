"""Functions to detect outlier images and bounding boxes."""

import numpy as np
import pandas as pd
from scipy import stats


def detect_whole_image_outliers(image_features_df, method="zscore", threshold=3):
    """
    Identifies outlier images based on feature distributions.
    
    Args:
        image_features_df: DataFrame with image features (brightness, contrast, etc.)
        method: Method for outlier detection ("zscore", "iqr")
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame containing only the outlier rows
    """
    if image_features_df.empty:
        return pd.DataFrame()
    
    if method == "zscore":
        # Calculate z-scores for each numeric column
        numeric_cols = image_features_df.select_dtypes(include=[np.number]).columns
        z_scores = stats.zscore(image_features_df[numeric_cols], nan_policy='omit')
        
        # Find where absolute z-scores exceed threshold
        outliers_mask = np.abs(z_scores) > threshold
        outliers_mask = outliers_mask.any(axis=1)
        
        return image_features_df[outliers_mask]
    
    elif method == "iqr":
        # IQR method
        numeric_cols = image_features_df.select_dtypes(include=[np.number]).columns
        outliers_mask = np.zeros(len(image_features_df), dtype=bool)
        
        for col in numeric_cols:
            Q1 = image_features_df[col].quantile(0.25)
            Q3 = image_features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            col_outliers = (image_features_df[col] < lower_bound) | (image_features_df[col] > upper_bound)
            outliers_mask = outliers_mask | col_outliers
            
        return image_features_df[outliers_mask]
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def detect_bounding_box_outliers(df_bboxes, z_thresh=3):
    """
    Identifies outlier bounding boxes by comparing dimensions within each class.
    
    Args:
        df_bboxes: DataFrame with columns 'class_id', 'width', 'height'
        z_thresh: Z-score threshold for outlier detection
        
    Returns:
        DataFrame containing only the outlier bounding boxes
    """
    if df_bboxes.empty:
        return pd.DataFrame()
    
    # If width/height not in df but xmin, ymin, xmax, ymax are, calculate width/height
    if 'width' not in df_bboxes.columns and all(col in df_bboxes.columns for col in ['xmin', 'xmax']):
        df_bboxes['width'] = df_bboxes['xmax'] - df_bboxes['xmin']
        
    if 'height' not in df_bboxes.columns and all(col in df_bboxes.columns for col in ['ymin', 'ymax']):
        df_bboxes['height'] = df_bboxes['ymax'] - df_bboxes['ymin']
    
    # Check required columns exist
    required_cols = ['class_id', 'width', 'height']
    if not all(col in df_bboxes.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_bboxes.columns]
        raise ValueError(f"Missing required columns in df_bboxes: {missing}")
    
    # Create a copy to avoid modifying the original
    df = df_bboxes.copy()
    
    # Add area and aspect ratio
    df['area'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    
    # Find outliers per class
    outliers = pd.DataFrame()
    for class_id, group in df.groupby('class_id'):
        if len(group) <= 1:  # Skip classes with only one instance
            continue
            
        # Calculate z-scores for this class
        for metric in ['width', 'height', 'area', 'aspect_ratio']:
            z_scores = stats.zscore(group[metric])
            outlier_mask = np.abs(z_scores) > z_thresh
            
            if outlier_mask.any():
                class_outliers = group[outlier_mask].copy()
                class_outliers['outlier_metric'] = metric
                class_outliers['z_score'] = z_scores[outlier_mask]
                outliers = pd.concat([outliers, class_outliers])
    
    return outliers.drop_duplicates()
