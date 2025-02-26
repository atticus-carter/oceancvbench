"""Utilities for extracting and reporting dataset statistics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from typing import Dict, List, Any, Tuple, Optional
import glob
import logging
from tqdm import tqdm

# Set up logging
logger = logging.getLogger('oceancvbench.data_preprocessing.dataset_stats')


def extract_image_stats(image_path: str) -> Dict[str, float]:
    """
    Extract statistical features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary of image statistics
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return {}
            
        # Convert to RGB for consistent stats
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width, channels = img.shape
        
        # Calculate basic statistics
        stats = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'size_bytes': os.path.getsize(image_path),
            'mean_r': np.mean(img_rgb[:, :, 0]),
            'mean_g': np.mean(img_rgb[:, :, 1]),
            'mean_b': np.mean(img_rgb[:, :, 2]),
            'std_r': np.std(img_rgb[:, :, 0]),
            'std_g': np.std(img_rgb[:, :, 1]),
            'std_b': np.std(img_rgb[:, :, 2]),
            'brightness': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        }
        
        # Calculate entropy (measure of information/texture)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / (height * width)
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        stats['entropy'] = entropy
        
        # Calculate blur detection (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        stats['blur_metric'] = laplacian
        
        # Calculate saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        stats['mean_saturation'] = np.mean(hsv[:, :, 1])
        stats['std_saturation'] = np.std(hsv[:, :, 1])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return {}


def analyze_dataset_images(folder_path: str) -> pd.DataFrame:
    """
    Analyze all images in a dataset folder.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        DataFrame with image statistics
    """
    # Find all images in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    
    # Check if folder has 'images' subfolder
    images_subfolder = os.path.join(folder_path, 'images')
    if os.path.exists(images_subfolder) and os.path.isdir(images_subfolder):
        search_folder = images_subfolder
    else:
        search_folder = folder_path
        
    # Find all image files
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(search_folder, ext)))
        # Also search subdirectories
        image_paths.extend(glob.glob(os.path.join(search_folder, "**", ext), recursive=True))
    
    if not image_paths:
        logger.warning(f"No image files found in {folder_path}")
        return pd.DataFrame()
    
    # Process each image
    all_stats = []
    
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        stats = extract_image_stats(img_path)
        if stats:
            stats['filename'] = os.path.basename(img_path)
            stats['filepath'] = img_path
            all_stats.append(stats)
    
    # Create DataFrame
    df = pd.DataFrame(all_stats)
    logger.info(f"Analyzed {len(df)} images in {folder_path}")
    
    return df


def plot_dataset_statistics(stats_df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Generate plots visualizing dataset statistics.
    
    Args:
        stats_df: DataFrame with image statistics
        output_dir: Directory to save plots (if None, plots are displayed)
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if stats_df.empty:
        logger.warning("Empty statistics DataFrame, cannot generate plots")
        return {}
        
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plot_files = {}
    
    # 1. Resolution distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(stats_df['width'], stats_df['height'], alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Image Resolution Distribution')
    
    if output_dir:
        path = os.path.join(output_dir, 'resolution_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plot_files['resolution'] = path
        plt.close()
    else:
        plt.show()
    
    # 2. Brightness and contrast distribution
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(stats_df['brightness'], kde=True)
    plt.xlabel('Mean Brightness')
    plt.title('Brightness Distribution')
    
    plt.subplot(1, 2, 2)
    # Use standard deviation of grayscale as a proxy for contrast
    contrast = (stats_df['std_r'] + stats_df['std_g'] + stats_df['std_b']) / 3
    sns.histplot(contrast, kde=True)
    plt.xlabel('Standard Deviation (Contrast)')
    plt.title('Contrast Distribution')
    
    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, 'brightness_contrast.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plot_files['brightness_contrast'] = path
        plt.close()
    else:
        plt.show()
    
    # 3. Blur metric
    plt.figure(figsize=(10, 6))
    sns.histplot(stats_df['blur_metric'], kde=True)
    plt.axvline(x=100, color='r', linestyle='--', label='Potential blur threshold')
    plt.xlabel('Laplacian Variance (lower = more blurry)')
    plt.ylabel('Count')
    plt.title('Image Blur Distribution')
    plt.legend()
    
    if output_dir:
        path = os.path.join(output_dir, 'blur_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plot_files['blur'] = path
        plt.close()
    else:
        plt.show()
    
    # 4. Color distribution
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(stats_df['mean_r'], color='r', kde=True)
    plt.xlabel('Mean Red Channel')
    
    plt.subplot(2, 2, 2)
    sns.histplot(stats_df['mean_g'], color='g', kde=True)
    plt.xlabel('Mean Green Channel')
    
    plt.subplot(2, 2, 3)
    sns.histplot(stats_df['mean_b'], color='b', kde=True)
    plt.xlabel('Mean Blue Channel')
    
    plt.subplot(2, 2, 4)
    sns.histplot(stats_df['mean_saturation'], color='purple', kde=True)
    plt.xlabel('Mean Saturation')
    
    plt.suptitle('Color Distribution in Dataset')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'color_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plot_files['color'] = path
        plt.close()
    else:
        plt.show()
    
    return plot_files


def identify_problematic_images(stats_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Identify potentially problematic images in the dataset.
    
    Args:
        stats_df: DataFrame with image statistics
        
    Returns:
        Dictionary mapping issue type to DataFrame of problematic images
    """
    results = {}
    
    # Skip if empty DataFrame
    if stats_df.empty:
        return results
    
    # 1. Very blurry images
    blur_threshold = stats_df['blur_metric'].quantile(0.1)  # Bottom 10%
    blurry = stats_df[stats_df['blur_metric'] < blur_threshold].copy()
    if not blurry.empty:
        results['blurry_images'] = blurry.sort_values('blur_metric')
    
    # 2. Extreme brightness/darkness
    bright_threshold = stats_df['brightness'].quantile(0.95)
    dark_threshold = stats_df['brightness'].quantile(0.05)
    
    too_bright = stats_df[stats_df['brightness'] > bright_threshold].copy()
    too_dark = stats_df[stats_df['brightness'] < dark_threshold].copy()
    
    if not too_bright.empty:
        results['too_bright'] = too_bright.sort_values('brightness', ascending=False)
    if not too_dark.empty:
        results['too_dark'] = too_dark.sort_values('brightness')
    
    # 3. Unusual aspect ratios
    median_aspect = stats_df['aspect_ratio'].median()
    aspect_deviation = abs(stats_df['aspect_ratio'] - median_aspect)
    unusual_aspect = stats_df[aspect_deviation > aspect_deviation.quantile(0.95)].copy()
    
    if not unusual_aspect.empty:
        results['unusual_aspect_ratio'] = unusual_aspect.sort_values('aspect_ratio')
    
    # 4. Low color variance (potentially uniform backgrounds)
    color_var = (stats_df['std_r'] + stats_df['std_g'] + stats_df['std_b']) / 3
    low_var_threshold = color_var.quantile(0.05)
    low_color_var = stats_df[color_var < low_var_threshold].copy()
    
    if not low_color_var.empty:
        results['low_color_variance'] = low_color_var
    
    return results


def generate_dataset_report(folder_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report on a dataset including statistics and plots.
    
    Args:
        folder_path: Path to the dataset folder
        output_dir: Directory to save report files
        
    Returns:
        Dictionary with report statistics and file paths
    """
    # Analyze the dataset
    stats_df = analyze_dataset_images(folder_path)
    
    if stats_df.empty:
        logger.warning(f"No valid images found in {folder_path}")
        return {'error': 'No valid images found'}
    
    # Generate plots
    plots = plot_dataset_statistics(stats_df, output_dir)
    
    # Identify problematic images
    issues = identify_problematic_images(stats_df)
    
    # Compile report
    report = {
        'dataset_path': folder_path,
        'total_images': len(stats_df),
        'unique_resolutions': stats_df.groupby(['width', 'height']).size().reset_index().rename(columns={0: 'count'}).to_dict('records'),
        'mean_resolution': {
            'width': stats_df['width'].mean(),
            'height': stats_df['height'].mean()
        },
        'brightness_stats': {
            'mean': stats_df['brightness'].mean(),
            'std': stats_df['brightness'].std(),
            'min': stats_df['brightness'].min(),
            'max': stats_df['brightness'].max()
        },
        'blur_stats': {
            'mean': stats_df['blur_metric'].mean(),
            'problematic_count': len(issues.get('blurry_images', pd.DataFrame()))
        },
        'color_stats': {
            'mean_rgb': [stats_df['mean_r'].mean(), stats_df['mean_g'].mean(), stats_df['mean_b'].mean()],
            'mean_saturation': stats_df['mean_saturation'].mean()
        },
        'plot_files': plots,
        'issues': {k: len(v) for k, v in issues.items()},
        'issue_examples': {k: v['filename'].tolist()[:5] for k, v in issues.items()}
    }
    
    # Save full stats to CSV if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_csv_path = os.path.join(output_dir, 'dataset_statistics.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        report['statistics_csv'] = stats_csv_path
        
        # Also save issues to separate CSVs
        for issue_type, issue_df in issues.items():
            issue_csv_path = os.path.join(output_dir, f'issue_{issue_type}.csv')
            issue_df.to_csv(issue_csv_path, index=False)
            
    return report
