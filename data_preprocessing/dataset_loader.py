"""
Dataset handling module for OceanCVBench.

This module downloads and prepares the training and benchmark datasets from HuggingFace.
"""

import os
import logging
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import Optional
import zipfile

# Import huggingface_hub for downloading datasets
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub package is required. "
        "Install it with: pip install huggingface_hub"
    )

# Set up logging
logger = logging.getLogger("oceancvbench.dataset_loader")

# HuggingFace repository info
HF_REPO_ID = "oceancvbench/marine-benchmark-dataset"
TRAIN_DATASET_FILENAME = "train_dataset.zip"
BENCHMARK_DATASET_FILENAME = "oceanbench.zip"
DATA_YAML_FILENAME = "data.yaml"

def download_file_from_hf(repo_id: str, filename: str, local_dir: str) -> str:
    """
    Download a file from a HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Name of the file to download
        local_dir: Directory where to save the file
        
    Returns:
        Path to the downloaded file
    """
    logger.info(f"Downloading {filename} from {repo_id}")
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=local_dir,
            force_download=True
        )
        logger.info(f"Successfully downloaded to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {filename} from {repo_id}: {e}")
        raise

def extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory where to extract the contents
    """
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Successfully extracted {zip_path}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise

def prepare_data_yaml(base_dir: str) -> str:
    """
    Prepare the data.yaml file for training.
    
    Args:
        base_dir: Base directory where datasets are stored
        
    Returns:
        Path to the data.yaml file
    """
    yaml_path = os.path.join(base_dir, DATA_YAML_FILENAME)
    
    try:
        # Download the data.yaml from HuggingFace
        downloaded_yaml = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=DATA_YAML_FILENAME,
            cache_dir=base_dir,
            force_download=True
        )
        
        # Read the downloaded yaml
        with open(downloaded_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to match local directories
        data_config['train'] = os.path.join(base_dir, 'train')
        data_config['val'] = os.path.join(base_dir, 'val')
        data_config['test'] = os.path.join(base_dir, 'test', 'oceanbench')
        
        # Write updated yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path
        
    except Exception as e:
        logger.error(f"Failed to prepare data.yaml: {e}")
        raise

def download_benchmark_datasets(data_dir: str) -> str:
    """
    Download and prepare benchmark datasets.
    
    Args:
        data_dir: Directory where to store the datasets
        
    Returns:
        Path to the data.yaml file
    """
    logger.info("Starting dataset download")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download training dataset
        train_zip_path = download_file_from_hf(HF_REPO_ID, TRAIN_DATASET_FILENAME, temp_dir)
        
        # Download benchmark dataset
        benchmark_zip_path = download_file_from_hf(HF_REPO_ID, BENCHMARK_DATASET_FILENAME, temp_dir)
        
        # Extract training dataset
        train_extract_dir = os.path.join(data_dir)
        extract_zip(train_zip_path, train_extract_dir)
        
        # Extract benchmark dataset
        test_dir = os.path.join(data_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        extract_zip(benchmark_zip_path, test_dir)
    
    # Verify expected directories exist
    expected_dirs = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val"),
        os.path.join(data_dir, "test", "oceanbench")
    ]
    
    for dir_path in expected_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Expected directory not found: {dir_path}")
            raise FileNotFoundError(f"Expected directory not found: {dir_path}")
    
    # Create data.yaml for YOLO training
    yaml_path = prepare_data_yaml(data_dir)
    
    logger.info(f"Dataset preparation complete. Data YAML path: {yaml_path}")
    return yaml_path
