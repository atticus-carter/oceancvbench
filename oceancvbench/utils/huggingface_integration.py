"""
Hugging Face integration utilities for OceanCVBench.

This module provides functions for interacting with the Hugging Face Hub,
including uploading models, downloading datasets, and submitting benchmark results.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

# Set up logging
logger = logging.getLogger("oceancvbench.utils.huggingface_integration")

try:
    from huggingface_hub import HfApi, HfFolder, Repository, upload_file, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    HF_AVAILABLE = True
except ImportError:
    logger.warning("huggingface_hub package not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False


def ensure_huggingface_installed() -> bool:
    """Check if huggingface_hub is installed."""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub package is required but not installed.")
        logger.error("Install with: pip install huggingface_hub")
        return False
    return True


def login_to_huggingface(token: Optional[str] = None) -> bool:
    """
    Log in to Hugging Face Hub using a token.
    
    Args:
        token: Hugging Face API token. If None, will look for env var HF_TOKEN or use cached token.
        
    Returns:
        True if login was successful, False otherwise
    """
    if not ensure_huggingface_installed():
        return False
        
    try:
        # Try to use provided token, then environment variable, then cached token
        if token:
            HfApi().set_access_token(token)
            return True
        
        # Check if already logged in
        if HfFolder().get_token() is not None:
            logger.info("Already logged in to Hugging Face Hub")
            return True
            
        # Try environment variable
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            HfApi().set_access_token(env_token)
            logger.info("Logged in to Hugging Face Hub using environment token")
            return True
            
        logger.warning("No Hugging Face token provided. Some operations may fail.")
        return False
    except Exception as e:
        logger.error(f"Error logging in to Hugging Face Hub: {e}")
        return False


def upload_model_to_hub(
    repo_id: str,
    model_path: str,
    metadata: Dict[str, Any],
    token: Optional[str] = None,
    repo_type: str = "model",
    commit_message: str = "Upload marine vision model",
    create_repo: bool = True
) -> str:
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        model_path: Path to the model file or directory
        metadata: Dictionary with model metadata
        token: Hugging Face API token
        repo_type: Repository type ("model", "dataset", etc.)
        commit_message: Commit message
        create_repo: Whether to create the repo if it doesn't exist
        
    Returns:
        URL to the uploaded model
    """
    if not ensure_huggingface_installed() or not login_to_huggingface(token):
        raise RuntimeError("Failed to login to Hugging Face Hub")
    
    try:
        # Initialize API
        api = HfApi()
        
        # Check if repo exists, create if needed
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            logger.info(f"Repository {repo_id} already exists")
        except RepositoryNotFoundError:
            if create_repo:
                logger.info(f"Creating repository {repo_id}")
                api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
            else:
                raise ValueError(f"Repository {repo_id} does not exist and create_repo=False")
        
        # Create a temporary directory for staging
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create README.md with metadata
            readme_path = os.path.join(tmp_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"# {os.path.basename(repo_id)}\n\n")
                f.write("This model was uploaded using OceanCVBench.\n\n")
                f.write("## Model Information\n\n")
                
                for key, value in metadata.items():
                    f.write(f"- **{key}**: {value}\n")
            
            # Also save metadata as JSON
            meta_path = os.path.join(tmp_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Copy model file(s)
            if os.path.isdir(model_path):
                # Copy entire directory
                for item in os.listdir(model_path):
                    s = os.path.join(model_path, item)
                    d = os.path.join(tmp_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
            else:
                # Copy single file
                shutil.copy2(model_path, tmp_dir)
            
            # Upload to hub
            logger.info(f"Uploading to Hugging Face Hub: {repo_id}")
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message
            )
            
        logger.info(f"Model uploaded successfully to {repo_id}")
        return f"https://huggingface.co/{repo_id}"
    
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise


def download_model_from_hub(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    specific_file: Optional[str] = None
) -> str:
    """
    Download a model from the Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        local_dir: Local directory to save the model
        token: Hugging Face API token
        specific_file: Download only a specific file from the repo
        
    Returns:
        Path to the downloaded model
    """
    if not ensure_huggingface_installed() or not login_to_huggingface(token):
        raise RuntimeError("Failed to login to Hugging Face Hub")
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        
        # Initialize repository
        repo = Repository(
            local_dir=local_dir, 
            clone_from=repo_id,
            use_auth_token=token or True
        )
        
        # Pull latest changes
        repo.git_pull()
        
        # Check if specific file requested
        if specific_file:
            file_path = os.path.join(local_dir, specific_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {specific_file} not found in repo {repo_id}")
            return file_path
            
        # Look for standard model files
        for model_file in ["model.pt", "best.pt", "weights/best.pt", "*.pt"]:
            if "*" in model_file:
                # Handle wildcards
                import glob
                matches = glob.glob(os.path.join(local_dir, model_file))
                if matches:
                    return matches[0]
            else:
                file_path = os.path.join(local_dir, model_file)
                if os.path.exists(file_path):
                    return file_path
        
        logger.warning(f"No model file automatically detected in {repo_id}")
        return local_dir  # Return directory if no specific file found
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


def submit_benchmark_results(
    repo_id: str,
    results: Dict[str, Any],
    token: Optional[str] = None,
    benchmark_name: str = "oceancvbench",
    commit_message: str = "Add benchmark results"
) -> str:
    """
    Submit benchmark results to the Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID for the model
        results: Dictionary with benchmark results
        token: Hugging Face API token
        benchmark_name: Name of the benchmark
        commit_message: Commit message
        
    Returns:
        URL to the results
    """
    if not ensure_huggingface_installed() or not login_to_huggingface(token):
        raise RuntimeError("Failed to login to Hugging Face Hub")
    
    try:
        # Initialize API
        api = HfApi()
        
        # Check if repo exists
        try:
            api.repo_info(repo_id=repo_id)
        except RepositoryNotFoundError:
            raise ValueError(f"Repository {repo_id} does not exist")
        
        # Create a temporary directory for staging
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create results file
            results_file = os.path.join(tmp_dir, f"{benchmark_name}_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Upload file
            uploaded_path = api.upload_file(
                path_or_fileobj=results_file,
                path_in_repo=f"benchmark_results/{benchmark_name}_results.json",
                repo_id=repo_id,
                commit_message=commit_message
            )
            
        logger.info(f"Benchmark results uploaded successfully to {repo_id}")
        return f"https://huggingface.co/{repo_id}/blob/main/{uploaded_path}"
    
    except Exception as e:
        logger.error(f"Error uploading benchmark results: {e}")
        raise


def list_benchmark_results(
    benchmark_name: str = "oceancvbench",
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all models with benchmark results for a specific benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        token: Hugging Face API token
        
    Returns:
        List of dictionaries with model info and results
    """
    if not ensure_huggingface_installed() or not login_to_huggingface(token):
        logger.warning("Not logged in to Hugging Face Hub, results may be incomplete")
    
    try:
        # Initialize API
        api = HfApi()
        
        # Search for models with the benchmark tag
        models = api.list_models(filter=benchmark_name)
        
        results = []
        for model in models:
            try:
                # Try to download the results file
                repo_id = model.modelId
                result_path = f"benchmark_results/{benchmark_name}_results.json"
                
                try:
                    # Download the results file content
                    content = api.hf_hub_download(repo_id=repo_id, filename=result_path, use_auth_token=token or True)
                    with open(content, "r") as f:
                        benchmark_data = json.load(f)
                    
                    # Add to results
                    results.append({
                        "repo_id": repo_id,
                        "results": benchmark_data
                    })
                except Exception as e:
                    logger.debug(f"No benchmark results for {repo_id}: {e}")
                    
            except Exception as e:
                logger.debug(f"Error processing model {model.modelId}: {e}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error listing benchmark results: {e}")
        return []
