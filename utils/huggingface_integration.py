"""
HuggingFace integration module for OceanCVBench.

This module handles uploading benchmark results to a HuggingFace leaderboard.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
import time
from pathlib import Path

# Set up logging
logger = logging.getLogger("oceancvbench.huggingface_integration")

# HuggingFace leaderboard constants
HF_LEADERBOARD_REPO = "oceancvbench/marine-benchmark-leaderboard"
HF_API_URL = "https://huggingface.co/api"

def validate_token(token: str) -> bool:
    """
    Validate a HuggingFace API token.
    
    Args:
        token: HuggingFace API token
        
    Returns:
        True if token is valid, False otherwise
    """
    try:
        # Try to access user info as a validation check
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{HF_API_URL}/whoami", headers=headers)
        
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Invalid HuggingFace token: {response.status_code}, {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating HuggingFace token: {e}")
        return False

def check_leaderboard_exists() -> bool:
    """
    Check if the leaderboard repository exists.
    
    Returns:
        True if leaderboard exists, False otherwise
    """
    try:
        response = requests.get(f"{HF_API_URL}/models/{HF_LEADERBOARD_REPO}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking leaderboard existence: {e}")
        return False

def get_current_leaderboard(token: str) -> Dict[str, Any]:
    """
    Retrieve the current leaderboard data.
    
    Args:
        token: HuggingFace API token
        
    Returns:
        Dictionary with leaderboard data
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        leaderboard_url = f"https://huggingface.co/{HF_LEADERBOARD_REPO}/resolve/main/leaderboard.json"
        
        response = requests.get(leaderboard_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Could not retrieve leaderboard: {response.status_code}, {response.text}")
            # Return empty leaderboard
            return {"entries": [], "last_updated": time.strftime("%Y-%m-%d")}
            
    except Exception as e:
        logger.error(f"Error retrieving leaderboard: {e}")
        return {"entries": [], "last_updated": time.strftime("%Y-%m-%d")}

def upload_leaderboard_entry(leaderboard_data: Dict[str, Any], token: str) -> bool:
    """
    Upload updated leaderboard data to HuggingFace.
    
    Args:
        leaderboard_data: Updated leaderboard data
        token: HuggingFace API token
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Create a temporary file with the leaderboard data
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
            json.dump(leaderboard_data, temp, indent=2)
            temp_path = temp.name
        
        # Upload the file using the HuggingFace Hub API
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="leaderboard.json",
            repo_id=HF_LEADERBOARD_REPO,
            token=token,
            commit_message="Update leaderboard with new submission"
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error uploading leaderboard entry: {e}")
        return False

def upload_to_leaderboard(
    results_path: str,
    model_name: str,
    model_path: str = None,
    hf_token: str = None
) -> bool:
    """
    Upload benchmark results to the HuggingFace leaderboard.
    
    Args:
        results_path: Path to benchmark results JSON
        model_name: Name to identify the model
        model_path: Optional path to model file
        hf_token: HuggingFace API token
        
    Returns:
        True if upload successful, False otherwise
    """
    logger.info(f"Preparing to upload results for model '{model_name}' to leaderboard")
    
    if not hf_token:
        logger.error("HuggingFace token not provided")
        return False
    
    # Validate token
    if not validate_token(hf_token):
        logger.error("Invalid HuggingFace token")
        return False
    
    # Check if results file exists
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return False
    
    # Load the results
    try:
        with open(results_path, 'r') as f:
            benchmark_scores = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load benchmark results: {e}")
        return False
    
    # Import the scoring module to generate a leaderboard entry
    from evaluation.scoring import generate_leaderboard_entry
    
    # Create the leaderboard entry
    entry = generate_leaderboard_entry(benchmark_scores, model_name)
    
    # Get current leaderboard
    leaderboard_data = get_current_leaderboard(hf_token)
    
    # Add the new entry
    entries = leaderboard_data.get("entries", [])
    
    # Check if model already exists in leaderboard
    existing_index = None
    for i, existing_entry in enumerate(entries):
        if existing_entry.get("model_name") == model_name:
            existing_index = i
            break
    
    # Replace existing entry or add new one
    if existing_index is not None:
        entries[existing_index] = entry
        logger.info(f"Replacing existing entry for model '{model_name}'")
    else:
        entries.append(entry)
        logger.info(f"Adding new entry for model '{model_name}'")
    
    # Sort entries by mAP (descending)
    entries.sort(key=lambda x: x.get("overall_metrics", {}).get("mAP", 0), reverse=True)
    
    # Update the leaderboard data
    leaderboard_data["entries"] = entries
    leaderboard_data["last_updated"] = time.strftime("%Y-%m-%d")
    
    # Upload the model if provided and token is available
    if model_path and os.path.exists(model_path) and hf_token:
        try:
            logger.info(f"Uploading model weights to HuggingFace")
            from huggingface_hub import HfApi
            
            # Create a model repo if it doesn't exist
            model_repo_name = f"oceancvbench/{model_name.replace(' ', '-').lower()}"
            
            api = HfApi()
            
            # Check if repo exists
            try:
                api.repo_info(repo_id=model_repo_name, repo_type="model")
                repo_exists = True
            except Exception:
                repo_exists = False
            
            # Create repo if needed
            if not repo_exists:
                logger.info(f"Creating new model repository: {model_repo_name}")
                api.create_repo(
                    repo_id=model_repo_name,
                    repo_type="model",
                    token=hf_token,
                    private=False
                )
            
            # Upload the model
            logger.info(f"Uploading model weights to {model_repo_name}")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=model_repo_name,
                token=hf_token,
                commit_message="Upload model weights from OceanCVBench"
            )
            
            # Also upload the benchmark results to the model repo
            api.upload_file(
                path_or_fileobj=results_path,
                path_in_repo="benchmark_results.json",
                repo_id=model_repo_name,
                token=hf_token,
                commit_message="Upload benchmark results from OceanCVBench"
            )
            
            # Add model repository URL to the leaderboard entry
            entry["model_repo_url"] = f"https://huggingface.co/{model_repo_name}"
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
    
    # Upload the updated leaderboard
    if upload_leaderboard_entry(leaderboard_data, hf_token):
        # Calculate rank
        rank = next((i + 1 for i, item in enumerate(leaderboard_data["entries"]) 
                    if item["model_name"] == model_name), None)
        
        logger.info(f"Successfully uploaded results to leaderboard")
        logger.info(f"Model '{model_name}' is ranked #{rank} of {len(entries)} entries")
        
        # Print leaderboard URL
        print(f"\nLeaderboard URL: https://huggingface.co/{HF_LEADERBOARD_REPO}")
        print(f"Your model '{model_name}' is ranked #{rank} of {len(entries)} entries")
        
        return True
    else:
        logger.error("Failed to upload results to leaderboard")
        return False


def download_leaderboard(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Download the current leaderboard data.
    
    Args:
        output_path: Optional path to save the leaderboard JSON
        
    Returns:
        Dictionary with leaderboard data
    """
    try:
        leaderboard_url = f"https://huggingface.co/{HF_LEADERBOARD_REPO}/resolve/main/leaderboard.json"
        response = requests.get(leaderboard_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to download leaderboard: {response.status_code}")
            return {}
        
        leaderboard_data = response.json()
        
        # Save to file if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(leaderboard_data, f, indent=2)
            
            logger.info(f"Saved leaderboard to {output_path}")
        
        return leaderboard_data
        
    except Exception as e:
        logger.error(f"Error downloading leaderboard: {e}")
        return {}
