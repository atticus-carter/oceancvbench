"""
Main orchestration module for OceanCVBench benchmarking.

This script provides a standardized benchmarking pipeline for marine 
computer vision models, allowing users to evaluate model performance 
across diverse underwater conditions.
"""

import os
import argparse
import json
import logging
from pathlib import Path
import time
import yaml
from typing import Dict, Any, Optional

from data_preprocessing.dataset_loader import download_benchmark_datasets
from oceancvbench.training.training_pipeline import fine_tune_model
from evaluation import evaluate_model_on_benchmark
from evaluation.scoring import calculate_benchmark_scores, save_benchmark_results
from utils.huggingface_integration import upload_to_leaderboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("oceancvbench.benchmark")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OceanCVBench: Benchmark marine computer vision models")
    
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Path to YOLO model weights (.pt file)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        default=640, 
        help="Image size for training and inference"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data", 
        help="Directory for datasets"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs", 
        help="Directory for output files"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of worker threads for data loading"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="", 
        help="Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)"
    )
    parser.add_argument(
        "--upload-results", 
        action="store_true", 
        help="Upload results to HuggingFace leaderboard"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip fine-tuning and use provided weights directly for evaluation"
    )
    parser.add_argument(
        "--hf-token", 
        type=str, 
        default="", 
        help="HuggingFace API token for uploading results"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="", 
        help="Name to identify your model on the leaderboard"
    )
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input arguments."""
    # Check if weights file exists
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Check if we need to upload results but no token provided
    if args.upload_results and not args.hf_token:
        logger.warning("--upload-results flag is set but no HuggingFace token provided. Will not upload results.")
        args.upload_results = False
    
    # Check if model name is provided for the leaderboard
    if args.upload_results and not args.model_name:
        args.model_name = f"model_{int(time.time())}"
        logger.warning(f"No model name provided. Using default: {args.model_name}")

def run_benchmark(args) -> Dict[str, Any]:
    """
    Run the complete benchmarking pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing benchmark results
    """
    benchmark_start_time = time.time()
    logger.info("Starting OceanCVBench benchmarking pipeline")
    
    # Step 1: Download and prepare datasets
    logger.info("Step 1: Downloading benchmark datasets")
    data_yaml_path = download_benchmark_datasets(args.data_dir)
    
    # Prepare output paths
    fine_tuned_model_path = os.path.join(args.output_dir, "benchmark_model.pt")
    results_json_path = os.path.join(args.output_dir, "benchmark_results.json")
    
    # Step 2: Fine-tune the model
    if not args.skip_training:
        logger.info("Step 2: Fine-tuning model")
        fine_tune_model(
            weights_path=args.weights,
            data_yaml_path=data_yaml_path,
            output_path=fine_tuned_model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            workers=args.workers,
            device=args.device
        )
    else:
        logger.info("Skipping fine-tuning as requested")
        fine_tuned_model_path = args.weights
    
    # Step 3: Run inference on benchmark test set
    logger.info("Step 3: Running inference on benchmark test set")
    inference_results = evaluate_model_on_benchmark(
        model_path=fine_tuned_model_path,
        benchmark_dir=os.path.join(args.data_dir, "test", "oceanbench"),
        img_size=args.img_size,
        device=args.device
    )
    
    # Step 4: Calculate benchmark scores
    logger.info("Step 4: Calculating benchmark scores")
    benchmark_scores = calculate_benchmark_scores(inference_results)
    
    # Step 5: Save results
    logger.info("Step 5: Saving benchmark results")
    save_benchmark_results(benchmark_scores, results_json_path)
    
    # Step 6: Upload results to leaderboard if requested
    if args.upload_results:
        logger.info("Step 6: Uploading results to HuggingFace leaderboard")
        upload_to_leaderboard(
            results_path=results_json_path,
            model_name=args.model_name,
            model_path=fine_tuned_model_path,
            hf_token=args.hf_token
        )
    
    # Calculate total benchmark time
    total_time = (time.time() - benchmark_start_time) / 60
    logger.info(f"OceanCVBench benchmark completed in {total_time:.2f} minutes")
    
    # Return the scores
    return benchmark_scores

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Run the benchmark
        benchmark_scores = run_benchmark(args)
        
        # Print summary of results
        print("\n==== OceanCVBench Results Summary ====")
        print(f"Model: {os.path.basename(args.weights)}")
        
        print("\nPerformance by environmental condition:")
        for folder, metrics in benchmark_scores.items():
            if folder != "Overall Score":
                print(f"  {folder}: mAP={metrics['mAP']:.3f}, F1={metrics['F1-score']:.3f}")
        
        overall = benchmark_scores["Overall Score"]
        print("\nOverall Performance:")
        print(f"  mAP: {overall['mAP']:.3f}")
        print(f"  F1-score: {overall['F1-score']:.3f}")
        print(f"  Precision: {overall['Precision']:.3f}")
        print(f"  Recall: {overall['Recall']:.3f}")
        
        output_path = os.path.join(args.output_dir, "benchmark_results.json")
        print(f"\nDetailed results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
