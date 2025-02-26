"""
Hyperparameter optimization module for OceanCVBench.

This module implements Bayesian optimization using Optuna to find optimal
hyperparameters for YOLO models on marine imagery.
"""

import os
import yaml
import json
import logging
import tempfile
import time
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import shutil

# Configure logging
logger = logging.getLogger("oceancvbench.hyperparameter_tuning")

# Try importing optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_INSTALLED = True
except ImportError:
    logger.warning("Optuna not installed. Hyperparameter tuning will not be available.")
    logger.warning("Install with: pip install optuna")
    OPTUNA_INSTALLED = False


class YOLOHyperparameterOptimizer:
    """
    Hyperparameter optimizer for YOLO models using Bayesian optimization.
    """
    
    def __init__(self, 
                data_yaml: str,
                initial_weights: str,
                output_dir: str,
                n_trials: int = 20,
                timeout: Optional[int] = None,
                eval_metric: str = "map50",
                device: str = "",
                seed: int = 42,
                study_name: Optional[str] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            data_yaml: Path to the YAML file with dataset configuration
            initial_weights: Path to the initial weights file (.pt)
            output_dir: Directory to save results
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            eval_metric: Evaluation metric to optimize ('map50', 'map50-95', 'f1', etc.)
            device: Device for training ('', '0', '0,1', etc.)
            seed: Random seed for reproducibility
            study_name: Optional name for the optimization study
        """
        if not OPTUNA_INSTALLED:
            raise ImportError("Optuna is required for hyperparameter optimization")
            
        self.data_yaml = os.path.abspath(data_yaml)
        self.initial_weights = os.path.abspath(initial_weights)
        self.output_dir = os.path.abspath(output_dir)
        self.n_trials = n_trials
        self.timeout = timeout
        self.eval_metric = eval_metric
        self.device = device
        self.seed = seed
        self.study_name = study_name or f"yolo_opt_{int(time.time())}"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize best parameters and metrics
        self.best_params = None
        self.best_metrics = None
        self.optimization_history = []
        
    def _validate_inputs(self):
        """Validate input parameters."""
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"Data YAML file not found: {self.data_yaml}")
        
        if not os.path.exists(self.initial_weights):
            raise FileNotFoundError(f"Initial weights file not found: {self.initial_weights}")
        
        # Load and validate data yaml
        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                
            # Check required keys
            required_keys = ['train', 'val', 'names']
            missing_keys = [key for key in required_keys if key not in data_config]
            if missing_keys:
                raise ValueError(f"Missing required keys in data YAML: {missing_keys}")
                
            # Check if paths exist
            train_path = data_config.get('train', '')
            val_path = data_config.get('val', '')
            
            if not os.path.exists(train_path):
                logger.warning(f"Training path not found: {train_path}")
            
            if not os.path.exists(val_path):
                logger.warning(f"Validation path not found: {val_path}")
                
        except Exception as e:
            raise ValueError(f"Error validating data YAML: {e}")
    
    def define_parameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the parameter search space for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Learning rate and scheduler
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
            "lrf": trial.suggest_float("lrf", 0.01, 0.2),
            
            # Regularization
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.001),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            
            # Data augmentation
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
            "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
            "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
            "degrees": trial.suggest_float("degrees", 0.0, 10.0),
            "translate": trial.suggest_float("translate", 0.0, 0.2),
            "scale": trial.suggest_float("scale", 0.0, 0.9),
            "shear": trial.suggest_float("shear", 0.0, 10.0),
            "perspective": trial.suggest_float("perspective", 0.0, 0.001),
            "flipud": trial.suggest_float("flipud", 0.0, 0.5),
            "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
            "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
            "mixup": trial.suggest_float("mixup", 0.0, 0.5),
            
            # Batch size and image size
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 24, 32]),
            "imgsz": trial.suggest_categorical("imgsz", [512, 640, 768]),
            
            # Model-specific
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            
            # Training specific
            "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
            "cos_lr": trial.suggest_categorical("cos_lr", [True, False]),
            "use_silu": trial.suggest_categorical("use_silu", [True, False])
        }
        
        # Add ocean-specific parameters for underwater images
        params.update({
            # Underwater-specific augmentations
            "underwater_gamma": trial.suggest_float("underwater_gamma", 0.0, 0.3),
            "underwater_blur": trial.suggest_float("underwater_blur", 0.0, 0.3),
            "underwater_noise": trial.suggest_float("underwater_noise", 0.0, 0.3),
        })
        
        return params
    
    def _prepare_yolo_config(self, params: Dict[str, Any], trial_dir: str) -> str:
        """
        Prepare a YOLO configuration file with the trial's hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            trial_dir: Directory for this trial
            
        Returns:
            Path to the created config file
        """
        # Base configuration (inherits from YOLO defaults)
        config = {
            # Dataset
            "path": os.path.dirname(self.data_yaml),
            "data": self.data_yaml,
            
            # Training settings
            "epochs": 30,  # Reduced for optimization trials
            "batch": params["batch_size"],
            "imgsz": params["imgsz"],
            "device": self.device,
            "workers": 4,
            "optimizer": "SGD",  # or "Adam", "AdamW"
            
            # Hyperparameters
            "lr0": params["lr0"],
            "lrf": params["lrf"],
            "momentum": params["momentum"],
            "weight_decay": params["weight_decay"],
            "warmup_epochs": params["warmup_epochs"],
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,  # Box loss gain
            "cls": 0.5,  # Class loss gain
            "dfl": 1.5,  # DFL loss gain
            "fl_gamma": 0.0,  # Focal loss gamma
            "cos_lr": params["cos_lr"],
            
            # Augmentation settings
            "hsv_h": params["hsv_h"],
            "hsv_s": params["hsv_s"],
            "hsv_v": params["hsv_v"],
            "degrees": params["degrees"],
            "translate": params["translate"],
            "scale": params["scale"],
            "shear": params["shear"],
            "perspective": params["perspective"],
            "flipud": params["flipud"],
            "fliplr": params["fliplr"],
            "mosaic": params["mosaic"],
            "mixup": params["mixup"],
            
            # Custom underwater augmentation params
            "hyp": {
                "underwater_gamma": params["underwater_gamma"],
                "underwater_blur": params["underwater_blur"],
                "underwater_noise": params["underwater_noise"],
            },
            
            # Saving and logging settings
            "project": trial_dir,
            "name": "train",
            "exist_ok": True,
            "pretrained": self.initial_weights,
            "verbose": False,
            "seed": self.seed,
            "save": True,
            "save_period": -1,  # Save only final model to reduce disk space
            "patience": 10  # Early stopping patience
        }
        
        # Save configuration to file
        config_path = os.path.join(trial_dir, "trial_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return config_path
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Evaluation metric value to maximize
        """
        # Create a directory for this trial
        trial_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Sample hyperparameters
        params = self.define_parameter_space(trial)
        
        # Log hyperparameters for this trial
        logger.info(f"Trial {trial.number}: {params}")
        
        # Create YOLO config for this trial
        config_path = self._prepare_yolo_config(params, trial_dir)
        
        try:
            # Run training with these hyperparameters
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(self.initial_weights)
            
            # Start training with the config
            results = model.train(cfg=config_path)
            
            # Extract metrics
            metrics = {}
            
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            elif isinstance(results, dict):
                metrics = results
            
            # Get the validation metric we want to optimize
            val_metric = 0.0
            
            if self.eval_metric == "map50":
                val_metric = metrics.get("metrics/mAP50(B)", 0)
            elif self.eval_metric == "map50-95":
                val_metric = metrics.get("metrics/mAP50-95(B)", 0)
            elif self.eval_metric == "precision":
                val_metric = metrics.get("metrics/precision(B)", 0)
            elif self.eval_metric == "recall":
                val_metric = metrics.get("metrics/recall(B)", 0)
            elif self.eval_metric == "f1":
                # Calculate F1 if not directly provided
                precision = metrics.get("metrics/precision(B)", 0)
                recall = metrics.get("metrics/recall(B)", 0)
                if precision + recall > 0:
                    val_metric = 2 * precision * recall / (precision + recall)
            else:
                val_metric = metrics.get(self.eval_metric, 0)
            
            # Store metrics for this trial
            trial_results = {
                "trial": trial.number,
                "params": params,
                "metrics": metrics,
                "eval_metric": self.eval_metric,
                "eval_value": val_metric
            }
            
            # Save trial results
            with open(os.path.join(trial_dir, "results.json"), "w") as f:
                json.dump(trial_results, f, indent=4)
            
            # Save to optimization history
            self.optimization_history.append(trial_results)
            
            # Report intermediate values for pruning
            if hasattr(results, 'epoch'):
                trial.report(val_metric, step=results.epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                return optuna.exceptions.TrialPruned()
            
            return val_metric
            
        except Exception as e:
            logger.error(f"Error during trial {trial.number}: {e}")
            # Return a very low value on error
            return -1.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with best parameters and metrics
        """
        if not OPTUNA_INSTALLED:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        logger.info(f"Target metric: {self.eval_metric}")
        
        # Create a sampler and pruner
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        # Create or load study
        storage_name = f"sqlite:///{os.path.join(self.output_dir, 'optuna_study.db')}"
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_name,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
            direction="maximize"
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best trial
        best_trial = study.best_trial
        
        # Update best parameters and metrics
        self.best_params = best_trial.params
        
        # Find the corresponding metrics
        for trial_result in self.optimization_history:
            if trial_result["trial"] == best_trial.number:
                self.best_metrics = trial_result["metrics"]
                break
        
        # Save best parameters
        best_params_path = os.path.join(self.output_dir, "best_parameters.json")
        with open(best_params_path, "w") as f:
            json.dump({
                "best_params": self.best_params,
                "best_metrics": self.best_metrics,
                "best_value": best_trial.value,
                "best_trial": best_trial.number,
                "n_trials": self.n_trials,
                "eval_metric": self.eval_metric
            }, f, indent=4)
        
        # Copy best model to output dir
        best_model_path = os.path.join(
            self.output_dir, 
            f"trial_{best_trial.number}", 
            "train", 
            "weights", 
            "best.pt"
        )
        
        if os.path.exists(best_model_path):
            shutil.copy2(
                best_model_path, 
                os.path.join(self.output_dir, "best_model.pt")
            )
        
        # Generate optimization report
        self._generate_report()
        
        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best {self.eval_metric}: {best_trial.value:.4f}")
        logger.info(f"Best parameters saved to {best_params_path}")
        
        return {
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_value": best_trial.value,
            "best_trial": best_trial.number,
            "n_trials": len(study.trials),
            "eval_metric": self.eval_metric
        }
        
    def _generate_report(self):
        """Generate a report with optimization results."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create a DataFrame from optimization history
            data = []
            for trial in self.optimization_history:
                row = {
                    "trial": trial["trial"],
                    "value": trial["eval_value"],
                }
                # Add parameters
                for param_name, param_value in trial["params"].items():
                    row[param_name] = param_value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(os.path.join(self.output_dir, "optimization_history.csv"), index=False)
            
            # Plot optimization history
            plt.figure(figsize=(10, 6))
            plt.plot(df["trial"], df["value"], "o-", color="blue", alpha=0.6)
            plt.axhline(y=df["value"].max(), color="green", linestyle="--", alpha=0.3, 
                       label=f"Best: {df['value'].max():.4f}")
            plt.grid(alpha=0.3)
            plt.xlabel("Trial")
            plt.ylabel(f"Validation {self.eval_metric}")
            plt.title("Hyperparameter Optimization History")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "optimization_history.png"), dpi=300)
            plt.close()
            
            # Plot parameter importance
            if len(df) > 5:  # Need enough trials for meaningful importance
                plt.figure(figsize=(12, 8))
                
                # Compute correlation with target metric
                param_cols = [col for col in df.columns if col not in ["trial", "value"]]
                correlations = {}
                
                for param in param_cols:
                    if df[param].nunique() > 1:  # Skip constant parameters
                        correlations[param] = abs(df[param].corr(df["value"]))
                
                # Sort by correlation
                sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                params = [x[0] for x in sorted_correlations[:15]]  # Top 15 parameters
                values = [x[1] for x in sorted_correlations[:15]]
                
                # Create bar chart
                plt.barh(params, values, color="darkblue", alpha=0.6)
                plt.xlabel("Correlation with target metric")
                plt.ylabel("Parameter")
                plt.title("Parameter Importance (by correlation)")
                plt.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "parameter_importance.png"), dpi=300)
                plt.close()
                
            # Generate HTML report
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Hyperparameter Optimization Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metrics {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                    .highlight {{ font-weight: bold; color: #2a6496; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Hyperparameter Optimization Report</h1>
                    
                    <h2>Overview</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Study Name</td><td>{self.study_name}</td></tr>
                        <tr><td>Number of Trials</td><td>{self.n_trials}</td></tr>
                        <tr><td>Evaluation Metric</td><td>{self.eval_metric}</td></tr>
                        <tr><td>Best Trial</td><td>{df['value'].idxmax() if not df.empty else 'N/A'}</td></tr>
                        <tr><td>Best Metric Value</td><td class="highlight">{df['value'].max() if not df.empty else 'N/A'}</td></tr>
                    </table>
                    
                    <h2>Best Hyperparameters</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
            """
            
            # Add best parameters to the report
            if self.best_params:
                for param, value in self.best_params.items():
                    html_report += f"<tr><td>{param}</td><td>{value}</td></tr>\n"
            
            html_report += """
                    </table>
                    
                    <h2>Optimization History</h2>
                    <img src="optimization_history.png" alt="Optimization History">
                    
                    <h2>Parameter Importance</h2>
                    <img src="parameter_importance.png" alt="Parameter Importance">
                    
                    <h2>All Trials</h2>
                    <table>
                        <tr><th>Trial</th><th>Value</th><th>Top Parameters</th></tr>
            """
            
            # Add trial data
            if not df.empty:
                # Sort by metric value (descending)
                df_sorted = df.sort_values("value", ascending=False)
                
                for _, row in df_sorted.iterrows():
                    # Select a few important parameters to show
                    important_params = ["lr0", "batch_size", "imgsz"]
                    param_str = ", ".join([f"{p}: {row[p]}" for p in important_params if p in row])
                    
                    html_report += f"""
                        <tr>
                            <td>{int(row['trial'])}</td>
                            <td>{row['value']:.4f}</td>
                            <td>{param_str}</td>
                        </tr>
                    """
            
            html_report += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            with open(os.path.join(self.output_dir, "optimization_report.html"), "w") as f:
                f.write(html_report)
                
            logger.info(f"Generated optimization report at {os.path.join(self.output_dir, 'optimization_report.html')}")
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")


# Wrapper function for external scripts
def optimize_hyperparameters(
    data_yaml: str,
    initial_weights: str,
    output_dir: str,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    eval_metric: str = "map50",
    device: str = "",
    seed: int = 42,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for a YOLO model.
    
    Args:
        data_yaml: Path to the YAML file with dataset configuration
        initial_weights: Path to the initial weights file (.pt)
        output_dir: Directory to save results
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds
        eval_metric: Evaluation metric to optimize ('map50', 'map50-95', 'f1', etc.)
        device: Device for training ('', '0', '0,1', etc.)
        seed: Random seed for reproducibility
        study_name: Optional name for the optimization study
    
    Returns:
        Dictionary with best parameters and metrics
    """
    if not OPTUNA_INSTALLED:
        raise ImportError("Optuna is required for hyperparameter optimization")
    
    optimizer = YOLOHyperparameterOptimizer(
        data_yaml=data_yaml,
        initial_weights=initial_weights,
        output_dir=output_dir,
        n_trials=n_trials,
        timeout=timeout,
        eval_metric=eval_metric,
        device=device,
        seed=seed,
        study_name=study_name
    )
    
    return optimizer.optimize()


# Run this module directly for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for YOLO model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, required=True, help='Path to initial weights')
    parser.add_argument('--output-dir', type=str, default='./hpo_results', help='Output directory')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--metric', type=str, default='map50', help='Evaluation metric')
    parser.add_argument('--device', type=str, default='', help='Device for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--study-name', type=str, default=None, help='Study name')
    
    args = parser.parse_args()
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run optimization
    optimize_hyperparameters(
        data_yaml=args.data,
        initial_weights=args.weights,
        output_dir=args.output_dir,
        n_trials=args.trials,
        timeout=args.timeout,
        eval_metric=args.metric,
        device=args.device,
        seed=args.seed,
        study_name=args.study_name
    )
