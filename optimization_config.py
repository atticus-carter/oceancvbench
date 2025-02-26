"""
Configuration module for hyperparameter optimization tasks.

This module defines parameter spaces and optimization strategies for
different types of models and datasets.
"""

from typing import Dict, Any, List, Optional, Union

# YOLO parameter space definition
YOLO_PARAM_SPACE = {
    # Learning rate and scheduler parameters
    "lr_params": {
        "lr0": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True},
        "lrf": {"type": "float", "min": 0.01, "max": 0.2},
        "momentum": {"type": "float", "min": 0.6, "max": 0.98},
        "weight_decay": {"type": "float", "min": 0.0, "max": 0.001},
        "warmup_epochs": {"type": "int", "min": 1, "max": 5},
        "warmup_momentum": {"type": "float", "min": 0.5, "max": 0.95},
        "warmup_bias_lr": {"type": "float", "min": 0.05, "max": 0.2},
        "cos_lr": {"type": "categorical", "choices": [True, False]},
    },
    
    # Data augmentation parameters
    "augmentation_params": {
        "hsv_h": {"type": "float", "min": 0.0, "max": 0.1},
        "hsv_s": {"type": "float", "min": 0.0, "max": 0.9},
        "hsv_v": {"type": "float", "min": 0.0, "max": 0.9},
        "degrees": {"type": "float", "min": 0.0, "max": 10.0},
        "translate": {"type": "float", "min": 0.0, "max": 0.2},
        "scale": {"type": "float", "min": 0.0, "max": 0.9},
        "shear": {"type": "float", "min": 0.0, "max": 10.0},
        "perspective": {"type": "float", "min": 0.0, "max": 0.001},
        "flipud": {"type": "float", "min": 0.0, "max": 0.5},
        "fliplr": {"type": "float", "min": 0.0, "max": 0.5},
        "mosaic": {"type": "float", "min": 0.0, "max": 1.0},
        "mixup": {"type": "float", "min": 0.0, "max": 0.5},
    },
    
    # Model hyperparameters
    "model_params": {
        "box": {"type": "float", "min": 5.0, "max": 10.0},
        "cls": {"type": "float", "min": 0.2, "max": 1.0},
        "dfl": {"type": "float", "min": 1.0, "max": 3.0},
        "cls_pw": {"type": "float", "min": 0.5, "max": 2.0},
        "obj": {"type": "float", "min": 0.3, "max": 1.0},
        "obj_pw": {"type": "float", "min": 0.5, "max": 2.0},
        "iou_t": {"type": "float", "min": 0.2, "max": 0.7},
        "anchor_t": {"type": "float", "min": 2.0, "max": 8.0},
        "fl_gamma": {"type": "float", "min": 0.0, "max": 3.0},
        "dropout": {"type": "float", "min": 0.0, "max": 0.2}
    },
    
    # Training parameters
    "training_params": {
        "batch_size": {"type": "categorical", "choices": [8, 16, 24, 32, 48, 64]},
        "imgsz": {"type": "categorical", "choices": [416, 512, 640, 768]},
        "patience": {"type": "int", "min": 5, "max": 50},
        "use_silu": {"type": "categorical", "choices": [True, False]},
        "optimizer": {"type": "categorical", "choices": ["SGD", "Adam", "AdamW", "RMSProp"]},
    },
    
    # Marine-specific parameters
    "marine_params": {
        "underwater_gamma": {"type": "float", "min": 0.0, "max": 0.3},
        "underwater_blur": {"type": "float", "min": 0.0, "max": 0.3},
        "underwater_noise": {"type": "float", "min": 0.0, "max": 0.3},
        "underwater_color_shift": {"type": "float", "min": 0.0, "max": 0.2},
    }
}

# Optimization strategies for different tasks
OPTIMIZATION_STRATEGIES = {
    "quick": {
        "n_trials": 10,
        "timeout": 7200,  # 2 hours
        "epochs_per_trial": 20,
        "pruning": True,
        "pruning_warmup": 5
    },
    "standard": {
        "n_trials": 30,
        "timeout": 86400,  # 24 hours
        "epochs_per_trial": 30,
        "pruning": True,
        "pruning_warmup": 10
    },
    "thorough": {
        "n_trials": 50,
        "timeout": 259200,  # 3 days
        "epochs_per_trial": 50,
        "pruning": True,
        "pruning_warmup": 15
    }
}

# Parameter importance weights based on prior knowledge
# Higher weight means this parameter should be explored more thoroughly
PARAMETER_IMPORTANCE = {
    "lr0": 10,
    "batch_size": 8,
    "imgsz": 7,
    "mosaic": 7,
    "mixup": 6,
    "box": 6,
    "cls": 5,
    "hsv_h": 3,
    "hsv_s": 3,
    "hsv_v": 3,
    "underwater_gamma": 5,
    "underwater_blur": 4,
    "underwater_noise": 4,
}

# Parameter groupings for ablation studies
PARAMETER_GROUPS = {
    "basic": ["lr0", "batch_size", "imgsz"],
    "augmentation": ["mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", 
                   "scale", "shear", "perspective", "flipud", "fliplr"],
    "loss_weights": ["box", "cls", "dfl", "cls_pw", "obj", "obj_pw"],
    "optimizer": ["lr0", "lrf", "momentum", "weight_decay", "optimizer"],
    "marine_specific": ["underwater_gamma", "underwater_blur", "underwater_noise", 
                      "underwater_color_shift"]
}


def get_optimization_config(strategy: str = "standard", task_type: str = "marine_detection") -> Dict[str, Any]:
    """
    Get optimization configuration for a specific strategy and task type.
    
    Args:
        strategy: Optimization strategy ('quick', 'standard', 'thorough')
        task_type: Type of task ('marine_detection', 'general_detection', etc.)
        
    Returns:
        Dictionary with optimization configuration
    """
    config = {}
    
    # Get base strategy config
    if strategy in OPTIMIZATION_STRATEGIES:
        config.update(OPTIMIZATION_STRATEGIES[strategy])
    else:
        config.update(OPTIMIZATION_STRATEGIES["standard"])
    
    # Adjust for specific task type
    if task_type == "marine_detection":
        # Focus more on underwater augmentations
        param_space = {}
        for group_name, params in YOLO_PARAM_SPACE.items():
            param_space.update(params)
        
        config["param_space"] = param_space
        
        # Add task-specific pruner settings
        config["pruner"] = {
            "type": "median",
            "startup_trials": 5,
            "warmup_steps": 5,
            "interval_steps": 1
        }
        
        # Recommend evaluation metrics for marine detection
        config["suggested_metrics"] = ["map50", "map50-95", "recall"]
        
    elif task_type == "general_detection":
        # Use standard parameter space but exclude marine-specific parameters
        param_space = {}
        for group_name, params in YOLO_PARAM_SPACE.items():
            if group_name != "marine_params":
                param_space.update(params)
        
        config["param_space"] = param_space
        config["suggested_metrics"] = ["map50"]
        
    return config


def suggest_best_optimization_strategy(dataset_size: int, available_time_hours: float) -> str:
    """
    Suggest the best optimization strategy based on dataset size and available time.
    
    Args:
        dataset_size: Number of images in the dataset
        available_time_hours: Available time in hours
        
    Returns:
        Recommended strategy name
    """
    if dataset_size < 200 or available_time_hours < 3:
        return "quick"
    elif dataset_size < 1000 or available_time_hours < 24:
        return "standard"
    else:
        return "thorough"


def get_parameter_subset(group_names: List[str]) -> Dict[str, Any]:
    """
    Get a subset of parameters based on named parameter groups.
    
    Args:
        group_names: List of group names to include
        
    Returns:
        Dictionary with parameter configuration
    """
    selected_params = set()
    
    for group_name in group_names:
        if group_name in PARAMETER_GROUPS:
            selected_params.update(PARAMETER_GROUPS[group_name])
    
    # Construct parameter space
    param_space = {}
    for group, params in YOLO_PARAM_SPACE.items():
        for param_name, param_config in params.items():
            if param_name in selected_params:
                param_space[param_name] = param_config
    
    return param_space