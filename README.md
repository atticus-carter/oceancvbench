# OceanCVBench

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-orange)](https://oceancvbench.readthedocs.io)

A benchmarking toolkit for marine computer vision models, focused on standardized evaluation and improvement of underwater object detection and segmentation models.

## Features

### Core Features

- **Modular Inference Pipeline**: Streamlined prediction workflow with components for preprocessing, model loading, inference, and postprocessing.
- **Standardized Dataset Handling**: Convert and validate various formats (YOLO, COCO, VOC, LabelMe) to a consistent structure.
- **Automated Hyperparameter Optimization**: Bayesian optimization for YOLO models using Optuna.
- **Marine-Specific Image Augmentation**: Specialized augmentations that simulate underwater conditions like light attenuation, visibility changes, and biofouling.
- **Class Balance Report**: Comprehensive analysis of dataset class distributions with visualization and recommendations.
- **Benchmark Evaluation**: Standardized performance evaluation with metrics specific to underwater object detection challenges.
- **Hugging Face Integration**: Seamless uploading of models and results to the Hugging Face Hub.

### Marine-Specific Tools

- **Underwater Visibility Simulation**: Tools to simulate different turbidity levels and lighting conditions.
- **Biofouling Simulation**: Add realistic growth patterns to simulate lens fouling.
- **Marine Object Detection Improvements**: Specialized techniques to enhance detection of camouflaged and partially visible marine species.

## Installation

### With pip

```bash
pip install oceancvbench
```

### From source

```bash
# Clone the repository
git clone https://github.com/atticus-carter/oceancvbench.git
cd oceancvbench

# Install the package
pip install -e .
```

## Usage Examples

### 1. Analyzing Dataset Quality and Balance

```python
from oceancvbench.analytics.class_balance_report import CBR, print_cbr_summary

# Generate class balance report from a YOLO dataset
report = CBR("path/to/data.yaml")

# Print a summary of the report
print_cbr_summary(report)
```

### 2. Handling Underrepresented Classes

```python
from oceancvbench.data_preprocessing.underrepresented_classes import (
    find_underrepresented_classes,
    generate_synthetic_data
)

# Find underrepresented classes
underrepresented = find_underrepresented_classes(df_bboxes, threshold=50)

# Generate synthetic data for an underrepresented class
if underrepresented:
    synthetic_results = generate_synthetic_data(
        class_id=underrepresented[0],
        count_needed=20,
        df_bboxes=df_bboxes,
        images_dir="path/to/images",
        output_dir="path/to/output"
    )
```

### 3. Running Object Detection

```python
from oceancvbench.inference.localize import localize_images

# Run detection on a folder of images
results = localize_images(
    folder="path/to/images",
    model_path="path/to/yolo_model.pt",
    conf_thresh=0.4,
    iou_thresh=0.5,
    csv=True
)

print(f"Found {len(results)} detections")
```

## Project Structure

```
oceancvbench/
├── oceancvbench/                    # Main package directory
│   ├── __init__.py                  # Package initialization
│   ├── version.py                   # Version information
│   ├── benchmark.py                 # Main benchmarking script
│   ├── optimization_config.py       # Hyperparameter optimization config
│   ├── dataset_handler.py           # Dataset handling utilities
│   ├── hyperparameter_tuning.py     # Hyperparameter optimization
│   ├── analytics/                   # Analytics submodule
│   │   ├── __init__.py
│   │   └── class_balance_report.py  # Class balance analysis tools
│   ├── data_preprocessing/          # Data preprocessing submodule
│   │   ├── __init__.py
│   │   ├── augmentation.py          # Image augmentation utilities
│   │   ├── dataset_loader.py        # Dataset loading utilities
│   │   ├── dataset_stats.py         # Dataset statistics utilities
│   │   ├── outlier_detection.py     # Outlier detection utilities
│   │   └── underrepresented_classes.py  # Class balance correction
│   ├── evaluation/                  # Evaluation submodule
│   │   ├── __init__.py
│   │   ├── evaluation.py            # Main evaluation functionality
│   │   ├── metrics.py               # Performance metrics calculation
│   │   └── scoring.py               # Scoring functionality
│   ├── inference/                   # Inference submodule
│   │   ├── __init__.py
│   │   ├── model_loader.py          # Model loading utilities
│   │   ├── preprocessing.py         # Preprocessing for inference
│   │   ├── inference.py             # Inference execution
│   │   └── postprocessing.py        # Postprocessing results
│   ├── training/                    # Training submodule
│   │   ├── __init__.py
│   │   └── training_pipeline.py     # Training functionality
│   └── utils/                       # Utilities submodule
│       ├── __init__.py
│       ├── common.py                # Common utilities
│       └── huggingface_integration.py  # HuggingFace integration
├── tests/                           # Test directory
│   ├── __init__.py
│   ├── test_analytics/
│   ├── test_data_preprocessing/
│   ├── test_evaluation/
│   └── test_inference/
├── setup.py                         # Package setup script
├── pyproject.toml                   # Project configuration
├── README.md                        # Project README (this file)
└── LICENSE                          # License file
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- OpenCV
- PyTorch (for YOLO integration)
- Matplotlib & Seaborn (for visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
