# OceanCVBench
OceanCVBench is a comprehensive toolkit for underwater computer vision tasks, specializing in object detection with YOLO model integration. It provides tools for data preprocessing, analysis, and model evaluation specifically designed for underwater imagery. Built by the team at [oceancv.org](https://oceancv.org)

## Key Features

- **Data Preprocessing**: Tools to handle class imbalance, generate synthetic data, detect outliers, and apply underwater-specific augmentations
- **YOLO Integration**: Seamless integration with YOLO models for underwater object detection
- **Analytics**: Class balance reporting, dataset statistics, and visualization tools
- **Evaluation**: Tools to evaluate model performance on underwater imagery

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/atticus-carter/oceancvbench.git
cd oceancvbench

# Install the package
pip install -e .
```

### Option 2: Install via pip (Coming soon!)

```bash
pip install oceancvbench
```

## Quick Start

### Analyzing Class Balance

```python
from oceancvbench.analytics.class_balance_report import CBR, print_cbr_summary

# Generate class balance report from a YOLO dataset
report = CBR("path/to/data.yaml")

# Print a summary of the report
print_cbr_summary(report)
```

### Handling Underrepresented Classes

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

### Running Object Detection

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
├── analytics/           # Tools for dataset analysis
├── data_preprocessing/  # Data preparation and augmentation tools
├── evaluation/          # Model evaluation utilities
├── inference/           # Model inference tools
├── utils/               # Common utilities
└── examples/            # Example scripts
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
