from setuptools import setup, find_packages

setup(
    name="oceancvbench",
    version="0.1.0",
    description="Ocean Computer Vision Benchmarking Tools",
    author="OceanCV Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "matplotlib>=3.1.0",
        "opencv-python>=4.2.0",
        "scipy>=1.4.0",
        "tqdm>=4.45.0",
        "pyyaml>=5.3.0",
        "seaborn>=0.11.0"
    ],
    extras_require={
        "dev": ["pytest", "pylint", "black"],
        "yolo": ["ultralytics>=8.0.0"],
    },
    python_requires=">=3.7",
)
