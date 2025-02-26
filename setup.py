from setuptools import setup, find_packages
import os

# Read the contents of README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read version from version.py
version = {}
with open(os.path.join("oceancvbench", "version.py")) as f:
    exec(f.read(), version)

setup(
    name="oceancvbench",
    version=version["__version__"],
    author="OceanCVBench Team",
    author_email="info@oceancvbench.org",
    description="Benchmarking toolkit for marine computer vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oceancvbench/oceancvbench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "huggingface-hub>=0.4.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.50.0",
        "requests>=2.25.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "albumentations>=0.5.2",
        "optuna>=2.10.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'flake8>=3.8.0',
            'black>=21.5b2',
            'isort>=5.9.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'myst-parser>=0.15.0',
        ],
    },
    entry_points={
        "console_scripts": [
            "oceancvbench=oceancvbench.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "oceancvbench": ["datasets/metadata/*.yaml"]
    },
)
