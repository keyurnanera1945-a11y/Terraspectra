# TerraSpectra

## Hyperspectral Crop Disease Forecasting

TerraSpectra is a precision agriculture project for analyzing hyperspectral imagery for early crop disease monitoring.

The current prototype uses UAV-based hyperspectral imagery from a potato disease detection dataset and is designed as a foundation for a satellite-ready crop monitoring platform.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyTorch
- Streamlit
- Hyperspectral Imaging
- PCA
- 3D CNN
- Computer Vision

## Dataset

The project uses UAV hyperspectral imagery containing selected spectral bands.

Current prototype:

- 1,115 hyperspectral tiles
- 120 × 120 spatial resolution
- 20 spectral bands
- YOLO bounding-box annotations
- 8,688 annotated bounding boxes

The dataset itself is not included in this repository because of its size and licensing restrictions.

## Current Dataset Analysis

All 1,115 hyperspectral tiles have matching annotation files.

```text
Hyperspectral files : 1115
Label files         : 1115
Missing labels      : 0
Missing HSI         : 0