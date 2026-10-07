# TerraSpectra

## Hyperspectral Potato Disease Detection & Agricultural Monitoring

TerraSpectra is an AI-powered precision agriculture project for analyzing UAV-based hyperspectral imagery and detecting potato crop disease patterns.

The project combines **hyperspectral image processing, spectral analysis, deep learning, FastAPI inference, Streamlit visualization, and GIS-based monitoring** into a complete prototype for agricultural disease monitoring.

> **Prototype note:** The current dataset does not contain GPS coordinates. The GIS monitoring module therefore uses simulated coordinates for demonstration. Real deployment should use GPS/geospatial information captured during UAV image acquisition.

---

## Project Overview

Traditional RGB imagery may not capture subtle spectral changes associated with early crop stress and disease.

TerraSpectra uses hyperspectral imagery containing multiple spectral bands to analyze crop regions and classify them into disease-related categories.

The current prototype performs:

1. Hyperspectral dataset validation
2. Spectral analysis using PCA
3. Hyperspectral patch extraction
4. PyTorch dataset preparation
5. 3D-CNN model training
6. Data augmentation experiments
7. Focal Loss experiments
8. Balanced sampling experiments
9. Tile-level train/validation splitting
10. Model evaluation
11. Prediction visualization
12. FastAPI model inference
13. Streamlit dashboard integration
14. GIS-based disease monitoring

---

## Key Features

### Hyperspectral Processing

- UAV hyperspectral imagery processing
- 20 spectral bands
- 120 × 120 spatial resolution
- Bounding-box based crop patch extraction
- Pixel normalization
- Spectral analysis

### Machine Learning

- Principal Component Analysis (PCA)
- PyTorch-based 3D Convolutional Neural Network
- Data augmentation
- Weighted sampling
- Focal Loss experimentation
- Tile-level train/validation split
- Classification metrics
- Confusion matrix generation

### AI Inference API

TerraSpectra provides a FastAPI backend for model inference.

```text
POST /predict
```

The API accepts an `.npz` hyperspectral file and returns the model prediction and class probabilities.

### Interactive Dashboard

The Streamlit dashboard provides:

- Overview
- Data Explorer
- Spectral Analysis
- Model Performance
- Prediction
- GIS Monitoring

### GIS Monitoring

The Monitoring page provides:

- Interactive Folium map
- Prediction locations
- Healthy/disease distribution
- Risk-level visualization
- Field-level prediction information
- Geospatial prediction table

---

# Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Computer Vision | Hyperspectral Imaging |
| Dimensionality Reduction | PCA |
| Backend API | FastAPI |
| API Server | Uvicorn |
| Dashboard | Streamlit |
| GIS Visualization | Folium |
| Streamlit GIS Integration | streamlit-folium |
| Visualization | Matplotlib |
| Version Control | Git / GitHub |

---

# Dataset

The project uses UAV hyperspectral imagery from a potato disease detection dataset.

The dataset is not included in this repository because of its size and licensing restrictions.

## Dataset Statistics

| Property | Value |
|---|---:|
| Hyperspectral tiles | 1,115 |
| Spatial resolution | 120 × 120 |
| Spectral bands | 20 |
| Annotation files | 1,115 |
| Bounding boxes | 8,688 |

### Annotation Distribution

| Class | Bounding Boxes |
|---|---:|
| Class 0 | 8,060 |
| Class 1 | 360 |
| Class 2 | 268 |
| **Total** | **8,688** |

---

# Dataset Validation

All hyperspectral tiles were checked against their corresponding annotation files.

```text
Hyperspectral files : 1115
Label files         : 1115
Missing labels      : 0
Missing HSI         : 0
```

This confirms that every hyperspectral tile has a corresponding annotation file in the current dataset.

---

# Hyperspectral Data

A typical hyperspectral image has the shape:

```text
(120, 120, 20)
```

where:

```text
120 = image height
120 = image width
20  = spectral bands
```

The spectral information allows the model to analyze crop characteristics across multiple wavelengths rather than relying only on RGB information.

---

# PCA Spectral Analysis

Principal Component Analysis was applied to analyze the spectral structure of the hyperspectral dataset.

A sample of **100,000 pixels** was used from approximately **1,440,000 pixels**.

The first principal components captured most of the variance:

```text
PC1 : 71.93%
PC2 : 27.31%
PC3 :  0.33%
```

The first three components therefore capture approximately:

```text
99.57% of the sampled variance
```

PCA is used primarily for spectral analysis and visualization rather than replacing the complete hyperspectral input to the 3D-CNN.

---

# Hyperspectral Patch Extraction

The YOLO bounding-box annotations were used to extract crop regions from the hyperspectral images.

The resulting patch dataset contains:

```text
Total patches : 8,688
Patch size    : 32 × 32
Spectral bands: 20
```

The model input is converted into:

```text
[Channels, Spectral Depth, Height, Width]
```

and ultimately processed by the 3D-CNN as:

```text
[Batch, 1, 20, 32, 32]
```

---

# 3D-CNN Architecture

TerraSpectra uses a lightweight 3D Convolutional Neural Network.

The model contains:

```text
Input
  │
  ▼
3D Convolution
  │
  ▼
Max Pooling
  │
  ▼
3D Convolution
  │
  ▼
Max Pooling
  │
  ▼
3D Convolution
  │
  ▼
Adaptive Average Pooling
  │
  ▼
Fully Connected Layer
  │
  ▼
3 Classes
```

The current model contains approximately:

```text
18,243 trainable parameters
```

The 3D convolution layers allow the network to learn both:

- Spatial information
- Spectral information

simultaneously.

---

# Tile-Level Train/Validation Split

To reduce leakage between spatially related samples, the project includes a tile-level split.

```text
Training tiles : 6,963
Validation tiles: 1,725
```

Training distribution:

```text
Class 0 : 6469
Class 1 :  290
Class 2 :  204
```

Validation distribution:

```text
Class 0 : 1591
Class 1 : 70
Class 2 : 64
```

Because the dataset is highly imbalanced, multiple approaches were evaluated.

---

# Model Experiments

The project experimented with:

### 1. Baseline 3D-CNN

A standard 3D-CNN model was trained on the extracted hyperspectral patches.

### 2. Data Augmentation

Training augmentation was introduced to improve generalization.

### 3. Focal Loss

Focal Loss was investigated to address class imbalance.

### 4. Balanced Sampling

Weighted sampling was introduced to provide greater representation for minority classes.

### 5. Tile-Level Training

A tile-level split was implemented to provide a more realistic validation setup.

---

# Tile-Level Model Results

The best tile-level model achieved:

```text
Validation Accuracy : 90.78%
```

The best checkpoint is:

```text
outputs/models/terraspectra_3dcnn_tilesplit_best.pt
```

## Confusion Matrix

```text
                 Predicted
              0     1     2
Actual  0   1535   55     1
        1     39   31     0
        2     62    2     0
```

The model performs strongly on Class 0 but currently struggles to detect Class 2.

This is important because the overall accuracy is strongly influenced by the dominant Class 0 population.

Therefore, **90.78% accuracy should not be interpreted as equally strong performance across all disease classes.**

---

# Classification Results

Current validation performance:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Class 0 | 0.94 | 0.96 | 0.95 |
| Class 1 | 0.35 | 0.44 | 0.39 |
| Class 2 | 0.00 | 0.00 | 0.00 |

The results demonstrate the main challenge of the current prototype:

> **Class imbalance significantly affects minority disease detection.**

Future work should focus on improving minority-class representation and model sensitivity.

---

# FastAPI Inference API

TerraSpectra includes a FastAPI backend for model inference.

Architecture:

```text
Streamlit
    │
    │ POST /predict
    ▼
FastAPI
    │
    ▼
3D-CNN
    │
    ▼
Prediction + Probabilities
    │
    ▼
Streamlit
```

## API Endpoints

### Health Check

```text
GET /health
```

Used to verify that the inference server is running.

### Prediction

```text
POST /predict
```

Accepts an uploaded hyperspectral `.npz` file.

The prediction response contains:

- Predicted class
- Model confidence/probabilities
- Class probability distribution
- Inference information

---

# Running the FastAPI Server

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
uvicorn api.main:app --reload
```

The API will run on:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available through FastAPI's Swagger interface.

---

# Running the Streamlit Dashboard

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
streamlit run frontend\app.py
```

The dashboard provides the complete TerraSpectra interface.

---

# Dashboard Pages

## 1. Overview

Displays the main TerraSpectra project information and dataset/model statistics.

## 2. Data Explorer

Provides access to hyperspectral dataset information and sample exploration.

## 3. Spectral Analysis

Displays spectral information and PCA-based analysis.

## 4. Model Performance

Displays model performance metrics and evaluation results.

## 5. Prediction

Allows users to upload a hyperspectral `.npz` file and obtain an AI prediction through the FastAPI backend.

The page displays:

- Predicted class
- Model confidence
- Probability distribution
- Spectral signature
- Prediction interpretation
- Inference information

## 6. Monitoring

Provides an interactive GIS-style agricultural monitoring interface using Folium.

The map displays predicted crop condition using:

```text
Green  → Healthy / Normal
Orange → Disease Class 1
Red    → Disease Class 2
```

---

# GIS Monitoring

The project generates:

```text
outputs/geospatial_predictions.csv
```

The current prototype contains simulated coordinates because the evaluation dataset does not contain real GPS information.

The generated GIS data contains:

```text
field_id
latitude
longitude
actual_class
predicted_class
class_name
confidence
confidence_status
risk_level
coordinate_source
```

The current GIS prototype uses:

```text
Field ID:
DEMO_FIELD_001
```

and explicitly marks the coordinate source as:

```text
SIMULATED DEMO COORDINATES
```

For real deployment, GPS coordinates should be collected from the UAV acquisition system and associated with each hyperspectral tile.

---

# Project Structure

```text
TerraSpectra_Project_2/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── raw/
│       ├── hyperspectral/
│       └── labels/
│
├── frontend/
│   └── app.py
│
├── outputs/
│   ├── models/
│   ├── hyperspectral_patches.csv
│   ├── tilesplit_confusion_matrix.csv
│   ├── tilesplit_classification_report.csv
│   ├── tilesplit_predictions.csv
│   └── geospatial_predictions.csv
│
├── src/
│   ├── dataset.py
│   ├── model_3dcnn.py
│   ├── train_3dcnn_tilesplit.py
│   ├── create_geospatial_dataset.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```powershell
git clone <YOUR_REPOSITORY_URL>
```

Enter the project:

```powershell
cd TerraSpectra_Project_2
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For GIS support:

```powershell
pip install folium streamlit-folium
```

---

# Important Dataset Setup

The raw hyperspectral dataset is not included in the repository.

Place the dataset in:

```text
data/raw/hyperspectral/0/
data/raw/labels/0/
```

The hyperspectral files should be available as `.npz` files and the corresponding annotations as YOLO-format `.txt` files.

---

# Model Files

Trained model checkpoints are stored under:

```text
outputs/models/
```

The main tile-level checkpoint is:

```text
terraspectra_3dcnn_tilesplit_best.pt
```

---

# Limitations

The current prototype has several limitations.

### 1. Class Imbalance

Class 0 significantly outnumbers the disease classes.

This causes the model to perform much better on the majority class.

### 2. Minority-Class Detection

Class 2 currently has:

```text
Precision : 0.00
Recall    : 0.00
F1        : 0.00
```

Further work is required before the model can be considered reliable for disease detection across all classes.

### 3. Simulated GIS Coordinates

The current GIS monitoring interface uses simulated coordinates.

Real UAV GPS information is required for actual field deployment.

### 4. Prototype Dataset

The current system has been developed and evaluated using a specific UAV potato disease dataset.

Performance may differ significantly on other crops, fields, sensors, or acquisition conditions.

### 5. Early Forecasting

The current prototype focuses on hyperspectral disease classification/monitoring.

Reliable multi-week disease forecasting would require additional temporal data and validation.

---

# Future Improvements

Planned improvements include:

- Improve minority-class detection
- Experiment with stronger 3D-CNN architectures
- Test Vision Transformer / spectral-spatial transformer approaches
- Improve hyperspectral augmentation
- Add real UAV GPS coordinates
- Integrate GIS field boundaries
- Add temporal disease monitoring
- Add real-time UAV data ingestion
- Improve confidence calibration
- Add explainable AI
- Add model comparison dashboard
- Deploy FastAPI and Streamlit services
- Add automated model retraining
- Support larger agricultural fields
- Integrate satellite and UAV data

---

# End-to-End Workflow

The complete TerraSpectra workflow is:

```text
UAV Hyperspectral Image
          │
          ▼
Dataset Validation
          │
          ▼
Spectral Analysis
          │
          ▼
PCA
          │
          ▼
YOLO Bounding Boxes
          │
          ▼
Hyperspectral Patch Extraction
          │
          ▼
Train / Validation Split
          │
          ▼
3D-CNN
          │
          ▼
Model Evaluation
          │
          ▼
FastAPI Inference
          │
          ▼
Streamlit Dashboard
          │
          ├───────────────┐
          ▼               ▼
     Prediction       GIS Monitoring
```

---

# Project Status

| Component | Status |
|---|---|
| Dataset validation | ✅ Complete |
| PCA spectral analysis | ✅ Complete |
| Patch extraction | ✅ Complete |
| PyTorch dataset | ✅ Complete |
| 3D-CNN baseline | ✅ Complete |
| Augmentation experiment | ✅ Complete |
| Focal Loss experiment | ✅ Complete |
| Balanced sampling | ✅ Complete |
| Tile-level split | ✅ Complete |
| Tile-level training | ✅ Complete |
| Model evaluation | ✅ Complete |
| Dashboard integration | ✅ Complete |
| Prediction visualization | ✅ Complete |
| FastAPI inference API | ✅ Complete |
| Streamlit → FastAPI integration | ✅ Complete |
| GIS monitoring | ✅ Complete |
| Final documentation | 🔄 In Progress |
| Final GitHub cleanup | ⏳ Pending |

---

# Disclaimer

TerraSpectra is an academic/research prototype developed for hyperspectral agricultural disease monitoring.

Model predictions should not be treated as professional agricultural diagnosis or direct treatment recommendations.

Real-world deployment requires additional validation using geographically diverse fields, real GPS information, independent test datasets, and domain-expert verification.

---

## Author

**Keyur Nanera**

B.Tech Information Technology  
Sarvajanik College of Engineering and Technology

---

## License

This project is intended for academic and research purposes. Dataset usage is subject to the original dataset's licensing and usage conditions.