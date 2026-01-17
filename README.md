# Object Detection Model Evaluation & Threshold Optimization

## Project Overview

This project implements a comprehensive evaluation pipeline for object detection models, with a focus on confidence threshold optimization. The system analyzes model predictions against ground truth annotations using polygon-based Intersection over Union (IoU) matching to determine optimal confidence thresholds that balance precision and recall.

## Problem Statement

Object detection models output confidence scores for each prediction, but selecting an appropriate confidence threshold is often arbitrary. This project addresses:
- Systematic evaluation of model performance across confidence thresholds
- Data-driven selection of optimal thresholds
- Analysis of precision-recall tradeoffs
- Professional reporting of findings and recommendations

## Dataset Structure

### Input Files:
1. **Annotations (Ground Truth)** - `anno_df.csv`
   - `image_id`: Unique identifier for each image
   - `id`: Annotation identifier
   - `defect_class_id`: Defect class (all class 7 in provided data)
   - `label`: Defect label description
   - `xy`: Polygon coordinates in `x1,y1,x2,y2,...` format
   - `x`: polygon coordinates in `x1,x2,x3,...` format
   - `y`: polygon coordiantes in `y1,y2,y3,..` format 

2. **Predictions (Model Outputs)** - `pred_df.csv`
   - `image_id`: Unique identifier for each image
   - `prediction_id`: Prediction identifier
   - `confidence`: Model confidence score (0-1)
   - `polygon_id`: Polygon identifier
   - `prediction_class`: Predicted class (all "Defect" in provided data)
   - `xy`: Polygon coordinates in `x1,y1,x2,y2,...` format

## Project Structure

Object-Defect-Detection   
├── data/   
│ ├── anno_df.csv # Annotations dataset  
│ ├── pred_df.csv # model output predictions dataset  
├── notebooks/ # Jupyter notebooks  
│ └── detection.ipynb # Main analysis notebook   
├── scripts/ # Core analysis modules  
│ ├── init.py # Package initialization  
│ ├── eda.py # Exploratory Data Analysis  
│ ├── polygon_utils.py # Polygon parsing and IoU calculation  
│ ├── matching_utils.py # Prediction-Ground truth matching  
│ ├── evaluation_utils.py # Metrics calculation and optimization  
│ └── visualization_utils.py # Plotting and visualization   
├── venv # Virtual environment  
├── .gitignore # Files and folders to ignore in source control   
├── README.md # Project overview and usage instructions  
└── requirements.txt # Python dependencies for reproducibility and setup  


## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

# Quick Start 

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/habrev/object-defect-detection.git
    ```

2.  **Set up the environment:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Place annotation and prediction CSV files in the project directory in data folder**
4. **Run the notebook cells sequentially**

# Contributing
1. **Fork the repository**

2. **Create a feature branch**

3. **Add tests for new functionality**

4. **Submit a pull request**

# License
**This project is provided for Job application assessment purposes.**
