# Passenger Overloading Detection System

A computer vision system for detecting passenger overloading violations on motorcycles using deep learning, pose estimation, and machine learning classification.

## 🎯 Project Overview

This system automatically detects various types of passenger overloading violations on motorcycles:

- **OVERLOADING**: Multiple passengers on a single motorcycle
- **SIDE_SADDLE**: Passengers sitting sideways on the motorcycle
- **REVERSE_SIDE_SADDLE**: Passengers sitting sideways in reverse direction
- **CHILD_IN_FRONT**: Children seated in front of the driver

## 🏗️ Architecture

The system uses a multi-stage pipeline:

1. **Object Detection**: YOLO11x model detects persons and motorcycles
2. **Pose Estimation**: YOLO11x-pose extracts 17 keypoints from detected persons
3. **Feature Extraction**: Computes depth, age, pose angles, and geometric features
4. **Classification**: HistGradientBoostingClassifier predicts violation types

## 📁 Project Structure

```
passenger-overloading/
├── models/                          # Trained models
│   ├── yolo11x.pt                  # YOLO detection model
│   ├── yolo11x-pose.pt             # YOLO pose estimation model
│   └── violation_classifier_balanced.pkl  # ML classifier
├── scripts/                         # Main scripts
│   ├── train_classifier.py          # Train the violation classifier
│   ├── predict_classifier.py        # Run predictions on annotations
│   ├── test.py                     # End-to-end testing on images
│   ├── extract_objects.py          # Extract object detections
│   ├── extract_pose.py             # Extract pose keypoints
│   ├── extract_depth.py            # Extract depth information
│   ├── extract_age.py              # Extract age estimates
│   ├── evaluate.py                 # Evaluate model performance
│   ├── generate_labels.py          # Generate training labels
│   ├── rename_images.py            # Utility for image renaming
│   └── test.py                     # Main testing script
├── data/                           # Dataset
│   ├── images/                     # Input images
│   └── test/                       # Test images
├── annotations/                    # Extracted features and labels
│   ├── detect_boxes.csv           # Object detection results
│   ├── keypoints.csv              # Pose keypoints
│   ├── depth.csv                  # Depth estimates
│   ├── age.csv                    # Age estimates
│   ├── labels.csv                 # Ground truth labels
│   └── predictions.csv            # Model predictions
├── runs/                          # Training outputs
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd passenger-overloading
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   ```bash
   # Models should be placed in the models/ directory
   # - yolo11x.pt (109MB)
   # - yolo11x-pose.pt (113MB)
   # - violation_classifier_balanced.pkl (13MB)
   ```

### Usage

#### 1. End-to-End Testing

Test the complete pipeline on a single image:

```bash
python scripts/test.py path/to/your/image.jpg
```

This will:
- Detect persons and motorcycles
- Extract pose keypoints
- Compute depth and age estimates
- Classify violations
- Output results

#### 2. Training the Classifier

If you have labeled data:

```bash
python scripts/train_classifier.py
```

This trains a multi-output classifier on the extracted features.

#### 3. Batch Prediction

For batch processing of annotated data:

```bash
python scripts/predict_classifier.py
```

## 🔧 Technical Details

### Feature Engineering

The system extracts comprehensive features for each detected person:

**Pose Features (51 dimensions)**:
- 17 keypoints × 3 (x, y, confidence)
- Normalized relative to left hip (kpt11)

**Geometric Features**:
- Bounding box dimensions and area
- Shoulder and hip differences
- Head-to-seat ratio
- Front-half overlap with motorcycle

**Pose Angles**:
- Left/right hip angles for side-saddle detection
- Left/right torso lean angles

**Context Features**:
- Number of persons per motorcycle
- Average depth estimate
- Age estimate (for helmet detection)
- Motorcycle association flag

### Model Architecture

- **Detection**: YOLO11x with COCO classes
- **Pose**: YOLO11x-pose with 17 keypoints
- **Depth**: MiDaS DPT_Large for depth estimation
- **Age**: InsightFace for age estimation
- **Classification**: HistGradientBoostingClassifier with balanced class weights

### Performance

The system achieves high accuracy on the test set with balanced class weights to handle imbalanced data.

## 📊 Data Format

### Input Images
- Supported formats: JPG, PNG
- Recommended resolution: 1920x1080 or higher
- Color space: RGB

### Annotations
All annotations are stored in CSV format:

- `detect_boxes.csv`: Object detection results
- `keypoints.csv`: Pose keypoints (17 per person)
- `depth.csv`: Depth estimates per person
- `age.csv`: Age estimates per person
- `labels.csv`: Ground truth violation labels
- `predictions.csv`: Model predictions



## 🙏 Acknowledgments

- YOLO11x models from Ultralytics
- MiDaS depth estimation from Intel ISL
- InsightFace for age estimation
- COCO dataset for object detection training


**Note**: This system is designed for research and educational purposes. Always ensure compliance with local regulations when deploying in production environments. 