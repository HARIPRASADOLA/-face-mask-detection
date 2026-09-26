
# Face Mask Detection System

A deep learning-based computer vision project that detects whether a person is wearing a face mask. The system combines a **Convolutional Neural Network (CNN)** for image classification with **OpenCV** for face detection and real-time webcam inference.

## Project Overview

The Face Mask Detection System classifies detected faces into two categories:

- **With Mask**
- **Without Mask**

The project covers the complete machine learning workflow, including dataset preparation, CNN model training, evaluation, confusion matrix analysis, and real-time webcam detection.

## Objectives

- Build a CNN-based image classification model.
- Detect faces using OpenCV.
- Classify detected faces as `With Mask` or `Without Mask`.
- Evaluate model performance using standard classification metrics.
- Implement real-time mask detection through a webcam.
- Organize the project for reproducible development and GitHub sharing.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV | Face detection and webcam processing |
| NumPy | Numerical operations |
| Scikit-learn | Model evaluation |
| Matplotlib | Visualization |
| PowerShell | Project setup and execution |
| Visual Studio Code | Development environment |

## Dataset

The project uses a public face-mask image dataset containing two classes:

```text
WithMask
WithoutMask
```

The working dataset contains:

- **With Mask:** 5,883 images
- **Without Mask:** 5,909 images
- **Total:** 11,792 images

The dataset is organized for image classification and contains separate training, validation, and testing data in the original public dataset structure.

## Dataset Structure

```text
Face Mask Dataset/
│
├── Train/
│   ├── WithMask/
│   └── WithoutMask/
│
├── Validation/
│   ├── WithMask/
│   └── WithoutMask/
│
└── Test/
    ├── WithMask/
    └── WithoutMask/
```

## Project Structure

```text
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   ├── face_mask_cnn.keras
│   └── face_mask_cnn_final.keras
│
├── results/
│   ├── evaluation_report.txt
│   ├── test_evaluation_report.txt
│   └── confusion_matrix.png
│
├── src/
│   ├── webcam_test.py
│   ├── face_detection.py
│   ├── check_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── evaluate_test.py
│   ├── confusion_matrix.py
│   └── webcam_mask_detection.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## CNN Architecture

The classification model uses a custom CNN architecture:

```text
Input Image
     ↓
128 × 128 × 3
     ↓
Rescaling
     ↓
Data Augmentation
     ↓
Conv2D - 32 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 64 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 128 Filters
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense - 128 Neurons
     ↓
Dropout
     ↓
Dense - 1 Neuron
     ↓
Sigmoid
     ↓
With Mask / Without Mask
```

## Model Training

The CNN was trained using:

- Image size: `128 × 128`
- Batch size: `32`
- Optimizer: `Adam`
- Loss function: `Binary Crossentropy`
- Activation: `ReLU` for convolutional/dense layers
- Output activation: `Sigmoid`
- Maximum epochs: `15`
- Early stopping
- Best-model checkpointing

### Training Validation Result

The training run achieved:

```text
Validation Loss: 0.0380
Validation Accuracy: 98.73%
```

The trained models were saved as:

```text
models/face_mask_cnn.keras
models/face_mask_cnn_final.keras
```

> Note: The validation result above comes from the training-time validation split. Final test-set performance should be reported separately using the original test dataset.

## Model Evaluation

The project includes evaluation using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation scripts:

```powershell
python .\src\evaluate_model.py
```

For evaluation against the original test dataset:

```powershell
python .\src\evaluate_test.py
```

The evaluation report is saved to:

```text
results/test_evaluation_report.txt
```

## Confusion Matrix

The project generates a confusion matrix using:

```powershell
python .\src\confusion_matrix.py
```

Output:

```text
results/confusion_matrix.png
```

The confusion matrix helps identify correct and incorrect predictions for both mask classes.

## Real-Time Webcam Detection

The trained CNN is integrated with OpenCV for real-time detection.

Run:

```powershell
python .\src\webcam_mask_detection.py
```

The application:

1. Opens the computer webcam.
2. Captures live video frames.
3. Converts frames for face detection.
4. Detects faces using OpenCV Haar Cascade.
5. Crops detected faces.
6. Resizes the face to `128 × 128`.
7. Sends the image to the trained CNN.
8. Predicts `WITH MASK` or `WITHOUT MASK`.
9. Displays the prediction and confidence on the webcam frame.

Press:

```text
Q
```

to close the webcam application.

## Installation

Create and activate a Python virtual environment:

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

## Verify TensorFlow

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Current development environment:

```text
TensorFlow: 2.22.0-rc0
OpenCV: 5.0.0
```

## Verify OpenCV

```powershell
python -c "import cv2; print(cv2.__version__)"
```

## Test Webcam

Before running mask detection:

```powershell
python .\src\webcam_test.py
```

## Test Face Detection

```powershell
python .\src\face_detection.py
```

## Check Dataset

```powershell
python .\src\check_dataset.py
```

## Train the Model

```powershell
python .\src\train_model.py
```

The trained model will be saved inside:

```text
models/
```

## Evaluate the Model

```powershell
python .\src\evaluate_test.py
```

## Generate Confusion Matrix

```powershell
python .\src\confusion_matrix.py
```

## Run Real-Time Detection

```powershell
python .\src\webcam_mask_detection.py
```

## Key Features

- CNN-based face mask classification
- Two-class image classification
- OpenCV face detection
- Real-time webcam inference
- Model checkpointing
- Early stopping
- Accuracy evaluation
- Precision, recall and F1-score
- Confusion matrix generation
- Saved Keras model
- PowerShell-based development workflow

## Learning Outcomes

Through this project, the following concepts were practiced:

- Image preprocessing
- Dataset organization
- CNN architecture
- Image augmentation
- Binary classification
- Model training
- Model validation
- Classification metrics
- Confusion matrix analysis
- OpenCV computer vision
- Real-time webcam inference
- Python virtual environments
- Project organization
- GitHub-ready documentation

## Future Improvements

Possible improvements include:

- Transfer learning using MobileNetV2 or EfficientNet.
- Improved performance under low-light conditions.
- Better face detection using modern object-detection models.
- Multi-face tracking.
- Confidence threshold tuning.
- Web application deployment.
- Edge-device optimization.
- Model conversion to TensorFlow Lite.

## Limitations

The model's performance can vary depending on:

- Lighting conditions
- Camera quality
- Face angle
- Occlusion
- Distance from the camera
- Mask type and appearance
- Dataset characteristics

The system is intended as a learning and demonstration project rather than a production-grade safety or compliance system.

## Author

**Hari Prasad Ola**

**Gmail** hrprsdola@gmail.com**

### Project Focus

**Deep Learning | Computer Vision | CNN | TensorFlow | OpenCV | Python**

