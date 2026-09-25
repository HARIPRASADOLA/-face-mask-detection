# Face Mask Detection System

A deep learning project that detects whether a person is wearing a face mask using a Convolutional Neural Network (CNN) and OpenCV.

## Project Overview

The system uses image classification to classify faces into two categories:

- With Mask
- Without Mask

The trained CNN model can also be used with a webcam for real-time face mask detection.

## Tech Stack

- Python
- TensorFlow / Keras
- CNN
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- PowerShell
- VS Code

## Dataset

The project uses a public face-mask image dataset containing separate training, testing, and validation folders for:

- WithMask
- WithoutMask

The images were prepared and organized into two classes for CNN training.

## Model Architecture

```text
Input Image
    ↓
128 × 128 × 3
    ↓
Rescaling
    ↓
Data Augmentation
    ↓
Conv2D (32 filters)
    ↓
MaxPooling
    ↓
Conv2D (64 filters)
    ↓
MaxPooling
    ↓
Conv2D (128 filters)
    ↓
MaxPooling
    ↓
Flatten
    ↓
Dense (128)
    ↓
Dropout
    ↓
Sigmoid
    ↓
With Mask / Without Mask