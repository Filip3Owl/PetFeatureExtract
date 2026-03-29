# Dog and Cat Classification with LeNet-5

This repository contains a Python implementation of a Convolutional Neural Network (CNN) based on the **LeNet-5** architecture, adapted for binary classification of dogs and cats. The project demonstrates the application of classical deep learning architectures to modern image datasets.

## Project Overview

The core objective is to adapt the original LeNet-5—initially designed for grayscale 32x32 digit recognition—to process 3-channel (RGB) images of pets. Modifications include adjustments to the input layer dimensions, the use of Max Pooling for better feature extraction in natural images, and a sigmoid output layer for binary classification.

## Dataset

The model is trained using the [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset). 

### Directory Structure
The project expects the following directory hierarchy:
```text
.
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
├── model/
│   └── train_lenet5.py
├── requirements.txt
└── README.md
```

## Dependencies

The following libraries are required to execute the training script:
* **TensorFlow 2.x**: Deep learning framework and Keras API.
* **NumPy**: Numerical operations.
* **Matplotlib**: Data visualization and training history plotting.

Installation command:
```bash
pip install tensorflow numpy matplotlib
```

## Architecture Details

The implemented model follows these stages:
1.  **Input Layer**: 64x64x3 RGB images.
2.  **Convolutional Layer (C1)**: 6 filters, 5x5 kernel, ReLU activation.
3.  **Max Pooling (S2)**: 2x2 pool size.
4.  **Convolutional Layer (C3)**: 16 filters, 5x5 kernel, ReLU activation.
5.  **Max Pooling (S4)**: 2x2 pool size.
6.  **Flattening Layer**: Converts 2D feature maps to 1D vectors.
7.  **Fully Connected (F5)**: 120 neurons, ReLU activation.
8.  **Fully Connected (F6)**: 84 neurons, ReLU activation.
9.  **Output Layer**: 1 neuron, Sigmoid activation.

## Usage

### Data Cleaning
Due to the presence of corrupted or non-JPEG files in the original dataset, a cleaning routine is included in the script to remove invalid images before training begins.

### Training the Model
To start the training process, ensure your environment is active and run:
```bash
python model/train_lenet5.py
```

## Implementation Notes

* **Pre-processing**: Images are rescaled to a [0, 1] range using a `Rescaling` layer.
* **Validation**: 20% of the training data is reserved for validation to monitor for overfitting.
* **Optimization**: The model uses the Adam optimizer and Binary Crossentropy loss function.

## License
This project is for educational purposes. Please refer to the original dataset provider for licensing regarding the image data.