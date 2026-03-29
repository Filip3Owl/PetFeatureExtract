# LeNet-5 Pet Image Classifier: Dogs vs. Cats

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images of dogs and cats. It utilizes a modernized version of the classic **LeNet-5 architecture** adapted for RGB images. The pipeline includes an automated data preprocessing module designed to detect and remove corrupted image files before training, ensuring pipeline stability and model reliability.

Built with **TensorFlow/Keras**, this repository serves as a practical implementation of computer vision fundamentals, data engineering best practices, and model optimization techniques such as dynamic learning rate callbacks and dataset caching.

## Project Structure

```text
LENET/
│
├── .venv/                     # Python Virtual Environment
├── data/
│   └── PetImages/             # Dataset directory
│       ├── Cat/               # Class 0: Cat images
│       └── Dog/               # Class 1: Dog images
│
├── model/
│   ├── train_lenet5.py        # Main training and data cleaning script
│   └── lenet5_cats_dogs.h5    # Saved model weights (generated after training)
│
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Prerequisites
* **Python 3.8+**
* Virtual Environment (recommended)

## Dataset Setup
This project expects a dataset structured into two main classes: `Cat` and `Dog`. 
1. Download a dataset (such as the Microsoft Kaggle Cats and Dogs Dataset).
2. Extract the images and place them in the corresponding folders under `data/PetImages/Cat` and `data/PetImages/Dog`.

## Getting Started

### 1. Set Up the Virtual Environment
If you haven't already activated your virtual environment, do so using the following commands:

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Dependencies include `tensorflow`, `Pillow`, and `matplotlib`)*

## Usage

### Training the Model
To start the data cleaning process and train the LeNet-5 model, execute the main script from the root directory:

```bash
python model/train_lenet5.py
```

**What the script does under the hood:**
1. **Data Cleaning:** Scans the dataset directories and permanently deletes corrupted or unreadable image files using `Pillow`.
2. **Data Loading:** Splits the dataset into Training (80%) and Validation (20%) sets.
3. **Preprocessing:** Normalizes pixel values from `[0, 255]` to `[0, 1]` and optimizes data loading using TensorFlow's `cache` and `prefetch`.
4. **Training:** Trains the LeNet-5 model with an `EarlyStopping` callback to prevent overfitting.
5. **Saving:** Exports the trained model to `model/lenet5_cats_dogs.h5` for future inference.

## Model Architecture (Modernized LeNet-5)
The original LeNet-5 was designed for 32x32 grayscale images using `Tanh` activations and Average Pooling. This implementation modernizes the architecture to better handle complex features:
* **Input Shape:** 32x32x3 (RGB Images)
* **Activations:** `ReLU` (Rectified Linear Unit) for hidden layers to mitigate the vanishing gradient problem.
* **Pooling:** `MaxPooling2D` to extract the most prominent features.
* **Output:** A single dense node with a `Sigmoid` activation for binary classification (0 = Cat, 1 = Dog).

## License
Distributed under the MIT License.
```