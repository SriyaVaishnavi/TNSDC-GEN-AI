# Character Recognition using Convolutional Neural Networks (CNN)

## Overview

This project utilizes Convolutional Neural Networks (CNNs) for character recognition. CNNs are powerful deep learning models particularly well-suited for image classification tasks due to their ability to automatically learn hierarchical features from data.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (optional, for image preprocessing)

## Dataset

The dataset used for training and testing the CNN model consists of images of characters. You can use publicly available datasets like MNIST, EMNIST, or create your own dataset by collecting and labeling images of characters.

## Installation

Clone the repository:

   ```
   git clone https://github.com/SriyaVaishnavi/TNSDC-GEN-AI.git
   ```

## Usage

1. Ensure all dependencies are installed.
2. Run the provided Python script.
3. View the training progress and the recognition results.

## Code Description

- The code first loads the MNIST dataset using TensorFlow's `mnist.load_data()` function.
- It preprocesses the data by normalizing the pixel values to the range [0, 1] and converting labels to one-hot encoding.
- A CNN model is built using TensorFlow's Keras API, consisting of convolutional layers with ReLU activation, max-pooling layers, and fully connected layers.
- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
- Training is performed using the training data for 5 epochs with a batch size of 64.
- Predictions are made on the test set, and recognized labels are extracted using `argmax`.
- The script displays a grid of images from the test set along with their recognized and actual labels using Matplotlib.


## Model Architecture

The CNN model architecture typically consists of convolutional layers followed by pooling layers for feature extraction, and then fully connected layers for classification. You can customize the architecture based on the complexity of your dataset and computational resources.

## Hyperparameter Tuning

Experiment with different hyperparameters such as learning rate, batch size, number of layers, and layer sizes to optimize the model's performance.

## Acknowledgments

- This project was inspired by various tutorials and resources available online.
- Special thanks to the creators and maintainers of TensorFlow and Keras for their invaluable contributions to the deep learning community.


