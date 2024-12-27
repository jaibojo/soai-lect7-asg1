# MNIST Digit Classification with CNN

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a focus on curriculum learning and dynamic learning rate scheduling.

## Project Structure

- `model5.py`: CNN architecture implementation
- `train.py`: Training loop with curriculum learning
- `training_utils.py`: Utility functions for training and evaluation

## Model Architecture (Model5)

A CNN architecture optimized for MNIST with:
- Total Parameters: 7,778
- Receptive Field: 26x26 (final)

### Layer Structure:
1. Input Block (RF: 3x3)
   - Conv 1x6x3x3 + ReLU + BN + Dropout
   - Output: 26x26x6

2. Convolution Block 1 (RF: 5x5)
   - Conv 6x32x3x3 + ReLU + BN + Dropout
   - Output: 24x24x32

3. Transition Block (RF: 6x6)
   - Conv 32x6x1x1
   - MaxPool 2x2
   - Output: 12x12x6

4. Convolution Block 2 (RF: 10x10)
   - Conv 6x16x3x3 + ReLU + BN + Dropout
   - Output: 10x10x16

5. Convolution Block 3 (RF: 14x14)
   - Conv 16x16x3x3 + ReLU + BN + Dropout
   - Output: 8x8x16

6. Convolution Block 4 (RF: 18x18)
   - Conv 16x16x3x3 + ReLU + BN + Dropout
   - Output: 8x8x16

7. Output Block (RF: 26x26)
   - Global Average Pooling
   - Conv 16x10x1x1
   - Log Softmax
   - Output: 1x1x10

## Training Approach

### Curriculum Learning
- Sorts images by difficulty after each epoch
- Presents hardest images first in subsequent epochs
- Difficulty measured by loss value for each image

### Learning Rate Schedule
Four distinct phases:
1. Epoch 1: LR = 0.01
2. Epochs 2-5: LR = 0.008
3. Epochs 6-10: LR = 0.004
4. Epochs 11-15: LR = 0.001

Each phase (except first) uses 5 stages with momentum variations.

### Data Handling
- Training Set: 50,000 images
- Validation Set: 10,000 images
- Test Set: 10,000 images
- Batch Size: 128

### Augmentation
- Random Rotation: ±7°
- Random Translation: up to 10%
- Normalization: Mean=0.1307, Std=0.3081

## Usage 