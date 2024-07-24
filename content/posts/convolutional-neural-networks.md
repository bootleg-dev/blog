---
title: "Convolutional Neural Networks - CNNs"
description: "What is the convolutional neural network?"
dateString: "Date: 17 March, 2024"
date: "2024-03-17T18:34:44.165668+0500"
draft: false
tags: ["Beginner", "Neural Networks", "CNN", "Convolution"]
weight: 500
cover:
    image: ""
---

## Understanding Convolutional Neural Networks (CNNs)

### Introduction
Convolutional Neural Networks (CNNs) are a class of deep learning models designed to process visual data. They have revolutionized computer vision, enabling applications like image classification, object detection, and facial recognition. CNNs mimic the way the human brain processes visual information, making them incredibly powerful for visual tasks.

### Key Concepts in CNNs

#### Convolution Operation
- **Convolutional Layer:** The primary building block of a CNN, responsible for feature extraction.
- **Filter (Kernel):** A small matrix that slides over the input image, performing multiplications and summations to produce a feature map.
- **Feature Map (Activation Map):** The result of the convolution operation, highlighting important features such as edges, textures, and patterns.

#### Activation Function
- **ReLU (Rectified Linear Unit):** Introduces non-linearity to the model. It replaces negative values with zero, allowing the network to learn complex patterns.

$$
\text{ReLU}(x) = \max(0, x)
$$

#### Pooling Layer
- **Max-Pooling:** Reduces the spatial dimensions of the feature map while retaining the most important information by selecting the maximum value within each window.
- **Purpose:** Reduces computational complexity and helps the network become invariant to small translations of the input image.

#### Flattening
- **Flatten Layer:** Converts the 2D pooled feature maps into a 1D vector for the fully connected layers.
- **Purpose:** Prepares the data for final classification or regression tasks.

#### Fully Connected Layer
- **Dense Layer:** Connects every neuron in one layer to every neuron in the next layer.
- **Purpose:** Combines the extracted features to make final decisions.

### Detailed Explanation of CNN Components

#### Convolutional Layer
The convolutional layer applies filters to the input image to extract features like edges and textures.

- **Formula:**

$$
(I * K)(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i + m, j + n) K(m, n)
$$

  - $\( I \)$: Input image
  - $\( K \)$: Kernel (filter)
  - $\( (i, j) \)$: Coordinates in the output feature map
  - $\( M, N \)$: Dimensions of the kernel

This operation allows the network to learn spatial hierarchies of features automatically from low-level to high-level.

#### Understanding Hyperparameters
- **Kernel Size:** Dimensions of the filter (e.g., 3x3, 5x5). Affects the amount of detail the filter can capture.
- **Stride:** Step size of the filter movement. Larger strides reduce the output size but increase computational efficiency.
- **Padding:** Adds zeros around the input image to maintain the output size. "Valid" means no padding, "same" keeps the output size the same as the input.

#### Non-Linearity (ReLU)
ReLU introduces non-linearity to help the network learn complex patterns.

$$
\text{ReLU}(x) = 
\begin{cases} 
x & \text{if } x > 0 \\\
0 & \text{otherwise}
\end{cases}
$$

By setting negative values to zero, ReLU prevents the network from simply becoming a linear classifier.

#### Pooling Layers
Max-pooling reduces the spatial dimensions by selecting the maximum value in each window, effectively down-sampling the feature map.

- **Formula:**

$$
Y(i, j) = \max_{m,n} X(i \cdot s + m, j \cdot s + n)
$$

  - $\( X \)$: Input feature map
  - $\( Y \)$: Output feature map
  - $\( s \)$: Stride
  - $\( m, n \)$: Window dimensions

Pooling helps in reducing the complexity of the network and prevents overfitting.

#### Fully Connected Layer
Fully connected layers make final decisions using the features extracted by the previous layers.

- **Formula:**

$$
y = f(W \cdot x + b)
$$

  - $\( W \)$: Weight matrix
  - $\( x \)$: Input vector
  - $\( b \)$: Bias vector
  - $\( f \)$: Activation function

The fully connected layer combines the high-level features learned by the convolutional layers to output a final prediction.

#### Output Layer
The output layer is typically a softmax layer in classification tasks. The softmax function converts the raw output scores into probabilities.

- **Softmax Function:**

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

- $\( z_i \)$: The \(i\)-th element of the input vector \$(z\)$
- $\( K \)$: Number of classes


The softmax function ensures that the output probabilities sum to 1, making it easier to interpret the results as the likelihood of each class.


### Example CNN Architecture
Here’s a simple CNN architecture for image classification:

1. **Input Layer:** 224x224 RGB image
2. **Convolutional Layer:** 32 filters of size 3x3, stride 1, ReLU activation
3. **Max-Pooling Layer:** 2x2 window, stride 2
4. **Convolutional Layer:** 64 filters of size 3x3, stride 1, ReLU activation
5. **Max-Pooling Layer:** 2x2 window, stride 2
6. **Flatten Layer:** Converts the feature maps into a 1D vector
7. **Fully Connected Layer:** 128 neurons, ReLU activation
8. **Output Layer:** Softmax activation for classification into multiple classes

This example illustrates the typical workflow in a CNN, from input to final classification.

### Applications of CNNs
CNNs have a wide range of applications, including:
- **Image Classification:** Identifying objects within an image (e.g., cats vs. dogs).
- **Object Detection:** Detecting and localizing objects in an image.
- **Semantic Segmentation:** Classifying each pixel of an image into different categories.
- **Facial Recognition:** Identifying and verifying individuals based on facial features.
- **Medical Image Analysis:** Detecting anomalies in medical scans, such as tumors in MRI images.

By understanding these components and their functions, you can appreciate the power and versatility of CNNs in solving complex visual tasks.
