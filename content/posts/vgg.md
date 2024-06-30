---
title: "Visual Geometry Group - VGG Architecture"
description: "VGG-16 and VGG-19."
dateString: "Date: 19 March, 2024"
date: "2024-03-19T18:34:44.165668+0500"
draft: false
tags: ["Beginner", "Neural Networks", "Deep Learning", "CNN", "VGG"]
weight: 499
cover:
    image: ""
---

## Introduction

The VGG (Visual Geometry Group) network is a renowned deep learning model known for its simplicity and effectiveness in image recognition tasks. Developed by K. Simonyan and A. Zisserman from Oxford University, the VGG network significantly advanced the field of computer vision and performed remarkably in the ILSVRC-2014 competition.

## VGG Architecture

VGG networks are characterized by their deep architecture, which involves stacking multiple convolutional layers. The two most commonly used versions are VGG-16 and VGG-19, featuring 16 and 19 layers, respectively.

### Key Components of VGG

![Sample Image and Landmarks from the Dataset](/blog/posts/vgg/img1.png)

- **Fixed Size Input:** The network accepts a fixed size of (224 x 224) RGB images.
- **Preprocessing:** The only preprocessing step involves subtracting the mean RGB value from each pixel, computed over the entire training set (ImageNet).
- **Kernel Size:** VGG uses a small receptive field of 3x3 kernels with a stride of 1.
- **Max-Pooling:** Performed over a 2x2 pixel window with a stride of 2.
- **Fully Connected Layers:** VGG has three fully connected layers. The first two layers have 4096 neurons each, and the final layer has 1000 neurons, corresponding to the 1000 classes in the ImageNet dataset.
- **Activation:** Uses ReLU (Rectified Linear Unit) to introduce non-linearity.


## VGG-16 vs. VGG-19

The primary difference between VGG-16 and VGG-19 lies in the number of layers:

- **VGG-16:** Comprises 13 convolutional layers and 3 fully connected layers, making it slightly less complex and faster to train compared to VGG-19.
- **VGG-19:** Contains 16 convolutional layers and 3 fully connected layers, offering slightly better accuracy at the cost of increased computational resources and training time.

Both models have demonstrated high accuracy in various benchmarks, but the choice between them depends on the specific application and the available computational resources.

## Achievements of VGG

The VGG-16 model achieved a test accuracy of 92.7% on the ImageNet dataset, which includes over 14 million images across 1000 categories. This performance made it one of the top models in the ILSVRC-2014 competition.

## Summary

VGG's legacy as a pioneering deep CNN architecture continues to shape the landscape of computer vision. Its depth, simplicity, and effectiveness have made it a valuable tool for researchers and practitioners alike. As the field progresses, VGG's contributions serve as a reminder of the power of deep learning to unlock the secrets hidden within images.

