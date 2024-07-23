---
title: "Image Segmentation Techniques: FCN and U-Net"
description: "FCN and U-Net."
dateString: "Date: 01 April, 2024"
date: "2024-04-01T18:34:44.165668+0500"
draft: false
tags: ["Beginner", "Image Segmentation", "Deep Learning", "FCN", "Unet"]
weight: 495
cover:
    image: ""
---


## Introduction

Image segmentation is a crucial technique in computer vision, enabling the division of an image into multiple meaningful and homogeneous regions or objects based on their inherent characteristics, such as color, texture, shape, or brightness. This process is fundamental in applications like object recognition, tracking, detection, medical imaging, and robotics. In this article, we delve into two powerful deep learning models for image segmentation: Fully Convolutional Networks (FCN) and U-Net. We will explore their architectures, key concepts, and practical applications, providing insights into their advantages and best practices for implementation.

## Fully Convolutional Networks (FCN)

### Overview

Fully Convolutional Networks (FCNs) are a class of neural networks designed specifically for semantic segmentation. Unlike traditional Convolutional Neural Networks (CNNs) that produce a single label for the entire image, FCNs output a segmentation map where each pixel is classified into a particular category.

### Architecture

The architecture of an FCN is based on an encoder-decoder structure:

1. **Encoder (Downsampling Path)**: This part of the network extracts complex features from the input image through a series of convolutional and pooling layers. The spatial resolution is reduced while increasing the depth of the feature maps, allowing the network to capture high-level semantic information.

2. **Decoder (Upsampling Path)**: The decoder part of the network upscales the reduced-resolution feature maps back to the original image size. This is achieved through transposed convolution layers (also known as deconvolution layers), which learn the appropriate strides and padding to reconstruct the high-resolution segmentation map.

### Key Concepts

- **Pooling (Downsampling)**: Pooling layers reduce the spatial resolution of the feature maps, which helps in capturing invariant features and reducing computational complexity. Common types include max pooling and average pooling.
  
- **Transposed Convolution (Upsampling)**: Transposed convolution layers are used to increase the spatial resolution of the feature maps, essentially reversing the effect of pooling layers to reconstruct the detailed segmentation map.

- **Skip Connections**: To address the loss of spatial information due to pooling, skip connections are introduced. These connections transfer features from the encoder directly to the corresponding decoder layers, enabling the network to recover fine-grained details and produce more accurate segmentation boundaries.

### Practical Implementation

Here is a simplified implementation of an FCN using PyTorch:

```python
import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleFCN, self).__init__()
        
        # Encoder: Downsampling part
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv layer
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv layer
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        )
        
        # Decoder: Upsampling part
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsampling
            nn.ReLU(),  # Activation
            nn.ConvTranspose2d(64, n_classes, kernel_size=2, stride=2)  # Upsampling to original size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = SimpleFCN(n_classes=21)  # Example with 21 classes

```

## U-Net

### Overview

U-Net is a specialized neural network architecture designed for biomedical image segmentation, introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. Its distinctive U-shaped design, which features a symmetric encoder-decoder structure, has made it a popular choice in various medical image analysis tasks due to its impressive performance and efficiency.

### Architecture

The U-Net architecture consists of two main parts:

- **Encoder (Contraction Path)**: Similar to FCN, the encoder part of U-Net captures high-level features through a series of convolutional and pooling layers. Each block typically consists of two 3x3 convolution layers followed by a ReLU activation function and a 2x2 max pooling layer.

- **Decoder (Expansion Path)**: The decoder part upscales the feature maps to the original image size using transposed convolutions. At each upsampling step, the decoder concatenates the feature maps from the corresponding encoder layer via skip connections, providing rich contextual information for precise segmentation.

### Key Concepts

- **Skip Connections**: By concatenating the feature maps from the encoder to the decoder at each corresponding level, U-Net can leverage both high-level and low-level features, resulting in better localization and segmentation accuracy.

- **Data Augmentation**: Given the often limited availability of annotated data in medical imaging, U-Net heavily relies on data augmentation techniques to enhance the diversity of the training dataset. This includes operations like rotations, flips, and elastic deformations.

### Practical Implementation

Here is a simplified implementation of a U-Net using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, n_classes):
        super(SimpleUNet, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder (Upsampling)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(512, 256)  # 256 + 256 from skip connection

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 128 + 128 from skip connection

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)   # 64 + 64 from skip connection

        # Final layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Decoder with skip connections
        dec1 = self.upconv1(enc4)
        dec1 = torch.cat((dec1, enc3), dim=1)  # Concatenate skip connection
        dec1 = self.dec1(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate skip connection
        dec2 = self.dec2(dec2)

        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)  # Concatenate skip connection
        dec3 = self.dec3(dec3)

        x = self.final_conv(dec3)
        return x

model = SimpleUNet(n_classes=21)  # Example with 21 classes

```

## Summary

Both FCN and U-Net are powerful architectures for image segmentation tasks. FCN is a more general-purpose segmentation network, while U-Net is specifically designed for biomedical image segmentation with its U-shaped encoder-decoder structure and extensive use of skip connections. 
