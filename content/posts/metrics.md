---
title: "Metrics in Machine Learning and Computer Vision"
description: "Understanding essential metrics for evaluating machine learning models, particularly in computer vision."
dateString: "Date: 22 April, 2024"
date: "2024-04-22T18:34:44.165668+0500"
draft: false
tags: ["Beginner", "Machine Learning", "Computer Vision", "Metrics", "Evaluation"]
weight: 494
cover:
    image: ""
---

### Introduction

Evaluating model performance is as critical as designing and training the models themselves. Metrics provide quantifiable measures to assess the effectiveness and accuracy of models, guiding decisions for improvement and deployment. 
### Classification Metrics

#### Accuracy

Accuracy is the most straightforward metric, representing the **ratio of correctly predicted instances to the total instances**.

$$
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Instances}}
$$

**Example Calculation**: 
- True Positives (TP) = 50
- True Negatives (TN) = 40
- False Positives (FP) = 10
- False Negatives (FN) = 0
- Total Instances = 100

$$
\text{Accuracy} = \frac{50 + 40}{100} = \frac{90}{100} = 0.90 \, \text{or} \, 90\text{%}
$$

#### Precision, Recall, and F1 Score

Precision and recall provide deeper insights, especially in **imbalanced datasets**.

- **Precision**: The ratio of true positive predictions to the total predicted positives.

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

**Example Calculation**: 
- TP = 30
- FP = 10

$$
\text{Precision} = \frac{30}{30 + 10} = \frac{30}{40} = 0.75 \, \text{or} \, 75\text{%}
$$

- **Recall**: The ratio of true positive predictions to all actual positives.

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

**Example Calculation**: 
- TP = 30
- FN = 20

$$
\text{Recall} = \frac{30}{30 + 20} = \frac{30}{50} = 0.60 \, \text{or} \, 60\text{%}
$$

- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics.

$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Example Calculation**: 
- Precision = 0.75
- Recall = 0.60

$$
\text{F1 Score} = 2 \cdot \frac{0.75 \cdot 0.60}{0.75 + 0.60} = 2 \cdot \frac{0.45}{1.35} = \frac{0.90}{1.35} = 0.67 \, \text{or} \, 67\text{%}
$$

#### Specificity and Sensitivity

- **Specificity**: The ratio of true negative predictions to all actual negatives. It is used to measure the ability of the model to correctly identify negative instances.

$$
\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}
$$

**Example Calculation**: 
- TN = 70
- FP = 10

$$
\text{Specificity} = \frac{70}{70 + 10} = \frac{70}{80} = 0.875 \, \text{or} \, 87.5\text{%}
$$

- **Sensitivity**: Also known as recall, it measures the ability of the model to correctly identify positive instances.

$$
\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

#### Confusion Matrix

A confusion matrix is a table that provides a detailed breakdown of the performance of a classification model. It shows the number of true positive, true negative, false positive, and false negative predictions.

**Example Calculation**:

|               | Predicted Positive                        | Predicted Negative                         |
|---------------|-------------------------------------------|--------------------------------------------|
| Actual Positive | True Positive (TP) = 50                   | **Type 2 Error**: False Negative (FN) = 10 |
| Actual Negative | **Type 1 Error**: False Positive (FP) = 5 | True Negative (TN) = 35                    |

#### ROC-AUC

The Receiver Operating Characteristic **(ROC) curve** is a graphical representation of a model's ability to discriminate between 
**positive and negative classes across different threshold values**. The Area Under the Curve (AUC) quantifies this ability into a single scalar value. 

- **ROC Curve**: Plots the true positive rate (recall) against the false positive rate (1 - specificity) at various threshold settings.
- **AUC**: The area under the ROC curve, where an AUC of 1 represents a perfect model, and an AUC of 0.5 represents a model with no discrimination capability.

**Example Calculation**: Suppose we have the following TPR and FPR values at different thresholds:

| Threshold | TPR (Recall) | FPR (1 - Specificity) |
|-----------|--------------|-----------------------|
| 0.1       | 0.95         | 0.50                  |
| 0.2       | 0.90         | 0.30                  |
| 0.3       | 0.85         | 0.20                  |
| 0.4       | 0.80         | 0.15                  |
| 0.5       | 0.75         | 0.10                  |

Plotting these points on the ROC curve and calculating the area under this curve gives us the AUC.

#### Log Loss (Cross-Entropy Loss)

Log Loss evaluates the performance of a classification model where the prediction output is a **probability value between 0 and 1**.

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

**Example Calculation**: 
- Suppose we have 3 instances with the following actual and predicted probabilities:
  - Instance 1: $\( y_1 = 1 \), \( \hat{y}_1 = 0.9 \)$
  - Instance 2: $\( y_2 = 0 \), \( \hat{y}_2 = 0.2 \)$
  - Instance 3: $\( y_3 = 1 \), \( \hat{y}_3 = 0.7 \)$

$$
\text{Log Loss} = -\frac{1}{3} [(1 \cdot \log(0.9) + (1 - 1) \cdot \log(1 - 0.9)) + (0 \cdot \log(0.2) + (1 - 0) \cdot \log(1 - 0.2)) + (1 \cdot \log(0.7) + (1 - 1) \cdot \log(1 - 0.7))]
$$

$$
\text{Log Loss} = -\frac{1}{3} [(-0.105) + (-0.223) + (-0.357)] = -\frac{1}{3} [-0.685] = 0.228
$$

#### Matthews Correlation Coefficient (MCC)

MCC is a balanced measure that can be used even if the classes are of very different sizes. It considers true and false positives and negatives.

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

**Example Calculation**: 
- TP = 50, TN = 40, FP = 10, FN = 10

$$
\text{MCC} = \frac{(50 \cdot 40) - (10 \cdot 10)}{\sqrt{(50 + 10)(50 + 10)(40 + 10)(40 + 10)}} = \frac{2000 - 100}{\sqrt{60 \cdot 60 \cdot 50 \cdot 50}} = \frac{1900}{150000} = 0.63
$$

### Regression Metrics

#### Mean Absolute Error (MAE)

MAE measures the average magnitude of errors in predictions without considering their direction.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Example Calculation**: 
- Errors = [1, 2, 3]

$$
\text{MAE} = \frac{1 + 2 + 3}{3} = \frac{6}{3} = 2
$$

#### Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

MSE squares the errors before averaging, penalizing larger errors more significantly.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Example Calculation**: 
- Errors = [1, 2, 3]

$$
\text{MSE} = \frac{1^2 + 2^2 + 3^2}{3} = \frac{1 + 4 + 9}{3} = \frac{14}{3} \approx 4.67
$$

RMSE is the square root of MSE, bringing the units back to the original scale.

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

**Example Calculation**: 
- MSE = 4.67

$$
\text{RMSE} = \sqrt{4.67} \approx 2.16
$$

#### R-Squared (R²)

R-Squared represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$


**Example Calculation**: 
- Total Sum of Squares (TSS) = 100
- Residual Sum of Squares (RSS) = 20

$$
R^2 = 1 - \frac{20}{100} = 1 - 0.20 = 0.80 \, \text{or} \, 80\text{\%}
$$


#### Mean Absolute Percentage Error (MAPE)

MAPE measures the average absolute percentage error between predicted and actual values.

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

**Example Calculation**: 
- Actual values = [100, 200, 300]
- Predicted values = [110, 190, 310]

$$
\text{MAPE} = \frac{1}{3} \left( \left| \frac{100 - 110}{100} \right| + \left| \frac{200 - 190}{200} \right| + \left| \frac{300 - 310}{300} \right| \right) = \frac{1}{3} \left(0.10 + 0.05 + 0.033\right) = \frac{1}{3} \left(0.183\right) \approx 0.061 \, \text{or} \, 6.1\text{%}
$$

#### Explained Variance Score

Explained variance measures how much of the variance in the target variable is explained by the model.

$$
\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
$$

**Example Calculation**: 
- Variance of residuals = 20
- Variance of target = 100

$$
\text{Explained Variance} = 1 - \frac{20}{100} = 1 - 0.20 = 0.80 \, \text{or} \, 80\text{%}
$$

### Computer Vision Metrics

#### Intersection over Union (IoU)

IoU is crucial for segmentation and object detection tasks, measuring the overlap between the predicted and ground truth bounding boxes or segments.

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

**Example Calculation**: 
- Area of Overlap = 30
- Area of Union = 50

$$
\text{IoU} = \frac{30}{50} = 0.60 \, \text{or} \, 60\text{%}
$$

#### Mean Average Precision (mAP)

mAP is commonly used in object detection, summarizing the precision-recall curve across multiple classes.

1. Calculate the precision-recall curve for each class.
2. Compute the Average Precision (AP) for each class.
3. Take the mean of AP values across all classes.

**Example Calculation**: 
- AP values for three classes = [0.8, 0.7, 0.9]

$$
\text{mAP} = \frac{0.8 + 0.7 + 0.9}{3} = \frac{2.4}{3} = 0.80 \, \text{or} \, 80\text{%}
$$

#### Dice Coefficient

The Dice Coefficient, similar to IoU, is another metric for segmentation tasks, focusing on the overlap between predicted and ground truth segments.

$$
\text{Dice Coefficient} = \frac{2 \times \text{Area of Overlap}}{\text{Total Area of Predicted} + \text{Total Area of Ground Truth}}
$$

**Example Calculation**: 
- Area of Overlap = 30
- Total Area of Predicted = 40
- Total Area of Ground Truth = 50

$$
\text{Dice Coefficient} = \frac{2 \times 30}{40 + 50} = \frac{60}{90} = 0.67 \, \text{or} \, 67\text{%}
$$

#### Pixel Accuracy

Pixel Accuracy measures the proportion of correctly classified pixels in the entire image.

$$
\text{Pixel Accuracy} = \frac{\text{Number of Correct Pixels}}{\text{Total Number of Pixels}}
$$

**Example Calculation**: 
- Number of Correct Pixels = 900
- Total Number of Pixels = 1000

$$
\text{Pixel Accuracy} = \frac{900}{1000} = 0.90 \, \text{or} \, 90\text{%}
$$

#### Mean IoU (mIoU)

mIoU is the mean of the Intersection over Union (IoU) for all classes. It is commonly used for semantic segmentation tasks.

$$
\text{mIoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

**Example Calculation**: 
- IoU values for three classes = [0.6, 0.7, 0.8]

$$
\text{mIoU} = \frac{0.6 + 0.7 + 0.8}{3} = \frac{2.1}{3} = 0.70 \, \text{or} \, 70\text{%}
$$

#### Structural Similarity Index (SSIM)

SSIM measures the similarity between two images, considering luminance, contrast, and structure.

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

**Example Calculation**: 
$$Assume \( \mu_x = 100 \), \( \mu_y = 105 \), \( \sigma_x = 15 \), \( \sigma_y = 20 \), \( \sigma_{xy} = 18 \), \( C_1 = 6.5 \), \( C_2 = 58 \)$$

$$
\text{SSIM} = \frac{(2 \cdot 100 \cdot 105 + 6.5)(2 \cdot 18 + 58)}{(100^2 + 105^2 + 6.5)(15^2 + 20^2 + 58)}
$$

$$
\text{SSIM} = \frac{(21000 + 6.5)(36 + 58)}{(10000 + 11025 + 6.5)(225 + 400 + 58)}
$$

$$
\text{SSIM} = \frac{21006.5 \cdot 94}{21031.5 \cdot 683} \approx 0.42
$$
