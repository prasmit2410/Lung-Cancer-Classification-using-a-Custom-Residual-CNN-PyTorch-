# Lung Cancer Classification using a Custom Residual CNN (PyTorch)

## Project Overview

This project presents a deep learning framework for multi-class lung cancer classification from histopathological images using a custom Residual Convolutional Neural Network (ResCNN) implemented entirely in PyTorch.

The model distinguishes between healthy lung tissue and two major lung cancer subtypes while leveraging ResNet-inspired residual learning to improve optimization stability, gradient propagation, and convergence speed. The objective is to establish a lightweight yet effective baseline for automated lung cancer diagnosis from histopathological data.

---

## Problem Statement

Lung cancer remains one of the leading causes of cancer-related mortality worldwide. Histopathological examination serves as the gold standard for diagnosis; however, manual analysis is both time-intensive and highly dependent on expert interpretation.

This project aims to develop an automated deep learning system capable of classifying histopathological lung tissue images into multiple categories, thereby supporting computer-aided diagnosis and reducing diagnostic workload.

---

## Dataset

The dataset consists of three classes:

| Class Label | Description |
|-------------|-------------|
| `lung_n` | Normal Lung Tissue |
| `lung_aca` | Lung Adenocarcinoma |
| `lung_scc` | Lung Squamous Cell Carcinoma |

**Dataset:** `<<INSERT DATASET LINK>>`

---

## Data Preparation & Training Pipeline

### Data Preparation

The following preprocessing workflow was adopted:

- Organized the dataset into training and validation splits.
- Applied image normalization using PyTorch transforms.
- Built efficient `DataLoader` pipelines for batched processing.
- Preserved class integrity throughout experimentation.
- Maintained reproducibility through consistent preprocessing procedures.

### Model Development

A custom residual architecture was designed with the following structure:

- Progressive feature extraction:
  `3 → 8 → 16 → 32 → 64 → 128` channels.
- Residual learning through skip connections.
- Projection shortcuts (`1×1` convolutions) whenever feature dimensions changed.
- Batch Normalization after every convolutional layer.
- Global Average Pooling for parameter efficiency.
- Fully-connected classifier with dropout regularization.

### Training Configuration

| Component | Configuration |
|------------|---------------|
| Framework | PyTorch |
| Optimizer | Adam |
| Initial Learning Rate | `1e-4` |
| Scheduler | ReduceLROnPlateau |
| Loss Function | CrossEntropyLoss |
| Dropout | `0.2` |
| Epochs | `20` |

The learning rate schedule evolved as follows:

```text
1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.3e-6 → 3.1e-6
```

This adaptive strategy enabled smoother convergence during later stages of training.

---

## Model Architecture

The proposed network consists of five residual stages followed by a lightweight classification head.

```text
Input Image
      ↓
Residual Block (3 → 8)
MaxPool(2)

Residual Block (8 → 16)
MaxPool(2)

Residual Block (16 → 32)
MaxPool(2)

Residual Block (32 → 64)
MaxPool(2)

Residual Block (64 → 128)
MaxPool(2)

Adaptive Average Pooling (1×1)

Flatten

Linear (128 → 256)
BatchNorm
ReLU
Dropout (0.2)

Linear (256 → 128)
BatchNorm
ReLU
Dropout (0.2)

Linear (128 → 3)
```

---

## Residual Learning

The model incorporates **ResNet-inspired residual blocks**:

```text
Output = F(x) + x
```

where:

- `F(x)` denotes the learned residual mapping.
- `x` denotes the shortcut connection.

Whenever channel dimensions change, projection shortcuts using `1×1` convolutions and Batch Normalization are employed.

### Benefits of Residual Connections

- Faster convergence during optimization.
- Improved gradient flow across deeper layers.
- Reduced vanishing-gradient effects.
- Easier training of deeper architectures.
- Better feature reuse and representation learning.
- Increased training stability.

The inclusion of skip connections played a significant role in achieving strong performance while maintaining a lightweight model footprint.

---

# Results

## Best Performance Metrics

| Metric | Value |
|----------|--------|
| Maximum Validation Accuracy | **96.46%** |
| Best Validation Loss | **0.1268** |
| Best Epoch | **12** |
| Overall Classification Accuracy | **95.00%** |
| Macro F1-Score | **0.95** |
| Weighted F1-Score | **0.95** |

---

## Training Performance

The network demonstrated rapid and stable convergence, exceeding **95% validation accuracy** while maintaining low validation loss.

The best-performing checkpoint was obtained at:

```text
Epoch               : 12
Validation Accuracy : 96.46%
Validation Loss     : 0.1268
```
# Training Curves

## Training Loss

<p align="center">
  <img src="assets/Epoch vs Train Loss.png" width="750">
</p>

<p align="center">
<i>Figure 1: Training loss across epochs.</i>
</p>

---

## Testing Loss

<p align="center">
  <img src="assets/Epoch vs Test Loss.png" width="750">
</p>

<p align="center">
<i>Figure 2: Testing loss across epochs.</i>
</p>

---

## Testing Accuracy

<p align="center">
  <img src="Epoch vs Test Accuracy.png" width="750">
</p>

<p align="center">
<i>Figure 3: Test accuracy across epochs.</i>
</p>


The combination of:

- Residual learning,
- Batch Normalization,
- Adaptive learning-rate scheduling,
- Global Average Pooling, and
- Dropout regularization,

contributed significantly to efficient optimization and strong generalization.

---

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---------|-----------|---------|-----------|----------|
| `lung_n` | 0.93 | 0.93 | 0.93 | 164 |
| `lung_aca` | 1.00 | 1.00 | 1.00 | 167 |
| `lung_scc` | 0.93 | 0.93 | 0.93 | 149 |

### Overall Metrics

| Metric | Score |
|---------|--------|
| Accuracy | **0.95** |
| Macro Average | **0.95** |
| Weighted Average | **0.95** |

---

## Confusion Matrix Analysis

> **Confusion Matrix:** `<<INSERT CONFUSION MATRIX IMAGE HERE>>`

| True Class | Correct Predictions | Total Samples | Class Accuracy |
|------------|--------------------|---------------|----------------|
| `lung_n` | 153 | 164 | **93.29%** |
| `lung_aca` | 167 | 167 | **100.00%** |
| `lung_scc` | 138 | 149 | **92.62%** |

### Key Observations

#### Lung Adenocarcinoma (`lung_aca`)

The model achieved **perfect classification performance** for adenocarcinoma samples:

- **167 / 167 correctly classified**
- No false positives
- No false negatives
- Precision = **1.00**
- Recall = **1.00**
- F1-score = **1.00**

This demonstrates exceptionally strong feature extraction capabilities and complete separation from the remaining classes.

#### Lung Squamous Cell Carcinoma (`lung_scc`)

The model correctly classified:

```text
138 / 149 samples
```

Misclassifications:

- 11 samples predicted as normal tissue.
- No confusion with adenocarcinoma samples.

This indicates strong subtype-specific feature learning and robust cancer-class separation.

#### Normal Lung Tissue (`lung_n`)

The model correctly classified:

```text
153 / 164 samples
```

Misclassifications:

- 11 samples predicted as squamous cell carcinoma.
- No samples predicted as adenocarcinoma.

The absence of confusion with adenocarcinoma highlights the effectiveness of the learned feature representations.

### Overall Interpretation

The confusion matrix reveals several important characteristics:

- **Zero confusion involving adenocarcinoma samples.**
- Misclassifications occur exclusively between **normal tissue** and **squamous cell carcinoma**.
- Both cancer classes remain distinctly separated.

Achieving perfect adenocarcinoma detection alongside more than **92% class accuracy** on the remaining categories demonstrates the effectiveness of residual learning for histopathological image classification.

---

## Model Availability

**Dataset:** `<<[INSERT DATASET LINK](https://drive.google.com/drive/folders/1vcw8JRQr8-62IVvix4rK4rDXrFnlap8e)>>`

**Hugging Face Model Repository:** `<<INSERT MODEL LINK>>`

---

## Future Work

Several directions can further enhance this work:

### Data Augmentation

Future experiments will explore:

- Random rotations
- Horizontal and vertical flips
- Color jittering
- Elastic transformations
- Stain normalization methods

Increasing sample diversity may further improve robustness and reduce confusion between normal tissue and squamous cell carcinoma.

### Advanced Training Strategies

Potential improvements include:

- MixUp and CutMix augmentation.
- Cross-validation experiments.
- Class-balanced sampling.
- Label smoothing techniques.
- Attention modules such as CBAM and SE blocks.
- Transfer learning from large-scale medical imaging models.

### Model Interpretability

Future work may also incorporate:

- Grad-CAM visualizations.
- Feature activation analysis.
- Explainable AI (XAI) methods for clinical interpretability.

### Deployment

Planned deployment directions include:

- Publishing pretrained weights on Hugging Face.
- Building an interactive inference interface.
- Packaging lightweight deployment pipelines for research and educational use.

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Conclusion

This project demonstrates that a lightweight, custom-designed Residual CNN can effectively classify lung histopathological images with strong generalization and excellent class-wise performance.

By incorporating residual learning, batch normalization, adaptive optimization, global average pooling, and regularization techniques, the model achieved:

- **96.46% peak validation accuracy**
- **0.1268 best validation loss**
- **100% adenocarcinoma detection (167/167 samples)**
- **95% overall classification accuracy**
- **0.95 macro and weighted F1-scores**

These results highlight the potential of residual architectures for medical image analysis and establish a strong foundation for future work involving larger datasets, advanced augmentation strategies, explainable AI techniques, and deployment-oriented applications.
