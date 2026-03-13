# Synthetic Image Detection

Baseline convolutional neural network (CNN) for detecting AI-generated images using the GenImage dataset.

This project investigates whether a custom CNN trained on multiple generators (BigGAN, VQDM) can generalize to unseen generators (ADM).

---

## Project Overview
This project aims to build a **robust classifier** capable of distinguishing real (natural) images from synthetic (AI-generated) ones across multiple generation models such as **BigGAN**, **VQDM**, and **ADM**.

Recent generative models (GANs, diffusion models) produce highly realistic synthetic images.  
This project explores whether low-level visual artifacts can be learned by a CNN to distinguish:

- **AI-generated images**
- **Natural (real) images**

The focus is on **cross-generator generalization** rather than single-generator accuracy.

---

## Dataset

**GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images**

Source: https://github.com/GenImage-Dataset/GenImage

Training generators:
- BigGAN
- VQDM

Unseen evaluation generator:
- ADM

Images are resized to 224×224 resolution.

---

## Model (v0.x Baseline)

Architecture: `SimpleCNN_v2`

Key components:
- 3 convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)
- Global Average Pooling
- Fully connected classifier
- Dropout regularization
- BCEWithLogitsLoss

Training features:
- Mixed precision (AMP)
- ReduceLROnPlateau scheduler
- Early stopping (patience = 3)
- Checkpointing and resume support
- Grad-CAM compatibility

---

## Experimental Setup

- Train: BigGAN + VQDM
- Validate: ADM (unseen generator)
- Optimizer: Adam (lr = 1e-4, weight_decay = 1e-4)
- Batch size: 64
- Early stopping enabled

The goal is to measure generalization across generator families.

---

## Future Work

- Transfer learning (ResNet / EfficientNet)
- Larger multi-generator training set
- Ensemble-based detector
- Robustness evaluation under compression and noise

---

## Tech Stack

- Python 3.11
- PyTorch (`torch`, `torchvision`)
- Matplotlib
- tqdm

---

## Project Structure
```
Synthetic-Image-Detector/
│
├── datasets/
│   └── GenImage/
│       └── BigGAN+VQDM/
│
├── models/
│   └── checkpoints/
│       ├── best_checkpoint.pth
│       └── last_checkpoint.pth
│
├── notebooks/
│   ├── Synthetic_Image_Detection_v0.1.ipynb
│   └── Synthetic_Image_Detection_v0.2.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```
