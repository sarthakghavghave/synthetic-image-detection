# 🧠 Synthetic Image Detector

A deep learning system for detecting AI-generated (synthetic) images using **Convolutional Neural Networks (CNNs)** built with **PyTorch framework**.  
The project begins with a baseline CNN and progressively moves toward transfer learning and advanced architectures.

---

## 📂 Project Overview
This project aims to build a **robust classifier** capable of distinguishing real (natural) images from synthetic (AI-generated) ones across multiple generation models such as **BigGAN**, **VQDM**, and **ADM**.

The main goals are:
- Understand low-level visual artifacts left by different image generators.
- Develop a CNN baseline capable of generalizing to unseen generators.
- Improve performance through **transfer learning** and **ensemble methods** in future iterations.

---

## Dataset
**GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images**

- Source: [GenImage GitHub Repository](https://github.com/GenImage-Dataset/GenImage)
- The dataset includes synthetic and natural images from multiple generation models:
  - **BigGAN**
  - **VQDM**
  - **ADM**
  - *(and more like Stable Diffusion, GLIDE, etc.)*
- Images are resized to **224×224** for standardization.

---

## Current Version — `Synthetic_Image_Detection_v0.x`
Baseline CNN implementation (**SimpleCNN_v2**) with:
- Batch Normalization and Dropout for stability.
- Mixed Precision Training for efficiency.
- Early Stopping, Checkpointing, and Learning Rate Scheduling.
- Grad-CAM support for visual explainability.

### Training Setup
- Train on: **BigGAN + VQDM**
- Validate on: **ADM** *(unseen generator for testing generalization)*

---

## Future Plans
- Transfer learning using pre-trained architectures (ResNet, EfficientNet).
- Larger training set covering more generators.
- Implement **Ensemble Learning** for improved robustness.
- Visualization dashboard for Grad-CAM and metrics tracking.

---

## Tech Stack
- **Language:** Python 3.11.0
- **Frameworks:** PyTorch, torchvision
- **Visualization:** Matplotlib, tqdm
- **Environment:** VS Code with CUDA virtual environment

---

## 📁 Folder Structure
```markdown
Synthetic-Image-Detector/
│
├── datasets/
│ └── GenImage/
│ └── BigGAN+VQDM/
│
├── models/
│ └── checkpoints/
│   ├── best_checkpoint.pth
│   └── last_checkpoint.pth
│
├── notebooks/
│ ├── Synthetic_Image_Detection_v0.1.ipynb
│ ├── Synthetic_Image_Detection_v0.2.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```


Sarthak Ghavghave
AI & Machine Learning Enthusiast