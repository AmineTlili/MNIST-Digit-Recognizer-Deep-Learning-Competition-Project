# ✍️ MNIST Digit Recognizer — Deep Learning Competition Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Kaggle](https://img.shields.io/badge/Kaggle-MNIST-blue?logo=kaggle)
![Accuracy](https://img.shields.io/badge/Accuracy-99.91%25-brightgreen)
![DL](https://img.shields.io/badge/Deep%20Learning-CNN%2C%20MLP%2C%20KNN-purple)

## 🧾 Overview

This project was developed as part of the **MNIST Digit Recognizer** competition on Kaggle. The goal was to build and evaluate deep learning models that can accurately classify handwritten digits from the famous MNIST dataset.

We implemented and compared three models:  
- 📈 **Convolutional Neural Network (CNN)**  
- 📊 **Multi-Layer Perceptron (MLP)**  
- 📍 **K-Nearest Neighbors (KNN)**  

The CNN model achieved an outstanding accuracy of **99.91%**, thanks to a carefully designed architecture and optimized training strategies.

---

## 🧠 Problem Statement

The task is to classify images of handwritten digits (0–9) into their correct classes. The input consists of **28x28 grayscale images**, and the output is a single digit label.

---

## 🗃️ Dataset

- 📦 **Training Data**: 60,000 labeled images  
- 🧪 **Test Data**: 10,000 unlabeled images  
- 📍 Format: Each row is an image flattened into 784 pixels.  
- 🧮 Target: Digit (0–9)  
- Source: [Kaggle MNIST Competition](https://www.kaggle.com/c/digit-recognizer)

---

## ⚙️ Models & Architecture

### 🧠 CNN Architecture
- 2 Convolutional blocks (Conv2D + BatchNorm + MaxPooling + Dropout)
- Dense layers: [512 ➝ 1024 ➝ 10]
- Regularization: Batch Normalization, Dropout, EarlyStopping
- Optimizer: Adam + Learning Rate Scheduler

### 🧠 MLP Architecture
- Input: 512 ➝ Hidden: 256 ➝ 128 ➝ Output: 10 (Softmax)
- Activation: ReLU, Optimizer: Adam
- Regularization: Dropout, EarlyStopping

### 🧠 KNN Classifier
- Classical algorithm used for benchmarking.
- GridSearchCV used to optimize `K` and distance metrics.

---

## 📈 Training & Optimization

- ✅ Optimizer: Adam with learning rate decay  
- 🕹️ Callback: EarlyStopping  
- 📉 Loss Function: Categorical Cross-Entropy  
- 🔄 Validation: 5-Fold Cross Validation  

---

## 📊 Performance Metrics

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| CNN   | **99.91%** | 0.998    |
| MLP   | ~98.5%    | ~0.97    |
| KNN   | ~96.4%    | ~0.95    |

---

## 🔍 Visualizations

- 📉 Training & Validation Loss & Accuracy Curves  
- 🔁 Confusion Matrices  
- 🎯 Classification Reports  

> These helped us identify class-specific misclassifications and ensure model stability.

---

## 🧪 Lessons Learned

- CNNs excel in spatial pattern recognition for image tasks.
- Regularization techniques like Dropout, BatchNorm, and EarlyStopping are essential for generalization.
- Simple models (MLP, KNN) are good baselines but underperform CNNs.
- Hyperparameter tuning is crucial for squeezing out performance.

---

## 📌 Final Notes

- This project was part of the academic year **2024–2025** deep learning coursework.
- Developed by:
  - 🧑‍💻 Mohamed Amine Tlili  
  - 🧑‍💻 Mohamed Aziz Loukil  
  - 🧑‍💻 Mohamed Laatar

🏁 Achieved a **top-tier accuracy of 99.91%** on the MNIST test set — approaching human-level performance.

