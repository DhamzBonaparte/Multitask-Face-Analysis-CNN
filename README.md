# 🧠 Facial Analysis: Gender Classifier & Multi-Task CNN

This repository contains two Deep Learning projects built from scratch using **TensorFlow/Keras** and **Convolutional Neural Networks (CNNs)**. 

Both models were trained on the **UTKFace** dataset using custom `tf.data` pipelines.

---

## 📂 Project 1: Multi-Task Learning (Age + Gender)
A single CNN with a shared Convolutional base that splits into two separate output branches to predict two different targets simultaneously.

### 📊 Performance
- **Gender Classification Accuracy:** **82.14%**
- **Age Regression MAE:** **9.68 years** (Average error)

### 🛠️ Architecture Highlights
- **Shared Base:** 3 Convolutional Blocks (Conv2D + BatchNormalization + ReLU + MaxPooling).
- **Output Heads:** 
  - `gender_output`: Dense(1, Sigmoid) -> Binary Crossentropy loss.
  - `age_output`: Dense(1, Linear) -> Mean Absolute Error (MAE) loss.
- **Techniques:** Data Augmentation (Flip, Rotate, Zoom) and `ReduceLROnPlateau` callbacks.

---

## 🚻 Project 2: Standalone Gender Classifier
Before tackling multi-task learning, I built a standalone binary classifier dedicated entirely to predicting gender.

### 📊 Performance
- **Gender Classification Accuracy:** **86.00%**
- **Validation Loss:** **0.33**

### 🛠️ Architecture Highlights
- **Architecture:** Sequential CNN with 3 Conv2D blocks and Dense heads.
- **Optimizer:** Adam with Dynamic Learning Rate scheduling.
- **Data Pipeline:** Optimized `tf.data` generator wrapping Hugging Face datasets.

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/DhamzBonaparte/CNN---Age-Gender.git