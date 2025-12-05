# Food-101 Subset: Classical ML vs. CNNs vs. Transfer Learning

This repository contains the code and report for my a study on food image classification.  
The goal is to compare classical machine learning models, a custom convolutional neural network (CNN), and a transfer-learning model based on EfficientNetV2B0 on a 10-class subset of the Food-101 dataset. 

---

## Project Overview

- **Dataset:** 10-class subset of Food-101 (2,500 training images, 750 test images; 250/75 per class).  
- **Classes:** chicken curry, chicken wings, fried rice, grilled salmon, hamburger, ice cream, pizza, ramen, steak, sushi. [file:2]
- **Task:** Top-1 multi-class image classification.
- **Key question:** How much more effective are modern deep transfer-learning architectures (EfficientNetV2B0) than classical ML models using the same deep features? 

---

## Repository Structure

food-image-classification/
│
├── README.md
├── requirements.txt
├── classical_models_training.ipynb
├── neural_models_training.ipynb
├── report.pdf
├── data/ 


- **`classical_models_training.ipynb`**  
  Loads a frozen EfficientNetV2B0 backbone, extracts 1,280‑D feature vectors, and trains classical models:
  Logistic Regression, SVM (RBF), Random Forest, KNN, and Gaussian Naive Bayes. 

- **`neural_models_training.ipynb`**  
  Implements two Keras models:
  1. A custom CNN trained from scratch.  
  2. An EfficientNetV2B0 transfer‑learning model with a frozen base and trainable classification head. 

- **`report/`**  
  Contains the full technical report (PDF) describing methodology, experiments, and results in detail. 

---

## Methods

### 1. Classical ML on Deep Features

1. Download 10‑class Food‑101 subset and create train/test generators (image size \(224 \times 224\)). 
2. Use `tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet")` as a frozen feature extractor. 
3. Apply global average pooling to get 1,280‑dimensional feature vectors per image.  
4. Train classical models (scikit‑learn) on these features with basic hyperparameter tuning. 

Best classical performance: **Logistic Regression — 38% test accuracy**. 

### 2. Custom CNN (from scratch)

- 3 convolutional blocks (32, 64, 128 filters) with batch normalization and max pooling.  
- Dense layer with 128 units + batch normalization + dropout 0.5, followed by a 10‑way softmax. 
- Trained with Adam, learning rate \(10^{-3}\), early stopping, and model checkpointing.

Custom CNN performance: **62.53% test accuracy**.

### 3. Transfer Learning with EfficientNetV2B0

- Load EfficientNetV2B0 without top, freeze all layers.  
- Add `GlobalAveragePooling2D` + dense softmax layer (10 classes). Only 12,810 parameters are trainable out of ~5.9M. 
- Train for ~10–15 epochs with Adam and sparse categorical cross‑entropy.

Transfer‑learning performance: **79.6% test accuracy**, beating both classical ML and the custom CNN.

---

## Key Results

| Model                                   | Test Accuracy |
|----------------------------------------|---------------|
| Logistic Regression on deep features   | 38.0%         |
| SVM (RBF) on deep features             | 28.0%         |
| Random Forest on deep features         | 23.0%         |
| K‑Nearest Neighbors on deep features   | 17.0%         |
| Gaussian Naive Bayes on deep features  | 14.0%         |
| Custom CNN (from scratch)             | 62.53%        |
| EfficientNetV2B0 (transfer learning)   | 79.6%         |

Deep transfer learning with EfficientNetV2B0 more than doubles the accuracy of the best classical baseline and substantially outperforms the custom CNN while training only a small classification head. [file:2][file:4]

---
## How to Run

1. **Clone the repository**

2. **Install dependencies**

Create a virtual environment (optional) and install dependencies:
Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Run the notebooks**

You can open the notebooks locally or in Google Colab (Colab is preffered to tae advantage of free GPU access).

- `classical_models_training.ipynb` — runs the feature extraction and classical ML experiments.  
- `neural_models_training.ipynb` — trains the custom CNN and EfficientNetV2B0 models and generates plots used in the report. [file:4]

---

