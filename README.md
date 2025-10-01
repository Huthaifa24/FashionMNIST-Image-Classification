# FashionMNIST Classification Project 👕👟👗

This project explores building and training deep learning models on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which contains **28x28 grayscale images** of 10 clothing categories such as shirts, sneakers, and coats.  

The goal is to compare different models, starting from simple architectures and gradually moving to a more complex **TinyVGG CNN architecture**.  

---

## 📌 Models Built

### 1. Linear Model
- A single fully connected layer.
- Used as a **baseline** to see how well a simple model performs.
- Limitations: Cannot capture complex spatial features in images.

### 2. Non-Linear Model
- Added **activation functions (ReLU)** and more fully connected layers.
- Performs better than the linear model, but still not ideal for image data.

### 3. TinyVGG Model (CNN)
- A **convolutional neural network** inspired by the **TinyVGG architecture** from the amazing [CNN Explainer website](https://poloclub.github.io/cnn-explainer/).  
- CNNs are well-suited for image classification because they capture **spatial hierarchies and patterns**.
- Architecture highlights:
  - Convolution → ReLU → Pooling layers
  - Fully connected classifier head
  - Smaller version of VGG networks, hence "TinyVGG"

This was my **second deep learning model experiment after linear and non-linear models**, and it showed significant performance improvements.

---

## ⚡ Training Details
- Dataset: **FashionMNIST**
- Optimizer: SGD / Adam (depending on experiment)
- Loss Function: CrossEntropyLoss
- Metrics: Accuracy

---

## 📊 Results

### Confusion Matrix
![Confusion Matrix](https://github.com/Huthaifa24/FashionMNIST-Image-Classification/blob/4a38c2569239db8a623f726557b30f8b44edf310/evaluation/ff.png)

### Training & Testing Curves
Loss and accuracy curves over epochs:
![Train Test Loss Accuracy](https://github.com/Huthaifa24/FashionMNIST-Image-Classification/blob/4a38c2569239db8a623f726557b30f8b44edf310/evaluation/final_curves.png)

### Random Predictions
Visualizations of some random test samples and their predicted labels:
![Random Predictions](https://github.com/Huthaifa24/FashionMNIST-Image-Classification/blob/4a38c2569239db8a623f726557b30f8b44edf310/random%20_preds/5.png)
![Random Predictions](https://github.com/Huthaifa24/FashionMNIST-Image-Classification/blob/4a38c2569239db8a623f726557b30f8b44edf310/random%20_preds/3.png)
![Random Predictions](https://github.com/Huthaifa24/FashionMNIST-Image-Classification/blob/4a38c2569239db8a623f726557b30f8b44edf310/random%20_preds/1.png)

## 📚 References
- Dataset: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- CNN Architecture Inspiration: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
