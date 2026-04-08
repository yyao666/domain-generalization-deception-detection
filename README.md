# Domain Generalization for Multimodal Deception Detection

## 🎯 Research Objective

The goal of this project is to develop a **robust deception detection model** that generalizes across:

- Different ethnic groups
- Different languages
- Different speakers

This work focuses on **domain generalization** using audio-based representations and contrastive learning.

---

## 🧠 Overview

This repository implements several domain generalization approaches:

- DG Baseline (LODO protocol)
- Gradient Reversal (Domain Adversarial Training)
- Contrastive Learning
- Multi-Objective Learning (Focal + NT-Xent)

All methods are evaluated under **Leave-One-Domain-Out (LODO)** setting.

---

## 🏗️ Methods

### 1. DG Baseline

- ResNet-50 audio encoder
- Spectrogram input
- Cross-entropy loss
- Leave-One-Domain-Out evaluation

---

### 2. Gradient Reversal

- Domain adversarial training
- Gradient Reversal Layer (GRL)
- Domain classifier
- Domain-invariant representation learning

Reference: Ganin & Lempitsky, 2015

---

### 3. Contrastive Learning

- Projection head
- NT-Xent contrastive loss
- Two-view spectrogram augmentation
- Representation alignment across domains

Reference: Chen et al., 2020

---

### 4. Multi-Objective Learning

- Cross-entropy loss
- Focal loss (class imbalance)
- NT-Xent loss (representation learning)
- Multi-objective optimization

References:  
Lin et al., 2017  
Chen et al., 2020

---


## 📊 Dataset

This project uses spectrogram representations extracted from:

- Multimodal deception detection dataset
- Multi-ethnic speakers
- Multi-language scenarios


Domains:

- Chinese
- Malay
- Hindi



Evaluation:

Leave-One-Domain-Out **(LODO)**



Example:

- Train: Chinese + Malay

- Test: Hindi

---


## 🧪 Methods Comparison


| Method      | Domain Invariance | Representation Learning |
| ----------- | ----------------- | ----------------------- |
| Baseline    | ❌                 | ❌                       |
| GRL         | ✅                 | ❌                       |
| Contrastive | ❌                 | ✅                       |
| Combined    | ✅                 | ✅                       |


---

## 🧠 Key Contributions
- Domain generalization for deception detection
- Multi-objective learning framework
- Cross-ethnic robustness analysis
- Contrastive learning for domain invariance


---




## 📚 References

[1] Ganin, Y. and Lempitsky, V., 2015.  
Unsupervised domain adaptation by backpropagation.  
ICML 2015.  
https://arxiv.org/abs/1409.7495

[2] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G., 2020.  
A Simple Framework for Contrastive Learning of Visual Representations.  
ICML 2020.  
https://arxiv.org/abs/2002.05709

[3] Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P., 2017.  
Focal Loss for Dense Object Detection.  
ICCV 2017.  
https://arxiv.org/abs/1708.02002
