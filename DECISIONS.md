# Architectural and Methodological Design Decisions

This document details the engineering, statistical, and clinical justifications behind every major design decision in the **Cascaded Dermatological Diagnosis & Severity Grading System**.

---

## 1. Why a Two-Stage (Cascaded) Architecture instead of a Single 6-Class Model?
* **Clinical Decoupling:** In real-world dermatology, pathology diagnosis (screening for cancer vs. inflammation) is fundamentally distinct from staging/grading (determining topical vs. systemic treatment for eczema).
* **Feature Space Purity:** A single flat classifier forces the convolutional backbone to simultaneously learn subtle textural variations (mild redness vs. severe lichenification in eczema) while also learning macro-level neoplastic features (melanocytic pigment networks, asymmetry in melanoma). This leads to conflicting gradient updates and negative feature interference.
* **Triage Protocol:** If Melanoma is diagnosed, calculating eczema severity is dangerous and clinically nonsensical. The cascaded pipeline immediately raises an oncology alarm and bypasses Stage 2.

---

## 2. Why MobileNetV2 was Chosen Over ResNet-50, VGG-16, and Vision Transformers (ViT)
* **Depthwise Separable Convolutions:** Standard convolutions perform spatial and channel filtering together ($D_K \cdot D_K \cdot M \cdot N$). MobileNetV2 factorizes this into a $3 \times 3$ Depthwise convolution followed by a $1 \times 1$ Pointwise convolution, reducing compute by $\approx 8\times$ to $9\times$.
* **Inverted Residual Bottlenecks:** Memory footprint is minimized by passing skip connections through thin bottleneck representations while expanding channels internally for non-linear feature transformation.
* **Small Dataset Generalization:** Heavy backbones (VGG-16: 138M params, ViT: 86M params) severely overfit on clinical skin datasets containing only hundreds to low thousands of images. MobileNetV2 (3.5M params) converges stably without memorizing photographic noise.
* **Edge / Mobile Feasibility:** Healthcare triage in developing countries requires point-of-care operation on low-cost tablets or smartphones. MobileNetV2 delivers $\approx 32\text{ ms}$ inference on standard CPUs with a lightweight $\approx 13.6\text{ MB}$ footprint.

---

## 3. Why Grad-CAM was Chosen for Explainability (XAI)
* **Clinical Accountability:** Deep neural networks cannot be deployed as black boxes in medicine. False positives or false negatives require visual auditability.
* **Artifact Detection:** Dermoscopic photographs often feature hair, air bubbles, ruler markings, and vignetting. Grad-CAM verifies whether the model is truly attending to lesion borders and erythema, or erroneously keying on photographic artifacts.
* **Architecture Agnostic:** Grad-CAM utilizes gradients flowing into the final convolutional layer (`model.features[-1]`), capturing high-level semantic localization without modifying model architecture.

---

## 4. Why Boxplots Were Used in Exploratory Data Analysis (EDA)
* **Outlier & Glare Identification:** Standard dermoscopic cameras frequently cause specular reflection (flash glare), represented by extreme outliers above the 75th percentile ($Q_3 + 1.5 \times \text{IQR}$). Dark shadows produce outliers below $Q_1$.
* **Interquartile Range (IQR) Comparison:** Boxplots immediately show whether lighting distribution is comparable across disease classes, justifying ImageNet channel standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

---

## 5. Training Hyperparameter Choices
* **Loss Function:** Categorical Cross-Entropy ($\mathcal{L}_{CE} = -\sum y_i \log(\hat{y}_i)$) with softmax activation.
* **Optimizer:** Adam ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$), updating only unfrozen parameters (`features[15:]` and `classifier[1]`).
* **Learning Rate Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.1, patience=2)` prevents gradient oscillations when validation loss reaches a plateau.
* **Training Platform:** Trained on Google Colab with an **NVIDIA Tesla T4 GPU (CUDA)** to accelerate backpropagation and avoid thermal throttling on local hardware.
