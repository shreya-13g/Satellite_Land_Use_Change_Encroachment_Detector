# Technical Report: Satellite Land Use Change and Encroachment Detection

**Course/Degree:** Bachelor of Technology in Computer Science & Engineering (Semester-V)  
**Project Title:** Satellite Land Use Change and Encroachment Detector  
**Domain:** Geospatial Intelligence (GEOINT) & Bi-Temporal Remote Sensing  
**Platform:** Java 21+ LTS, Apache Maven 3.9.6, ONNX Runtime Java (CPU), JUnit 5  

---

## 1. Executive Summary

Unregulated urbanization and unauthorized physical construction in protected riparian corridors, environmental reserves, and public land pose severe ecological threats. This project implements an automated, reproducible bi-temporal satellite image processing pipeline in Java to detect land-cover transitions and screen for potential encroachment.

The system features:
1. **Classical Change-Detection Baseline:** Euclidean RGB pixel difference, binary thresholding, morphological noise removal (opening/closing), and 8-neighbourhood connected component analysis (BFS).
2. **Real Deep-Learning Model:** Pretrained Siamese ResNet-18 deployed via ONNX Runtime Java on CPU to extract deep semantic visual features and compute structural change masks invariant to illumination and seasonal shifts.
3. **Empirical Evaluation:** Evaluated on real LEVIR-CD building change-detection rasters ($256 \times 256$ pixels), computing Precision, Recall, F1-Score, and Intersection over Union (IoU).
4. **Deep-Learning Fundamentals Suite:** Educational implementations of Tensors, SVD, Numerical Gradient Checking, McCulloch-Pitts neurons, Sigmoid neurons, XOR Multilayer Perceptron, 4 optimizers (BGD, SGD, Momentum, Nesterov), and 5 classical learning rules.

---

## 2. Methodology & Mathematical Formulations

### 2.1 Classical Change-Detection Baseline
Given two co-registered optical satellite rasters $I_A, I_B \in \mathbb{R}^{H \times W \times 3}$:

1. **Euclidean Pixel Distance:**
   $$\Delta I(x, y) = \sqrt{\frac{1}{3} \sum_{c \in \{R, G, B\}} \left( I_B(x, y, c) - I_A(x, y, c) \right)^2}$$
2. **Thresholding:**
   $$M_{\text{raw}}(x, y) = \begin{cases} 1 & \text{if } \Delta I(x, y) \ge \tau \\ 0 & \text{otherwise} \end{cases} \quad (\tau = 35.0)$$
3. **Morphological Noise Filtering:**
   $$M_{\text{clean}} = (M_{\text{raw}} \circ K) \bullet K$$
   where $\circ$ denotes opening (erosion followed by dilation with a $3 \times 3$ kernel to suppress salt-and-pepper noise) and $\bullet$ denotes closing (dilation followed by erosion to bridge minor structural gaps).
4. **Connected Components & Screening:**
   An 8-connectivity Breadth-First Search (BFS) groups contiguous changed pixels into clusters $C_k$. Each cluster yields an axis-aligned bounding box and centroid. If total changed area $\ge 0.40\%$ or any cluster area $\ge 150$ pixels, the status is flagged as **`POSSIBLE_ENCROACHMENT`**.

### 2.2 Deep-Learning Pipeline: Siamese ResNet-18
- **Motivation:** Classical differencing cannot distinguish between true structural changes (e.g., new buildings, cleared land) and radiometric noise (sun angle shifts, surface shadows, phenological vegetation discoloration).
- **Architecture:** A Siamese feature extractor where branches share identical ResNet-18 weights pretrained on ImageNet.
- **Inference Flow:**
  $$\mathbf{F}_A = f_{\text{ResNet}}(I_A), \quad \mathbf{F}_B = f_{\text{ResNet}}(I_B)$$
  $$\mathbf{F}_{\text{diff}} = |\mathbf{F}_B - \mathbf{F}_A|$$
  A lightweight $1 \times 1$ convolutional projection head with bilinear upsampling maps feature differences into a $1 \times 1 \times H \times W$ logit map.
- **Inference Runtime:** Executed in pure Java via `ai.onnxruntime:onnxruntime:1.17.1` on standard CPU hardware without GPU or native Python dependencies.

---

## 3. Empirical Experimental Results

### 3.1 LEVIR-CD Benchmark (Tile `test_100_y0000_x0256`)

| Method | Changed Area (px) | Changed Area (%) | Precision | Recall | F1-Score | IoU | Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline (Pixel Diff)** | 52,295 | 79.80% | 0.0164 | 0.9990 | 0.0322 | 0.0164 | **66 ms** |
| **Siamese ResNet-18 (ONNX CPU)** | 7,814 | 11.92% | **0.1341** | **0.6558** | **0.2227** | **0.1253** | 98 ms |

#### Key Empirical Observations:
- **Baseline False Positive Problem:** The classical pixel difference detector flags 79.80% of the image as changed because slight radiometric variation across seasons causes Euclidean RGB distance to exceed threshold $\tau=35.0$. While Recall is high (0.9990), Precision drops to 0.0164, yielding an IoU of only **0.0164**.
- **Deep Learning Invariance:** The Siamese ResNet-18 extracts deep convolutional features that are invariant to low-level lighting variations. It filters out 44,481 false positive pixels, boosting the IoU from 0.0164 to **0.1253** (a **7.6x improvement**) at a modest CPU latency of 98 ms.

### 3.2 Optimizer Benchmark (XOR Problem, 2,000 Epochs)

| Optimizer | Learning Rate ($\eta$) | Hyperparameters | Initial Loss | Final MSE Loss | Convergence Behaviour |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Batch Gradient Descent (BGD)** | 0.1 | None | 0.250000 | 0.124875 | Plateaus in local saddle |
| **Stochastic Gradient Descent (SGD)** | 0.1 | Sample-by-sample | 0.250000 | 0.124960 | High variance, slow progress |
| **Momentum** | 0.1 | $\beta = 0.85$ | 0.250000 | **0.000575** | Fast convergence; escapes saddle |
| **Nesterov Accelerated Gradient** | 0.1 | $\beta = 0.85$ | 0.250000 | **0.000659** | Smooth trajectory; stable convergence |

---

## 4. Operational & Legal Notice

All encroachment screening indicators produced by this software (`POSSIBLE_ENCROACHMENT`, `POSSIBLE_CHANGE`, `NO_SIGNIFICANT_CHANGE`) are automated computational screening indicators based on bi-temporal raster analysis. They do not constitute legal verification of land-use infringement or property boundary violations. They serve strictly to prioritize human field survey inspections.

