# Object Detection Architecture Comparison for Geospatial Imagery

This document provides an academically rigorous conceptual comparison of single-stage and two-stage object detection architectures referenced in the project requirements.

> [!NOTE]
> **Academic Transparency Notice:**
> The primary change detection pipeline in this repository uses a **Classical Radiometric Baseline** and a **Deep-Learning Siamese ResNet-18 Model** (`models/onnx/resnet18_change_detector.onnx`).
> The detection architectures compared in this document are analyzed **conceptually** from published computer-vision literature.
> In accordance with academic honesty principles, **no simulated or fabricated accuracy numbers are reported below**.

---

## 1. Architectural Taxonomy and Conceptual Matrix

| Architecture | Paradigm / Family | Two-Stage vs One-Stage | Proposed Backbone | Primary Strengths | Primary Weaknesses | Suitability for Satellite / Remote Sensing | Implementation Status in Repo |
|:---|:---|:---:|:---|:---|:---|:---|:---:|
| **R-CNN** (Girshick et al., 2014) | Region Proposal + CNN | Two-Stage | AlexNet / VGG-16 | Pioneered deep CNN features for localized detection. | Extremely slow (~47s/image); proposals generated via heuristic Selective Search; no shared computation. | Unsuitable for real-time or large-scale satellite tile screening due to prohibitive computational latency. | **Conceptual Comparison** (Not executed) |
| **Fast R-CNN** (Girshick, 2015) | Region of Interest (RoI) Pooling | Two-Stage | VGG-16 | Shared convolutional feature map across entire image; RoI Pooling speeds inference ~10x over R-CNN. | Region proposals still generated on CPU via external Selective Search; end-to-end training bottleneck. | Better, but external proposal extraction still hampers processing of high-resolution gigapixel rasters. | **Conceptual Comparison** (Not executed) |
| **Faster R-CNN** (Ren et al., 2015) | Region Proposal Network (RPN) | Two-Stage | ResNet-50 / ResNet-101 + FPN | Fully differentiable end-to-end architecture; RPN replaces Selective Search; high localization accuracy. | Anchor hyperparameter tuning required; higher inference latency than single-stage detectors (~5-10 FPS). | Highly effective for small, dense geospatial targets (e.g., naval vessels, isolated buildings, mining pits). | **Conceptual Comparison** (Not executed) |
| **YOLO** (You Only Look Once - v1 to v8) | Dense Single-Stage Grid Regression | One-Stage | DarkNet / CSP-DarkNet | Real-time throughput (30-100+ FPS); global contextual reasoning; unified regression loss. | Struggles with very small, closely packed objects compared to two-stage architectures with FPN. | Excellent for operational real-time monitoring of vehicles, aircraft, and large civil constructions in satellite feeds. | **Conceptual Comparison** (Not executed) |
| **RetinaNet** (Lin et al., 2017) | Dense Anchor-Based with Focal Loss | One-Stage | ResNet + Feature Pyramid Network (FPN) | **Focal Loss** resolves extreme foreground-background class imbalance; matches two-stage accuracy at one-stage speed. | High memory footprint due to multi-scale pyramid levels; anchor density requires careful calibration. | Outstanding for geospatial intelligence where tiny change targets occupy < 1% of the raster background. | **Conceptual Comparison** (Not executed) |

---

## 2. Theoretical Analysis of Key Concepts

### A. Two-Stage vs. One-Stage Detectors
- **Two-Stage Detectors (R-CNN, Fast R-CNN, Faster R-CNN):**
  Divide the problem into two sequential phases:
  1. *Hypothesis Generation:* Propose candidate regions with high "objectness" probability.
  2. *Classification & Refinement:* Extract features from candidate regions, classify the object category, and regress bounding box offsets.
  *Trade-off:* Highest localization precision, but slower frame rates.

- **One-Stage Detectors (YOLO, RetinaNet):**
  Treat object detection as a single-pass regression problem directly from pixel grids to bounding box coordinates and class probabilities.
  *Trade-off:* Real-time speed and low latency, but historically lower recall on small objects until Focal Loss and Feature Pyramids were introduced.

### B. The Class-Imbalance Problem in Remote Sensing & Focal Loss
In satellite imagery, background terrain (forest, soil, water) represents over 95–99% of total raster pixels. Standard cross-entropy loss is overwhelmed by easily classified negative examples.
Lin et al. (2017) introduced **Focal Loss**:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $(1 - p_t)^\gamma$ acts as a modulating factor. When an easy background pixel has $p_t \approx 1$, the modulating factor approaches zero, down-weighting easy negatives and focusing gradients on hard, ambiguous change boundaries.

---

## 3. Summary of Project Implementation Choices

- **Classical Baseline:** Multitemporal Radiometric Difference + Morphological Filtering + 8-neighbour Connected Components.
- **Deep Learning Model:** Siamese ResNet-18 Deep Feature Differencing with ONNX Runtime Java.
- **Detection Architectures (R-CNN, YOLO, RetinaNet):** Documented and evaluated conceptually above; not executed in production code to preserve project simplicity, stability, and viva defendability.

