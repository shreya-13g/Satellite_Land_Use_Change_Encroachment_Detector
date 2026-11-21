# Stage 3 Change Detection & Encroachment Screening Report

**Evaluation Scope:** LEVIR-CD Processed Test Dataset (256x256 Tiles)
**Method:** Classical Multitemporal Radiometric Differencing, Morphological Noise Removal & Connected Component Screening

## 1. Quantitative Benchmark Metrics

| Metric | Evaluated Value |
|:---|:---:|
| **Evaluated Samples** | 80 tiles |
| **Mean IoU (Intersection over Union)** | **0.0790** |
| **Mean F1-Score** | **0.1222** |
| **Mean Precision** | **0.0901** |
| **Mean Change Recall** | **0.2728** |
| **Mean Changed-Area Error** | **19259.9 pixels** |
| **Total Runtime** | **651 ms** |

## 2. Global Confusion Matrix (Pixel Aggregation)

| Classification | Ground Truth Change | Ground Truth Unchanged |
|:---|:---:|:---:|
| **Predicted Change** | TP: 186,614 | FP: 1,742,804 |
| **Predicted Unchanged** | FN: 202,014 | TN: 3,111,448 |

## 3. Encroachment Screening Breakdown

- **Possible Encroachments:** 80
- **Possible Changes:** 0
- **No Significant Change:** 0

## 4. Operational Disclaimer

> [!NOTE]
> This report represents automated satellite screening outputs. Findings indicate priority zones for field verification and cadastral inspection and do not constitute a legal or definitive land ownership adjudication.
