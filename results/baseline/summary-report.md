# Stage 3 Change Detection & Encroachment Screening Report

**Evaluation Scope:** LEVIR-CD Processed Test Dataset (256x256 Tiles)
**Method:** Classical Multitemporal Radiometric Differencing, Morphological Noise Removal & Connected Component Screening

## 1. Quantitative Benchmark Metrics

| Metric | Evaluated Value |
|:---|:---:|
| **Evaluated Samples** | 1 tiles |
| **Mean IoU (Intersection over Union)** | **0.0164** |
| **Mean F1-Score** | **0.0322** |
| **Mean Precision** | **0.0165** |
| **Mean Change Recall** | **0.7613** |
| **Mean Changed-Area Error** | **51164.0 pixels** |
| **Total Runtime** | **102 ms** |

## 2. Global Confusion Matrix (Pixel Aggregation)

| Classification | Ground Truth Change | Ground Truth Unchanged |
|:---|:---:|:---:|
| **Predicted Change** | TP: 861 | FP: 51,434 |
| **Predicted Unchanged** | FN: 270 | TN: 12,971 |

## 3. Encroachment Screening Breakdown

- **Possible Encroachments:** 1
- **Possible Changes:** 0
- **No Significant Change:** 0

## 4. Operational Disclaimer

> [!NOTE]
> This report represents automated satellite screening outputs. Findings indicate priority zones for field verification and cadastral inspection and do not constitute a legal or definitive land ownership adjudication.
