# Stage 3 Change Detection & Encroachment Screening Report

**Evaluation Scope:** LEVIR-CD Processed Test Dataset (256x256 Tiles)
**Method:** Classical Multitemporal Radiometric Differencing, Morphological Noise Removal & Connected Component Screening

## 1. Quantitative Benchmark Metrics

| Metric | Evaluated Value |
|:---|:---:|
| **Evaluated Samples** | 6 tiles |
| **Mean IoU (Intersection over Union)** | **0.0126** |
| **Mean F1-Score** | **0.0243** |
| **Mean Precision** | **0.0137** |
| **Mean Change Recall** | **0.4675** |
| **Mean Changed-Area Error** | **50080.8 pixels** |
| **Total Runtime** | **146 ms** |

## 2. Global Confusion Matrix (Pixel Aggregation)

| Classification | Ground Truth Change | Ground Truth Unchanged |
|:---|:---:|:---:|
| **Predicted Change** | TP: 3,543 | FP: 305,958 |
| **Predicted Unchanged** | FN: 5,473 | TN: 78,242 |

## 3. Encroachment Screening Breakdown

- **Possible Encroachments:** 6
- **Possible Changes:** 0
- **No Significant Change:** 0

## 4. Operational Disclaimer

> [!NOTE]
> This report represents automated satellite screening outputs. Findings indicate priority zones for field verification and cadastral inspection and do not constitute a legal or definitive land ownership adjudication.
