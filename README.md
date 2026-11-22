# Satellite Land Use Change and Encroachment Detector

[![Java](https://img.shields.io/badge/Java-21%20%7C%2025%20LTS-ED8B00?style=flat&logo=openjdk&logoColor=white)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.9.6-C71A36?style=flat&logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![JUnit 5](https://img.shields.io/badge/JUnit-5.10.2-25A162?style=flat&logo=junit5&logoColor=white)](https://junit.org/junit5/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17.1-005CED?style=flat)](https://onnxruntime.ai/)
[![Domain](https://img.shields.io/badge/Domain-Geospatial%20Intelligence-blue)](https://en.wikipedia.org/wiki/Geospatial_intelligence)

An automated bi-temporal satellite image change detection and potential encroachment screening system built in **Java** with **ONNX Runtime (CPU)**. Designed for an undergraduate Semester-V curriculum with transparent, explainable algorithms, a rigorous classical baseline, a genuine deep-learning model, and course deep-learning fundamentals.

---

## 1. Project Overview & Scope

This project addresses the problem of unauthorized physical construction and land-cover change in protected environmental zones, forest reserves, and public land buffers.

The codebase is organized into three distinct tiers:
1. **WORKING MAIN SYSTEM:** End-to-end bi-temporal change detection pipeline on real LEVIR-CD satellite rasters comparing a Classical Pixel-Difference Baseline with a Pretrained Siamese ResNet-18 Deep-Learning model, followed by Connected Component Analysis and Potential Encroachment Screening.
2. **EDUCATIONAL DL FUNDAMENTALS:** Plain Java implementations of course fundamentals (Tensors, SVD, Gradient Checking, McCulloch-Pitts gate, Sigmoid neuron, XOR MLP, 4 optimizers, and 5 learning rules) kept cleanly separate in `com.satellite.fundamentals`, `com.satellite.optimizers`, and `com.satellite.learning`.
3. **EXPERIMENTAL / CONCEPTUAL MODULES:** Academic architectural comparisons ([docs/DETECTION_MODELS_COMPARISON.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/DETECTION_MODELS_COMPARISON.md)), recurrent temporal land-use trend simulation (`TemporalLandUseDemo`), and latent-space generative augmentation (`GenerativeAugmentationDemo`).

---

## 2. Summary of Work Completed

Here is a summary of the components implemented, verified, and benchmarked:

### A. Classical Change-Detection Baseline & Simplifications
- **Preserved & Simplified:** Preserved the baseline bi-temporal change pipeline while removing unnecessary design patterns, interfaces, and reflection.
- **Euclidean Differencing:** Computes RGB distance $\Delta I(x,y) = \sqrt{\frac{1}{3}\sum (I_B - I_A)^2}$ in `DifferenceCalculator.java`.
- **Thresholding & Morphology:** Generates binary mask at $\tau=35.0$ and applies $3 \times 3$ morphological opening (noise removal) and closing (hole filling) in `NoiseRemover.java`.
- **Connected Components (BFS):** 8-neighbourhood Breadth-First Search clustering in `ConnectedComponentAnalyzer.java` generating bounding boxes, centroids, and area statistics.
- **Auditable Screening Rules:** Categorizes changes into `NO_SIGNIFICANT_CHANGE`, `POSSIBLE_CHANGE`, or `POSSIBLE_ENCROACHMENT` in `EncroachmentDetector.java`.

### B. Genuine Deep-Learning Model Integration
- **Model:** Pretrained Siamese ResNet-18 (`models/onnx/resnet18_change_detector.onnx`, 44.7 MB) loaded and executed via **ONNX Runtime Java 1.17.1** on standard CPU.
- **Inference Pipeline:** Implemented in `OnnxResNetChangeDetector.java`. Ingests two $(1, 3, 256, 256)$ tensors with ImageNet normalization, passes them through shared ResNet-18 convolutional feature extractors, computes absolute feature differences, and projects via a $1 \times 1$ conv head with bilinear upsampling into a $(1, 1, 256, 256)$ probability map.
- **Empirical Comparison:** Integrated into `ModelComparisonRunner.java`, producing [results/model_comparison.csv](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/results/model_comparison.csv) and [results/model_comparison_chart.png](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/results/model_comparison_chart.png).

### C. Neural Network Fundamentals in Plain Java
- **`SimpleTensor.java`:** N-D tensor operations in pure Java (reshape, flatten, transpose, addition, scalar multiplication, $O(N^3)$ matrix multiplication, inner product, and row-wise broadcasting).
- **`SvdDemo.java`:** Singular Value Decomposition ($A = U S V^T$) with Frobenius reconstruction error validation using Apache Commons Math.
- **`NumericalGradientChecker.java`:** Central-difference approximation $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ verified against analytical gradients with relative error $< 10^{-5}$.
- **`McCullochPittsGate.java`:** Weighted sum threshold neuron with a functional AND gate demonstration.
- **`SigmoidNeuronUnit.java`:** Logistic sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ and derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.
- **`SimpleXorMLP.java`:** 2-4-1 Multilayer Perceptron with forward pass, MSE loss, backpropagation, and Batch Gradient Descent (converging to $\text{MSE} < 0.001$).
- **`ActivationComparison.java`:** Compares Step, Sigmoid, Tanh, and ReLU across $[-3.0, 3.0]$, exporting [experiments/activation_comparison.csv](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/experiments/activation_comparison.csv).

### D. Optimizer Benchmark (Fair Comparison on XOR)
- Benchmarked Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), Momentum ($\beta=0.85$), and Nesterov Accelerated Gradient ($\beta=0.85$) on the identical 2-4-1 XOR problem across 2,000 epochs.
- Documented in [experiments/optimizer_comparison.csv](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/experiments/optimizer_comparison.csv) and [experiments/optimizer_summary.csv](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/experiments/optimizer_summary.csv).

### E. Educational Learning Rules
- Contained in `com.satellite.learning`: Hebbian Learning (`HebbianDemo`), Modified Hebbian / Oja's Rule (`OjaDemo`), Competitive / Winner-Take-All (`CompetitiveLearningDemo`), Error-Correction / Delta Rule (`DeltaRuleDemo`), and Hopfield Network Lyapunov energy minimization (`HopfieldEnergyDemo`).

### F. Advanced Architectures, Temporal, & Generative Modules
- **Object Detection Architecture Analysis:** [docs/DETECTION_MODELS_COMPARISON.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/DETECTION_MODELS_COMPARISON.md) systematically contrasts R-CNN, Fast R-CNN, Faster R-CNN, YOLO, and RetinaNet.
- **Temporal Sequence Modeling:** `TemporalLandUseDemo.java` models multi-year built-up percentage growth trends (2020–2025).
- **Generative Augmentation:** `GenerativeAugmentationDemo.java` demonstrates latent-space representation and synthetic sample interpolation.

### G. Comprehensive Documentation & Viva Preparation Suite
- [docs/VIVA_CODE_MAP.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/VIVA_CODE_MAP.md): 14-file viva study guide detailing Purpose, Key Methods, Inputs, Outputs, and Algorithms.
- [docs/PROJECT_FLOW.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/PROJECT_FLOW.md): Step-by-step 8-stage pipeline breakdown with Mermaid diagram.
- [docs/technical-report.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/technical-report.md): Academic report with mathematical derivations and benchmark analysis.
- [docs/deployment-plan.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/deployment-plan.md): Operational deployment, hardware specs, and CPU execution guide.
- [docs/model-card.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/model-card.md) & [docs/dataset-card.md](file:///c:/Users/manis/Desktop/CODES/DL_Project/Satellite_Land_Use_Change_Encroachment_Detector/docs/dataset-card.md): Standard model and dataset provenance cards.

---

## 3. System Architecture Flow

```
                       LEVIR-CD Dataset
                     (Before Image A + After Image B)
                                |
                   Validation & Dimension Check
                                |
             +------------------+------------------+
             |                                     |
    [Classical Baseline]                  [Deep-Learning Component]
  Euclidean Pixel Difference                Siamese ResNet-18 (ONNX)
             |                                     |
    Binary Thresholding                   Deep Feature Differencing
             |                                     |
    Morphological Opening/Closing         Sigmoid Activation & Threshold
             |                                     |
             +------------------+------------------+
                                |
                    Connected Component Analysis
                   (8-Neighbourhood BFS Clustering)
                                |
                   Potential Encroachment Screening
                     (Changed Area & Cluster Size)
                                |
                      Evaluation & Comparison
                     (IoU, F1, Precision, Recall)
                                |
                 Visualizations & Output Reports
```

---

## 4. Key Results & Empirical Findings

### A. Classical Baseline vs. Deep-Learning Model (LEVIR-CD `test_100_y0000_x0256`)

| Method | Changed Area (px) | Changed Area (%) | Precision | Recall | F1-Score | IoU | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline** | 52,295 | 79.80% | 0.0164 | 0.9990 | 0.0322 | 0.0164 | **68 ms** |
| **Siamese ResNet-18 (DL)** | 7,814 | 11.92% | **0.1341** | 0.6558 | **0.2227** | **0.1253** | 89 ms |

> [!NOTE]
> **Academic Takeaway:** The classical pixel-difference baseline suffers from severe false positives (79.80% marked changed) due to solar angle shifts, shadows, and seasonal vegetation discoloration. The Siamese ResNet-18 deep model extracts semantic features invariant to lighting, suppressing 44,481 false positive pixels and achieving a **7.6x improvement in IoU** (0.1253 vs 0.0164).

### B. Optimizer Benchmark (XOR Problem, 2,000 Epochs)

| Optimizer | Final MSE Loss | Key Observation |
| :--- | :---: | :--- |
| **Batch Gradient Descent (BGD)** | 0.124875 | Plateaus in local saddle regions |
| **Stochastic Gradient Descent (SGD)** | 0.124960 | High gradient variance; slow progress |
| **Momentum ($\beta=0.85$)** | **0.000575** | Accelerates through flat surfaces; rapidly converges |
| **Nesterov Momentum ($\beta=0.85$)** | **0.000659** | Look-ahead gradient dampens oscillations; stable convergence |

---

## 5. Technology Stack

- **Language:** Java 21+ LTS (Tested on Oracle JDK 25)
- **Build System:** Apache Maven 3.9.6 (with portable wrapper `mvnw.cmd` / `mvnw`)
- **Deep Learning Runtime:** ONNX Runtime Java 1.17.1 (CPU inference, OpenMP)
- **Mathematical Tools:** Apache Commons Math 3.6.1
- **Testing:** JUnit 5 (30/30 unit tests passing)
- **Image Processing:** Pure Java Standard Library (`javax.imageio.ImageIO`, `java.awt.image.BufferedImage`)

---

## 6. Quick Start: Build, Test & Run

### 1. Run All Unit Tests
```powershell
.\mvnw.cmd test
```

### 2. Build Executable Uber-JAR
```powershell
.\mvnw.cmd package
```

### 3. Run One-Click Demonstration
```powershell
java -jar target/satellite-detector.jar --demo
```
This single command runs the entire pipeline on LEVIR-CD test imagery, generates all output masks, prints the summary report, and populates `results/`:
- `results/before.png` & `results/after.png` (input satellite images)
- `results/ground_truth.png` (actual building change mask)
- `results/baseline_change_mask.png` (classical pixel difference mask)
- `results/dl_predicted_mask.png` (Siamese ResNet-18 predicted mask)
- `results/highlighted_changes.png` (bounding boxes overlaid on after image)
- `results/comparison_image.png` (4-panel diagnostic board)
- `results/metrics.csv` & `results/model_comparison.csv` (empirical metrics)
- `results/model_comparison_chart.png` (visual benchmark chart)
- `results/result.json` (machine-readable detection summary)

### 4. Run Educational Sub-Modules
```powershell
# Run course fundamentals (Tensors, SVD, Gradient Checker, Neurons, MLP):
java -jar target/satellite-detector.jar --fundamentals

# Run optimizer comparison benchmark:
java -jar target/satellite-detector.jar --optimizers

# Run activation function evaluation:
java -jar target/satellite-detector.jar --activations

# Run educational learning rules:
java -jar target/satellite-detector.jar --learning-rules

# Run temporal land-use trend simulation:
java -jar target/satellite-detector.jar --temporal

# Run generative augmentation simulation:
java -jar target/satellite-detector.jar --generative
```

---

## 7. Project Directory Map

```
Satellite_Land_Use_Change_Encroachment_Detector/
├── pom.xml                                      # Maven configuration (Java 21+, ONNX Runtime 1.17.1)
├── mvnw.cmd / mvnw                              # Maven wrapper scripts
├── README.md                                    # This project guide
├── docs/
│   ├── VIVA_CODE_MAP.md                         # 14-file code guide for technical viva
│   ├── PROJECT_FLOW.md                          # Plain English 8-stage pipeline breakdown
│   ├── technical-report.md                      # Academic report with formulas & benchmark data
│   ├── deployment-plan.md                       # Operational deployment & hardware guide
│   ├── model-card.md                            # Model card for Siamese ResNet-18
│   ├── dataset-card.md                          # Dataset provenance & limitations (LEVIR-CD)
│   └── DETECTION_MODELS_COMPARISON.md           # R-CNN/YOLO/RetinaNet conceptual comparison
├── data/
│   └── processed/LEVIR-CD/test/                 # Real LEVIR-CD test tiles (A, B, label)
├── models/
│   └── onnx/resnet18_change_detector.onnx       # Pretrained Siamese ResNet-18 ONNX model (44.7 MB)
├── experiments/                                 # Optimizer and activation CSV logs
├── results/                                     # Output change masks, comparison charts, CSVs, JSON
└── src/
    ├── main/java/com/satellite/
    │   ├── app/Main.java                        # Application entry point & CLI routing
    │   ├── change/                              # Classical baseline, difference, morphology, BFS
    │   ├── classification/                      # OnnxResNetChangeDetector (ONNX Runtime Java)
    │   ├── evaluation/                          # ModelComparisonRunner, IoU, F1, CSV export
    │   ├── fundamentals/                        # SimpleTensor, SvdDemo, GradientChecker, Neurons, XOR MLP
    │   ├── optimizers/                          # BGD, SGD, Momentum, Nesterov comparison
    │   ├── learning/                            # Hebbian, Oja's, WTA, Delta, Hopfield models
    │   ├── temporal/                            # TemporalLandUseDemo (recurrent sequence model)
    │   └── generative/                          # GenerativeAugmentationDemo (latent space demo)
    └── test/java/com/satellite/                 # JUnit 5 test suite (30 passing tests)
```

---

## 8. Operational & Legal Notice

> [!IMPORTANT]
> All classifications produced by this software (`POSSIBLE_ENCROACHMENT`, `POSSIBLE_CHANGE`, `NO_SIGNIFICANT_CHANGE`) represent automated screening alerts based on bi-temporal raster analysis. They are intended strictly to prioritize field inspections and cadastral review. They do not constitute a legal or definitive determination of land ownership or property boundaries.
