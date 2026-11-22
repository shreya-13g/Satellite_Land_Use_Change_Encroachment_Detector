# Viva Code Map — 14 Critical Files to Explain

This document contains **only the essential classes** needed to explain the entire project during the technical viva and demonstration. The student only needs to understand these 14 files.

---

## 1. Production Satellite Detection Pipeline

### 1. `Main.java`
* **File:** `src/main/java/com/satellite/app/Main.java`
* **Purpose:** Main entry point and CLI orchestrator for the entire system.
* **Important Method:** `runDemo(Map<String, String> params)`
* **Input:** Command-line flags (e.g. `--demo`, `--detect-change`), configuration paths.
* **Output:** Executes pipeline, prints benchmark metrics to console, and saves artifacts to `results/`.
* **Algorithm (3 sentences):**
  Loads application configuration and verifies project directories. Parses command-line arguments to run either the full automated demonstration or single-pair change detection. Calls the Classical Baseline and the Deep-Learning Siamese ResNet-18 model on LEVIR-CD test rasters, benchmarks both, and saves visualizations and summary CSVs.

---

### 2. `DifferenceCalculator.java`
* **File:** `src/main/java/com/satellite/change/DifferenceCalculator.java`
* **Purpose:** Computes pixel-wise radiometric difference between Before and After images for the classical baseline.
* **Important Method:** `calculateDifference(BufferedImage imgA, BufferedImage imgB)`
* **Input:** Two co-registered RGB satellite images ($T_1$ and $T_2$) of dimensions $W \times H$.
* **Output:** 2D floating-point difference map `float[H][W]` in range $[0.0, 255.0]$.
* **Algorithm (3 sentences):**
  Iterates through corresponding pixel coordinates $(x, y)$ in both images and extracts RGB components. Computes the Euclidean distance:
  $$\Delta = \sqrt{(R_B - R_A)^2 + (G_B - G_A)^2 + (B_B - B_A)^2}$$
  Stores the resulting intensity distance in a 2D float grid representing raw radiometric change.

---

### 3. `ChangeMaskGenerator.java`
* **File:** `src/main/java/com/satellite/change/ChangeMaskGenerator.java`
* **Purpose:** Binarizes continuous difference maps into binary change masks.
* **Important Method:** `generateMask(float[][] diffMap)`
* **Input:** 2D difference map `float[H][W]`.
* **Output:** 2D binary mask `byte[H][W]` (1 = changed, 0 = unchanged).
* **Algorithm (2 sentences):**
  Compares each pixel difference against a calibrated radiometric threshold (default: 35.0 or dynamic Otsu threshold). If the difference is greater than or equal to the threshold, the pixel is marked as changed (`1`), otherwise unchanged (`0`).

---

### 4. `NoiseRemover.java`
* **File:** `src/main/java/com/satellite/change/NoiseRemover.java`
* **Purpose:** Eliminates sensor noise, isolated specks, and fills small voids in the change mask.
* **Important Method:** `cleanMask(byte[][] rawMask)`
* **Input:** Raw binary mask `byte[H][W]`.
* **Output:** Cleaned binary mask `byte[H][W]`.
* **Algorithm (3 sentences):**
  Applies mathematical morphological opening (erosion followed by dilation with a $3 \times 3$ kernel) to eliminate isolated noise pixels. Then applies morphological closing (dilation followed by erosion) to close tiny interior pinholes in change clusters. Finally filters out any residual connected clusters whose total area is below `minRegionArea` (30 pixels).

---

### 5. `ConnectedComponentAnalyzer.java`
* **File:** `src/main/java/com/satellite/change/ConnectedComponentAnalyzer.java`
* **Purpose:** Groups adjacent changed pixels into coherent geographic change clusters.
* **Important Method:** `findRegions(byte[][] mask)`
* **Input:** Cleaned binary change mask `byte[H][W]`.
* **Output:** List of `ChangeRegion` objects with bounding boxes, pixel areas, and centroids.
* **Algorithm (3 sentences):**
  Iterates across the mask and initiates a Breadth-First Search (BFS) using 8-connectivity from unvisited changed pixels. Accumulates all reachable changed pixels into a cluster while recording minimum and maximum $x$ and $y$ coordinates. Computes the bounding box rectangle, centroid coordinate, and retains clusters whose pixel count exceeds the minimum area threshold.

---

### 6. `EncroachmentDetector.java`
* **File:** `src/main/java/com/satellite/detection/EncroachmentDetector.java`
* **Purpose:** Screens detected change regions against structural encroachment rules.
* **Important Method:** `evaluate(String sampleId, int changedPixels, int totalPixels, List<ChangeRegion> regions)`
* **Input:** Total changed pixels, total image pixels, and detected connected regions.
* **Output:** `EncroachmentResult` containing screening status, confidence score, and operational disclaimer.
* **Algorithm (3 sentences):**
  Calculates the percentage of the image changed and identifies the largest coherent cluster. Assigns **"POSSIBLE_ENCROACHMENT"** if the changed area exceeds 0.3% of the image or contains any single cluster $\ge 150$ pixels. Appends an operational disclaimer stating that findings indicate priority zones for field verification and do not constitute legal land ownership adjudication.

---

### 7. `OnnxResNetChangeDetector.java`
* **File:** `src/main/java/com/satellite/classification/OnnxResNetChangeDetector.java`
* **Purpose:** Executes deep-learning bi-temporal change detection using a pretrained Siamese ResNet-18 model via ONNX Runtime Java.
* **Important Method:** `processPair(BufferedImage imgA, BufferedImage imgB, BufferedImage gtMask, String sampleId)`
* **Input:** Before image A, After image B, optional Ground Truth mask.
* **Output:** `DetectionOutput` containing deep change mask, metrics, and encroachment result.
* **Algorithm (4 sentences):**
  Normalizes Image A and Image B using ImageNet mean and standard deviation into NCHW tensors `[1, 3, 256, 256]`. Runs CPU inference on `models/onnx/resnet18_change_detector.onnx`, where the twin ResNet-18 backbones extract 512-channel deep spatial feature maps and compute spatial L2 distance. Applies adaptive statistical thresholding to generate a clean binary change mask. Passes the mask to morphological filtering, 8-neighbour BFS region extraction, and encroachment screening.

---

### 8. `ChangeMetrics.java`
* **File:** `src/main/java/com/satellite/evaluation/ChangeMetrics.java`
* **Purpose:** Computes quantitative pixel-level segmentation and change-detection evaluation metrics against ground truth.
* **Important Method:** `compute(byte[][] pred, byte[][] gt, String sampleId, long runtimeMs)`
* **Input:** Predicted binary mask `pred[H][W]` and ground truth mask `gt[H][W]`.
* **Output:** `ChangeMetrics` object containing TP, FP, FN, TN, Accuracy, Precision, Recall, F1, IoU, and Area Error.
* **Algorithm (3 sentences):**
  Compares corresponding pixels across the prediction and ground-truth arrays:
  - $TP$: $\text{pred}=1 \land \text{gt}=1$, $FP$: $\text{pred}=1 \land \text{gt}=0$, $FN$: $\text{pred}=0 \land \text{gt}=1$, $TN$: $\text{pred}=0 \land \text{gt}=0$.
  Computes Intersection over Union: $\text{IoU} = \frac{TP}{TP + FP + FN}$ and F1-Score: $\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$.
  Returns genuine zero-divided-safe metrics with execution runtime.

---

### 9. `ModelComparisonRunner.java`
* **File:** `src/main/java/com/satellite/evaluation/ModelComparisonRunner.java`
* **Purpose:** Conducts a head-to-head empirical comparison between Classical Baseline and Deep Learning.
* **Important Method:** `compareModels(List<Path> filesA, List<Path> filesB, List<Path> filesL, Path resultsDir, int limit)`
* **Input:** Paths to Before, After, and Ground-truth test tiles from LEVIR-CD.
* **Output:** Saves `results/model_comparison.csv` and renders `results/model_comparison_chart.png`.
* **Algorithm (3 sentences):**
  Evaluates all test tiles sequentially through the Classical Baseline and the Siamese ResNet-18 model. Aggregates mean Precision, Recall, F1, IoU, and total runtime across both methods. Exports the comparison table to CSV and renders a 2D comparative bar chart visualizing the performance gains of deep features over raw pixel differences.

---

## 2. Educational Deep-Learning Fundamentals

### 10. `SimpleTensor.java`
* **File:** `src/main/java/com/satellite/fundamentals/SimpleTensor.java`
* **Purpose:** Demonstrates core N-dimensional tensor operations in plain Java using basic 1D/2D arrays.
* **Important Method:** `matmul(SimpleTensor other)` and `transpose()`
* **Input:** 1D or 2D double arrays and dimensions.
* **Output:** New `SimpleTensor` containing operation results.
* **Algorithm (3 sentences):**
  Stores flat double arrays alongside an integer shape array and provides intuitive methods for reshaping, flattening, transposition, addition, and scalar scaling. Implements standard matrix multiplication: $C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}$ with dimension validation. Demonstrates vector inner product (dot product) and row-wise broadcasting.

---

### 11. `NumericalGradientChecker.java`
* **File:** `src/main/java/com/satellite/fundamentals/NumericalGradientChecker.java`
* **Purpose:** Validates analytical neural network gradients using central finite differences.
* **Important Method:** `computeNumericalGradient(DoubleUnaryOperator f, double x, double epsilon)`
* **Input:** Objective function $f$, evaluation point $x$, small perturbation $\epsilon$ ($10^{-5}$).
* **Output:** Numerical gradient approximation $\approx f'(x)$.
* **Algorithm (2 sentences):**
  Evaluates the symmetric central-difference quotient:
  $$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2 \epsilon}$$
  Compares the numerical result against the analytical derivative and verifies that relative error is within tolerance ($< 10^{-4}$).

---

### 12. `SimpleXorMLP.java`
* **File:** `src/main/java/com/satellite/fundamentals/SimpleXorMLP.java`
* **Purpose:** Demonstrates neural learning from scratch on the non-linearly separable XOR problem.
* **Important Method:** `train(double[][] X, double[] Y, int epochs, double learningRate)`
* **Input:** 2D input vectors (e.g. $[0,0], [0,1], [1,0], [1,1]$) and target outputs ($[0, 1, 1, 0]$).
* **Output:** Final Mean Squared Error (MSE) loss and trained predictions.
* **Algorithm (4 sentences):**
  Implements a 2-4-1 network architecture with Sigmoid activations: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  Performs forward propagation to compute hidden and output activations. Computes output error $\delta_2 = (\hat{y} - y) \cdot \sigma'(z_2)$ and hidden error $\delta_1 = (\delta_2 W_2^T) \cdot \sigma'(z_1)$ using the chain rule. Updates all weights and biases using batch gradient descent until MSE loss converges to $< 0.001$.

---

### 13. `MomentumOptimizer.java`
* **File:** `src/main/java/com/satellite/optimizers/MomentumOptimizer.java`
* **Purpose:** Accelerates gradient descent and dampens oscillations using Polyak momentum.
* **Important Method:** `update(List<Tensor> params, List<Tensor> grads, double lr)`
* **Input:** Trainable parameter tensors, analytical gradient tensors, learning rate $\eta$.
* **Output:** Updated parameter tensors in-place.
* **Algorithm (3 sentences):**
  Maintains an internal velocity buffer $v$ for each parameter tensor initialized to zero. Updates the velocity buffer as an exponential moving average: $v = \beta \cdot v + \eta \cdot \text{grad}$ (where $\beta = 0.85$). Adjusts the parameter tensor by subtracting velocity: $\theta \leftarrow \theta - v$, enabling fast traversal through flat ravines.

---

### 14. `TemporalLandUseDemo.java`
* **File:** `src/main/java/com/satellite/temporal/TemporalLandUseDemo.java`
* **Purpose:** Demonstrates temporal sequence modeling of multi-year land-use transitions without overcomplicating.
* **Important Method:** `runTemporalDemo(int[] years, double[] builtUpPercentages)`
* **Input:** Time series of observation years and built-up land percentages (e.g., 2018: 10%, 2020: 13%, 2022: 17%, 2024: 23%, 2026: 31%).
* **Output:** Forecast for future time point (e.g., 2028: 41.1%) and annual expansion rate.
* **Algorithm (3 sentences):**
  Computes the compound annual growth rate across the historical bi-temporal sequence. Updates a recurrent hidden state $h_t = \tanh(w_h \cdot h_{t-1} + w_x \cdot x_t)$ to capture expansion momentum over time. Projects the next temporal horizon by combining growth rate trends with the recurrent persistence state.

