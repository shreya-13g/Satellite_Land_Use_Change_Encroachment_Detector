# Deployment & Operational Guide

**System:** Satellite Land Use Change and Encroachment Detector  
**Runtime:** Java Virtual Machine (JDK 21 or 25 LTS), ONNX Runtime Java 1.17.1  
**Target Environment:** Local Workstation / Server CPU (No GPU required)  

---

## 1. System Requirements & Hardware Specifications

- **Operating System:** Windows 10/11 (64-bit), Ubuntu 22.04 LTS+, or macOS (x86_64 / arm64)
- **Java Runtime Environment:** JDK 21 LTS or JDK 25 LTS (Oracle or Eclipse Temurin)
- **Memory (RAM):** Minimum 2 GB available RAM (Heap allocation: `-Xmx2g`)
- **Processor:** Dual-core 2.0 GHz or higher (Tested on 8-core AMD/Intel CPU)
- **Storage:** ~300 MB disk space (includes shaded JAR and ONNX model weights)
- **GPU:** Optional. ONNX Runtime runs natively on CPU using optimized OpenMP and BLAS routines.

---

## 2. Packaging and Verification

Build the self-contained executable shaded JAR:
```powershell
.\mvnw.cmd clean package
```
The build produces:
- `target/satellite-detector.jar` (~98 MB, includes all dependencies and ONNX Java bindings).

To verify the test suite:
```powershell
.\mvnw.cmd test
```

---

## 3. Operational Execution Modes

### 3.1 One-Click Demonstration & Evaluation Mode
Runs the complete baseline, deep learning inference, encroachment screening, and metric evaluation on the LEVIR-CD test raster:
```powershell
java -jar target/satellite-detector.jar --demo
```
**Artifacts Generated in `results/`:**
- `before.png`: Time $T_1$ satellite image.
- `after.png`: Time $T_2$ satellite image.
- `ground_truth.png`: Ground-truth building change raster.
- `baseline_change_mask.png`: Classical pixel-difference change mask.
- `dl_predicted_mask.png`: Siamese ResNet-18 predicted change mask.
- `highlighted_changes.png`: Encroachment bounding box overlays on time $T_2$.
- `comparison_image.png`: 4-panel visual comparison board.
- `metrics.csv`: Performance metrics for the tested tile.
- `model_comparison.csv`: Direct tabular benchmark (Baseline vs. DL).
- `model_comparison_chart.png`: Visual comparison chart.
- `result.json`: Machine-readable summary of detection and screening status.

### 3.2 Single-Pair Detection Mode
Run change detection on arbitrary satellite image pairs:
```powershell
java -jar target/satellite-detector.jar --detect-change `
  --before data/processed/LEVIR-CD/test/A/test_100_tile_y0000_x0000.png `
  --after data/processed/LEVIR-CD/test/B/test_100_tile_y0000_x0000.png `
  --label data/processed/LEVIR-CD/test/label/test_100_tile_y0000_x0000.png `
  --threshold 35 `
  --min-region-area 30 `
  --output results/
```

### 3.3 Educational Module Flags
- `--fundamentals`: Runs Tensor arithmetic, SVD, Numerical Gradient Checker, McCulloch-Pitts gate, Sigmoid neuron, and XOR MLP.
- `--optimizers`: Benchmarks BGD, SGD, Momentum, and Nesterov on XOR and exports CSV logs.
- `--activations`: Evaluates Step, Sigmoid, Tanh, and ReLU across $[-3.0, 3.0]$.
- `--learning-rules`: Runs Hebbian, Oja's, Competitive, Delta, and Hopfield energy models.
- `--temporal`: Executes multi-year temporal land-use trend simulation.
- `--generative`: Executes educational latent space generative augmentation.

---

## 4. Maintenance & Scalability

- **Model Replacement:** The Siamese model is decoupled behind `OnnxResNetChangeDetector.java`. Any ONNX model accepting two $(1, 3, 256, 256)$ tensors and emitting a $(1, 1, 256, 256)$ change probability map can be substituted into `models/onnx/` without code changes.
- **Batch Processing:** The system processes tiles independently, allowing straightforward horizontal scaling across multiple worker processes or CPU threads.

