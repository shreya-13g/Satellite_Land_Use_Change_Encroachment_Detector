# GeoShield — Satellite Land Use Change & Encroachment Detector — User Guide

This user guide provides step-by-step instructions for setting up, building, running, and demonstrating the **GeoShield** system.

---

## 1. System Requirements

- **Operating System:** Windows 10/11, Linux, or macOS
- **Java Development Kit:** JDK 21 or JDK 25 (64-bit)
- **Python:** Python 3.10+ (for Streamlit dashboard)
- **Maven:** Bundled via `./mvnw.cmd` (Windows) or `./mvnw` (Linux/macOS)
- **Memory:** Minimum 4 GB RAM (8 GB recommended)

---

## 2. Building the Project

### Windows (Command Prompt / PowerShell)
```powershell
# Compile and run automated JUnit 5 unit tests
.\mvnw.cmd test

# Build executable Uber-JAR (all dependencies shaded)
.\mvnw.cmd package -DskipTests
```

### macOS / Linux
```bash
# Compile and run automated JUnit 5 unit tests
./mvnw test

# Build executable Uber-JAR
./mvnw package -DskipTests
```

The resulting executable artifact is created at:
`target/satellite-detector.jar`

---

## 3. Running the Graphical GeoShield Dashboard

To launch the interactive web dashboard for demonstrations and visual evaluation:

```bash
# 1. (Recommended) Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dashboard dependencies (Streamlit, Pillow, Pandas)
pip install -r dashboard/requirements.txt

# 3. Start the GeoShield application
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`.

### Dashboard Features:
1. **Interactive Detection Tab:**
   - Select built-in LEVIR-CD benchmark samples or upload your own pre/post satellite images.
   - Select algorithm (Dual Baseline & Siamese ResNet-18, Baseline only, or DL only).
   - Toggle Otsu adaptive thresholding or set manual radiometric thresholds.
   - Click **Run Change Detection** to execute the Java ONNX inference pipeline.
   - View Before (T1), After (T2), Predicted Masks, and Bounding Box clusters.
   - View real-time KPIs: Screening status, Changed Area (px and %), IoU, F1, Recall.
   - **Export Suite:** One-click downloads for Change Mask (PNG), Bounding Boxes (PNG), Confusion Overlay (PNG), and formal Screening Inspection Report (JSON).
2. **Model Benchmarks Tab:**
   - Head-to-head empirical metrics table (Precision, Recall, F1, IoU, Runtime) with CSV download.
   - Visual diagnostic comparison chart and 2x2 confusion matrix heatmap.
   - Candidate architectural taxonomy (R-CNN, YOLO, RetinaNet, GoogLeNet).
3. **Fundamentals & Ablations Tab:**
   - 4-Optimizer convergence curves (BGD, SGD, Momentum, Nesterov).
   - Overfitting vs generalization study curve (Train vs Validation loss) with CSV export.
   - Full hyperparameter ablation matrix with CSV export.
4. **Temporal LSTM & Generative Models Tab:**
   - 2D/3D Convolution feature maps.
   - Multitemporal sequence forecasting using gated LSTM cell and Transformer self-attention with CSV export.
   - VAE latent-space organization scatter plot and synthesized augmented patches.
5. **Robustness & Error Analysis Tab:**
   - Noise, illumination, contrast, and spatial misregistration stress test results with CSV export.
   - Detailed failure mode analysis and responsible-use limits.

---

## 4. Command-Line Interface (CLI) Usage

The application can also be executed completely via terminal flags:

### A. Run Automated Benchmark Demonstration
```powershell
java -jar target/satellite-detector.jar --demo
```
Runs both classical baseline and Siamese ResNet-18 on test tiles, outputs comparison CSVs and rendered charts in `results/`.

### B. Single-Pair Change Detection
```powershell
java -jar target/satellite-detector.jar --detect-change \
  --before data/processed/LEVIR-CD/test/A/test_0001_x0256_y0256.png \
  --after data/processed/LEVIR-CD/test/B/test_0001_x0256_y0256.png \
  --label data/processed/LEVIR-CD/test/label/test_0001_x0256_y0256.png \
  --otsu \
  --output results/my-run
```

### C. Neural Network Fundamentals & Learning Rules
```powershell
# Run basic tensor operations, McCulloch-Pitts gate, Sigmoid unit, SVD & Gradient Check
java -jar target/satellite-detector.jar --fundamentals

# Run 4-optimizer benchmark comparison
java -jar target/satellite-detector.jar --optimizers

# Export activation functions comparison
java -jar target/satellite-detector.jar --activations

# Run 5 educational learning rules (Hebbian, Oja, Competitive, Delta, Hopfield)
java -jar target/satellite-detector.jar --learning-rules
```

### D. Ablations & Overfitting Study (Requirement 5)
```powershell
java -jar target/satellite-detector.jar --ablations
```
Executes ablation experiments across learning rates, initializations, normalizations, regularizations, activations, and optimizers, and runs train vs validation overfitting divergence analysis.

### E. 2D/3D Convolution Operations Visualization (Requirement 7)
```powershell
java -jar target/satellite-detector.jar --convolution
```
Renders spatial feature maps for Sobel, Laplacian, ReLU, blur, sharpen, and 3D spatio-temporal convolution to `visualizations/convolution_feature_maps.png`.

### F. Temporal Sequence LSTM & Self-Attention (Requirement 9)
```powershell
java -jar target/satellite-detector.jar --temporal
```
Executes sequence analysis using the gated `LSTMCell` and `SimpleAttentionDemo` scaled dot-product attention, saving predictions to `experiments/temporal_forecast.csv`.

### G. Generative VAE/GAN Augmentation (Requirement 9)
```powershell
java -jar target/satellite-detector.jar --generative
```
Demonstrates VAE reparameterization, runs a minimax GAN adversarial training loop, and renders 2D latent space organization to `experiments/latent_space_visualization.png`.

### H. Robustness Stress Testing & Error Analysis (Requirement 10)
```powershell
java -jar target/satellite-detector.jar --robustness
```
Evaluates pipeline stability under sensor noise, illumination shifts, contrast drops, and spatial misregistration, exporting markdown report to `experiments/robustness_report.md`.
