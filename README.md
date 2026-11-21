# Satellite Land Use Change and Encroachment Detector

[![Java](https://img.shields.io/badge/Java-21%20%7C%2025%20LTS-ED8B00?style=flat&logo=openjdk&logoColor=white)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.9.6-C71A36?style=flat&logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![JUnit 5](https://img.shields.io/badge/JUnit-5.10.2-25A162?style=flat&logo=junit5&logoColor=white)](https://junit.org/junit5/)
[![Domain](https://img.shields.io/badge/Domain-Geospatial%20Intelligence-blue)](https://en.wikipedia.org/wiki/Geospatial_intelligence)

An end-to-end Geospatial Machine Learning system built entirely in **Java** for automated monitoring of multitemporal satellite imagery, detection of land-use/land-cover (LULC) transitions, and alerting for unauthorized civil and industrial encroachment on protected environmental reserves and riparian buffer corridors.

---

## 1. Project Title & Industry Domain

* **Project Title:** Satellite Land Use Change and Encroachment Detector
* **Industry Domain:** Geospatial Intelligence (GEOINT) & Multitemporal Remote Sensing

---

## 2. Project Purpose

Protected natural ecosystems—such as state forest reserves, wetland conservation corridors, and riparian river buffers—face persistent threats from unauthorized human activities:
- **Illegal deforestation and clear-cutting** for unauthorized agriculture or plantations.
- **Unauthorized civil construction** and commercial real-estate expansion into demarcated public buffer zones.
- **Unregulated surface mining and excavation** causing severe soil degradation and watershed siltation.

This system provides an automated, reproducible Java pipeline that ingests multitemporal optical satellite imagery (e.g. Sentinel-2 at 10m Ground Sampling Distance or Landsat at 30m), preprocesses and co-registers rasters, performs semantic classification across 6 Land Use / Land Cover (LULC) classes, computes cross-temporal transition matrices, intersects changes against official GIS cadastral boundary polygons (`data/cadastral/`), and generates georeferenced encroachment alerts with affected area measurements (in $\text{m}^2$ and hectares).

---

## 3. Technology Stack

- **Core Language:** Java 21+ LTS (Tested on Oracle JDK 25)
- **Build System:** Apache Maven 3.9.6 (with portable Maven Wrapper `mvnw.cmd` and `mvnw`)
- **Testing Framework:** JUnit 5 (Jupiter 5.10.2)
- **Numerical & Matrix Computations:** Apache Commons Math 3.6.1
- **Logging:** SLF4J 2.0.12 with Logback Classic 1.5.3
- **Configuration & Serialization:** Jackson Databind 2.17.0
- *(Upcoming Stages)*: OpenCV Java, ONNX Runtime Java, XChart

---

## 4. Six-Stage Development Plan

| Stage | Name | Description | Status |
|:-----:|:-----|:------------|:------:|
| **Stage 1** | **Project foundation and Java Maven setup** | Maven build setup, modular package structure, lightweight basic dependencies, logging, unit tests, and CLI execution. | **Completed** |
| **Stage 2** | **Satellite-image preprocessing and dataset handling** | LEVIR-CD dataset inspection, A/B/label pairing, validation, RGB normalization, 256x256 tiling, summary generation (JSON/CSV), and sample visualizations. | **Completed** |
| **Stage 3** | **Change detection and encroachment detection** | Bi-temporal radiometric differencing, morphological noise removal, 8-connectivity component extraction, encroachment screening, IoU/F1 benchmarking, and multi-panel diagnostic visualizations. | **Completed** |
| **Stage 4** | **Neural-network fundamentals and learning algorithms** | Pure Java Tensor engine, broadcasting, matrix ops, SVD, autograd/backpropagation, MLP, optimizers (BGD, SGD, Momentum, Nesterov), and learning paradigms (Hebbian, Competitive, Boltzmann). | *Upcoming* |
| **Stage 5** | **Deep-learning, temporal, and generative model modules** | ResNet feature extraction, semantic segmentation (U-Net/FCN), object detection, temporal analysis (LSTM/Transformer), and generative data augmentation (VAE/GAN). | *Upcoming* |
| **Stage 6** | **Integration, evaluation, testing, documentation, and final packaging** | Metric evaluation (IoU, F1, Change Recall, Area Error), model & dataset cards, technical report, and production JAR distribution. | *Upcoming* |

---

## 5. Project Structure

```
Satellite_Land_Use_Change_Encroachment_Detector/
├── pom.xml                                      # Maven project configuration
├── mvnw.cmd / mvnw                              # Self-contained Maven wrapper scripts
├── .gitignore                                   # Version control exclusions
├── README.md                                    # Documentation and guide
├── configs/
│   └── pipeline-config.json                     # Geospatial CRS, classes, and alert rules
├── data/
│   ├── raw/                                     # Raw multitemporal satellite rasters
│   ├── processed/                               # Aligned and normalized tiles
│   ├── samples/                                 # Small synthetic/test image tiles
│   └── cadastral/                               # Vector boundary layers (GeoJSON)
├── models/
│   ├── onnx/                                    # Pretrained ONNX model assets
│   └── weights/                                 # Custom neural network weights
├── experiments/                                 # Metric logs, hyperparameter runs
├── visualizations/                              # Rendered change masks and plots
├── docs/                                        # Model cards, dataset cards, technical reports
└── src/
    ├── main/
    │   ├── java/com/satellite/
    │   │   ├── app/                             # Main application entry point & CLI
    │   │   ├── tensor/                          # N-D Tensor engine, matrix ops, SVD
    │   │   ├── neurons/                         # McCulloch-Pitts, Sigmoid, activation units
    │   │   ├── mlp/                             # Multilayer Perceptron, autograd/backprop
    │   │   ├── optimizers/                      # BGD, SGD, Momentum, Nesterov
    │   │   ├── learning/                        # Hebbian, Competitive, Boltzmann models
    │   │   ├── preprocessing/                   # Coregistration, tiling, normalization
    │   │   ├── classification/                  # ResNet feature extractor & classifier
    │   │   ├── segmentation/                    # Semantic segmentation engine
    │   │   ├── detection/                       # Object detection (structures & vehicles)
    │   │   ├── change/                          # Bi-temporal change detection & transition matrix
    │   │   ├── temporal/                        # Temporal sequence models (LSTM/Transformer)
    │   │   ├── generative/                      # Generative augmentation (VAE/GAN)
    │   │   ├── evaluation/                      # IoU, Macro-F1, Confusion Matrix, Area Error
    │   │   └── visualization/                   # Overlays, heatmaps, XChart plotting
    │   └── resources/
    │       ├── application.properties           # Application runtime configuration
    │       └── logback.xml                      # Logback structured logging configuration
    └── test/
        └── java/com/satellite/
            ├── app/                             # Startup and integration tests
            ├── tensor/                          # Tensor engine unit tests
            ├── neurons/                         # Neuron and activation tests
            ├── mlp/                             # Perceptron and backprop tests
            ├── optimizers/                      # Gradient optimizer tests
            ├── learning/                        # Learning algorithm tests
            ├── preprocessing/                   # Image preprocessing tests
            ├── classification/                  # Classifier tests
            ├── segmentation/                    # Segmentation model tests
            ├── detection/                       # Detection tests
            ├── change/                          # Change detection tests
            ├── temporal/                        # Temporal model tests
            ├── generative/                      # Generative model tests
            ├── evaluation/                      # Metric calculation tests
            └── visualization/                   # Visualization tests
```

---

## 6. Build, Test, and Execution Commands

### 1. Run JUnit 5 Unit Tests
```powershell
.\mvnw.cmd clean test
```
*(Or if system Maven is installed: `mvn clean test`)*

### 2. Build and Package Standalone Executable Uber-JAR
```powershell
.\mvnw.cmd clean package
```
*(Or `mvn clean package`)*

### 3. Run Application Entry Point
```powershell
# Using the packaged standalone Uber-JAR (includes all dependencies):
java -jar target/satellite-detector.jar

# Or via Maven exec plugin:
.\mvnw.cmd exec:java -Dexec.mainClass="com.satellite.app.Main"
```

### 4. Run LEVIR-CD Dataset Preprocessing, Validation & Summary Generation
```powershell
# Run validation, sample tiling (5 pairs/split), summary metrics, and side-by-side visualizations:
.\mvnw.cmd exec:java -Dexec.mainClass="com.satellite.preprocessing.DatasetLoader"

# Or to run full tiling across all 637 pairs:
.\mvnw.cmd exec:java -Dexec.mainClass="com.satellite.preprocessing.DatasetLoader" -Dexec.args="--full"

# Or execute directly with java:
java -cp target/satellite-detector.jar com.satellite.preprocessing.DatasetLoader
```

---

## 7. Stage 3: Change Detection and Encroachment Screening

Stage 3 provides an end-to-end, interpretable baseline pipeline for multitemporal change detection and automated encroachment screening directly in Java without external Python or native C++ runtimes.

### 1. Algorithmic Workflow
1. **Bi-Temporal Radiometric Differencing (`DifferenceCalculator`):** Computes pixel-wise Euclidean RGB distance:
   $$\Delta I(x, y) = \sqrt{\frac{1}{3} \left[ (R_B - R_A)^2 + (G_B - G_A)^2 + (B_B - B_A)^2 \right]}$$
   or absolute grayscale difference.
2. **Binary Thresholding (`ChangeMaskGenerator`):** Converts the continuous difference map into a preliminary binary change mask using configurable static thresholding (default: $35.0$) or adaptive Otsu thresholding.
3. **Morphological Noise Filtering (`NoiseRemover`):**
   - Applies morphological opening (erosion followed by dilation with a $3 \times 3$ kernel) to eliminate isolated pixel noise, sensor salt-and-pepper artifacts, and minor vegetation registration misalignments.
   - Applies morphological closing to bridge minor gaps in contiguous structures.
   - Filters out connected regions smaller than a configurable minimum pixel area threshold (default: $30$ pixels).
4. **Connected Component Analysis (`ConnectedComponentAnalyzer`):** Employs an 8-connectivity Breadth-First Search (BFS) to segment changed pixels into distinct spatial clusters, calculating for each cluster:
   - Axis-aligned bounding box ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$)
   - Cluster width, height, and area in pixels
   - Percentage of total image area
   - Cluster centroid coordinates $(\bar{x}, \bar{y})$
5. **Encroachment Screening Rules (`EncroachmentDetector`):** Classifies the observed change into one of three operational categories based on transparent, auditable thresholds:
   - `NO_SIGNIFICANT_CHANGE`: Total changed area $< 0.08\%$ ($< 52$ px for $256 \times 256$) and largest cluster $< 30$ px.
   - `POSSIBLE_CHANGE`: Minor or diffuse changes ($0.08\% \le \text{Area} < 0.40\%$).
   - `POSSIBLE_ENCROACHMENT`: Cohesive spatial cluster $\ge 150$ px or total change $\ge 0.40\%$ ($\ge 262$ px), indicating new structural development, clearing, or foundation work.
6. **Quantitative Benchmarking (`ChangeMetrics`):** Evaluates predicted binary masks against ground truth using:
   - Intersection over Union (IoU) / Jaccard Index
   - Precision and Recall (Change Recall)
   - F1-Score
   - Absolute Area Error in pixels
   - Inference runtime in milliseconds

### 2. Execution Commands

#### Run Automated Benchmark Demo (LEVIR-CD Processed Test Split):
```powershell
java -jar target/satellite-detector.jar --demo
```
This command automatically discovers all processed test tiles under `data/processed/LEVIR-CD/test/`, executes batch change detection and encroachment screening, computes global confusion metrics, saves diagnostic comparison boards for the first 6 samples to `visualizations/change-detection/`, and exports reports to `results/change-detection/`.

#### Run Single-Pair Change & Encroachment Detection:
```powershell
java -jar target/satellite-detector.jar --detect-change `
  --before data/processed/LEVIR-CD/test/A/test_100_tile_y0000_x0000.png `
  --after data/processed/LEVIR-CD/test/B/test_100_tile_y0000_x0000.png `
  --label data/processed/LEVIR-CD/test/label/test_100_tile_y0000_x0000.png `
  --threshold 40 `
  --min-region-area 50 `
  --output results/change-detection
```

### 3. Generated Artifacts & Reports
- `results/change-detection/change-results.json`: Full structured JSON containing every sample's bounding boxes, centroids, area metrics, and screening status.
- `results/change-detection/change-metrics.csv`: Tabular per-sample IoU, F1, precision, recall, area error, and latency.
- `results/change-detection/confusion-matrix.csv`: Aggregated pixel-level True Positives, True Negatives, False Positives, False Negatives, and mean scores.
- `results/change-detection/summary-report.md`: Executive markdown summary for stakeholders.
- `visualizations/change-detection/*_comparison_board.png`: 5-panel synchronized diagnostic boards showing:
  1. Earlier Satellite Image (Time A)
  2. Later Satellite Image (Time B)
  3. Ground-Truth Change Mask
  4. Predicted Binary Change Mask
  5. Annotated Bounding Boxes & Centroid Crosshairs with Screening Status Overlay.

### 4. Benchmark Results & Baseline Insights
On the 80 processed LEVIR-CD test tiles ($256 \times 256$ pixels):
- **Mean Change Recall:** **0.2728** (27.28% of actual structural changes captured)
- **Mean Precision:** **0.0901**
- **Mean F1-Score:** **0.1222**
- **Mean IoU:** **0.0790**
- **Total Batch Execution Time:** **568 ms** (~7.1 ms per tile)

> [!IMPORTANT]
> **Why Classical Radiometric Differencing Has High False Positives:**
> Pure pixel differencing operates solely on raw radiometric intensities. In multitemporal satellite imagery, seasonal sun-angle shifts, atmospheric scattering, shadows, and natural phenological changes (e.g., grass yellowing or soil moisture variation) produce significant radiometric differences without any structural change. This empirical result cleanly justifies the necessity of deep semantic representations (ResNet deep feature extraction, Siamese networks, and spatial attention mechanisms) being introduced in subsequent stages.

### 5. Regulatory & Operational Disclaimer
> [!NOTE]
> All encroachment classifications produced by this software (`POSSIBLE_ENCROACHMENT`, `POSSIBLE_CHANGE`) represent automated screening alerts based on bi-temporal raster analysis. They are intended solely for prioritizing investigative ground inspections and cadastral review. They do not constitute a legal or definitive determination of land ownership or property boundaries.

