# Dataset Card: LEVIR-CD (LEVIR Change Detection)

## 1. Dataset Overview

* **Dataset Name:** LEVIR-CD (LEVIR Building Change Detection Dataset)
* **Domain:** Geospatial Intelligence & Multitemporal Remote Sensing
* **Primary Reference:** *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection* (Chen & Shi, IEEE TGRS 2020)
* **Originating Institution:** Image Processing Center, Beihang University (BUAA), Beijing, China
* **Dataset Location in Repository:** `data/LEVIR-CD/`
* **Processed Location:** `data/processed/LEVIR-CD/`

---

## 2. Dataset Purpose & Semantic Modalities

LEVIR-CD is a large-scale, very-high-resolution (VHR) bi-temporal remote sensing benchmark dataset designed for evaluating change detection algorithms. Each sample represents an identical geographic parcel imaged at two distinct temporal epochs:

| Subfolder | Modality Name | Time Step | Representation |
|:---|:---|:---:|:---|
| **`A/`** | Earlier Satellite / Aerial Image | $T_1$ (Baseline) | 3-channel optical RGB image before construction or expansion |
| **`B/`** | Later Satellite / Aerial Image | $T_2$ (Current) | 3-channel optical RGB image after construction or expansion |
| **`label/`** | Ground Truth Change Mask | Bi-temporal Delta | Binary mask where `0` = Unchanged / Background, `255` (or `1`) = Structural Building Change |

---

## 3. Split Distribution & Catalog Counts

The dataset consists of **637** very-high-resolution bi-temporal image pairs (totaling **1,911** image rasters) partitioned into standardized benchmark splits:

| Dataset Split | Number of Pairs | Images per Split ($A + B + \text{label}$) | Native Dimensions | Spatial Resolution (GSD) |
|:---|:---:|:---:|:---:|:---:|
| **Train** | 445 pairs | 1,335 images | $1024 \times 1024$ | 0.5 m / pixel |
| **Validation** | 64 pairs | 192 images | $1024 \times 1024$ | 0.5 m / pixel |
| **Test** | 128 pairs | 384 images | $1024 \times 1024$ | 0.5 m / pixel |
| **Total** | **637 pairs** | **1,911 images** | **$1024 \times 1024$** | **0.5 m / pixel** |

### Archive & Extraction Status
* `train.zip`: ~1.72 GB (Extracted in `data/LEVIR-CD/train/` with `A`, `B`, `label`)
* `val.zip`: ~246 MB (Extracted in `data/LEVIR-CD/val/` with `A`, `B`, `label`)
* `test.zip`: ~496 MB (Extracted in `data/LEVIR-CD/test/` with `A`, `B`, `label`)
* *All three splits are verified extracted and ready on disk.*

---

## 4. Preprocessing & Tiling Strategy

1. **Deterministic Pairing:** Filename-based stem matching connects each $T_1$ image (`train_1.png`) to its corresponding $T_2$ image and binary label mask.
2. **Integrity Validation:** Every raster is checked for file readability, corruptions, dimension matching ($W_A = W_B = W_{\text{label}}$, $H_A = H_B = H_{\text{label}}$), and binary pixel distributions.
3. **RGB Normalization:** Pixel values $[0, 255]$ are scaled to floating-point $[0.0, 1.0]$.
4. **Binary Mask Normalization:** Mask pixels are validated and binarized ($r > 127 \rightarrow 1$, else $0$).
5. **Sliding-Window $256 \times 256$ Tiling:**
   * Each $1024 \times 1024$ scene is tiled into 16 non-overlapping sub-tiles ($4 \times 4$ grid, stride 256) of size $256 \times 256$.
   * Tiled assets are stored under `data/processed/LEVIR-CD/{split}/A/`, `.../B/`, and `.../label/` with geocoded grid suffixes (e.g. `train_1_tile_y0000_x0000.png`).
6. **Comparison Visualizations:**
   * 4-panel composite diagnostic charts ($T_1$ Earlier, $T_2$ Later, Ground Truth Mask, and Encroachment Overlay) are saved to `visualizations/dataset/`.

---

## 5. Domain Suitability: Building Change & Encroachment Detection

LEVIR-CD is well-suited for **unauthorized civil and structural encroachment detection** because:
* **High Spatial Resolution (0.5 m/pixel):** Sub-meter resolution clearly resolves individual residential villas, warehouses, multi-story buildings, garages, and civil structures.
* **Complex Environmental Backgrounds:** Covers varied land types including agricultural margins, hillside topography, urban-rural fringes, and riparian clearings.
* **Temporal Span:** The bi-temporal pairs span 5 to 14 years across various cities in Texas, USA, capturing significant real-world building growth and site transformations.

---

## 6. Dataset Limitations & Non-Claims

> [!IMPORTANT]
> **LEVIR-CD primarily represents building-related changes.** It does **NOT** capture every type of land-use / land-cover (LULC) transition:
> - **Vegetation and Deforestation:** While surrounding vegetation changes over seasons, LEVIR-CD annotations strictly label *building structures*, not agricultural crop phenology or non-structural tree felling.
> - **Hydrology and Water Corridors:** Natural shoreline shifts, river erosion, or wetland expansions are unannotated unless accompanied by civil building activity.
> - **Sensor Homogeneity:** Imagery is derived from high-resolution aerial orthophotography rather than multispectral satellite bands (e.g., lacks Shortwave Infrared / SWIR or red-edge bands found on Sentinel-2).
> - **Extreme Class Imbalance:** Building change pixels occupy only ~1.2% to 4.5% of total scene pixels, requiring change-detection algorithms that handle high foreground-background imbalance.

---

## 7. License & Academic Restrictions

* **Intended Use:** Non-commercial academic research and educational development.
* **Attribution:** Any publication or derived model utilizing this dataset must cite the original authors (Hao Chen and Zhenwei Shi, *IEEE Transactions on Geoscience and Remote Sensing*, 2020).
* **Redistribution:** The original archives (`train.zip`, `val.zip`, `test.zip`) remain preserved intact in `data/LEVIR-CD/` without deletion or in-place modification.
