# Dataset-Free Pre-Weld Seam Detection Pipeline

**Fuzzy Number Enhancement of Classical Laser Stripe Center Extraction**

A dataset-free, analytically grounded pipeline for pre-weld seam detection
using structured-light laser triangulation. The system compares **classical**
center extraction methods (Steger's algorithm + Center of Gravity) against
**fuzzy-enhanced** variants (Interval Type-2 Fuzzy Logic, Triangular Fuzzy
Numbers) to quantify improvements in accuracy and robustness on reflective
metallic surfaces.

**No neural networks or training data are required.**

---

## Architecture

```
Input Image (BGR)
    │
    ├── Classical ROI Extraction (Otsu + Connected Components)
    │       │
    │       ├── Path A: Classical (Baseline)
    │       │     ├── Steger 2D Ridge Detection → ridge points (x, y)
    │       │     └── Intensity-weighted Center of Gravity per column
    │       │
    │       └── Path B: Fuzzy-Enhanced (Novelty)
    │             ├── FCM Segmentation → membership map
    │             ├── Fuzzy Morphological Cleaning
    │             ├── Steger 2D → ridge points + normals n̂
    │             ├── Profile Sampling along n̂ (bilinear interpolation)
    │             ├── IT2FLS (EKM) → [y_l, y_c, y_r] interval
    │             │   ├── Fallback: Fuzzy Barycentric (saturated profiles)
    │             │   └── Fallback: TFN (noisy profiles)
    │             └── Uncertainty propagation
    │
    ├── Post-Processing (Savitzky-Golay smoothing, outlier removal)
    │
    ├── 3D Triangulation (ray-plane intersection)
    │     ├── Crisp 3D points (classical path)
    │     └── 3D confidence intervals (fuzzy path)
    │
    └── Seam Feature Extraction + Robot Coordinate Export
```

## Hardware Setup

| Component        | Specification                  |
|------------------|--------------------------------|
| **Laser**        | 650 nm red line laser, 50 mW   |
| **Camera**       | 1920×1200, Sony IMX174 sensor  |
| **Band-pass**    | 650 ± 10 nm                    |
| **Triangulation**| 30° angle, 100 mm baseline     |
| **Working dist.**| 200 mm                         |

## Key Algorithms

### Steger's Algorithm [1]
Sub-pixel ridge detection using the 2D Hessian matrix. At each pixel:
- Build H = [[Ixx, Ixy], [Ixy, Iyy]] from Gaussian derivatives
- Find eigenvector (nx, ny) of the largest-magnitude eigenvalue
- Compute sub-pixel offset t = -(nx·Ix + ny·Iy) / (nx²·Ixx + 2·nx·ny·Ixy + ny²·Iyy)
- Accept if |t| < 0.5 and eigenvalue exceeds threshold

### IT2FLS with Enhanced Karnik-Mendel [2]
Interval Type-2 Fuzzy Logic System operating on cross-ridge intensity profiles:
- **UMF**: μ̄(I) = 1 − (I/I_sat)^k — monotonically decreasing with intensity
- **LMF**: μ(I) = μ̄(I) for I < I_sat, 0 otherwise
- **EKM algorithm** computes left/right centroid bounds [y_l, y_r]
- **Defuzzified center**: y_c = (y_l + y_r) / 2
- **Uncertainty interval**: |y_r − y_l| quantifies positional confidence

### Fuzzy Gray Barycentric [3]
Fuzzy membership-weighted barycentric center for saturated/specular profiles:
- Piecewise membership: ramp → plateau → Gaussian decay
- Pal-King intensification operator for membership contrast
- y_center = Σ(y_i · μ_i · I_i) / Σ(μ_i · I_i)

### Triangular Fuzzy Numbers (TFN)
For noisy/low-SNR profiles, models each pixel intensity as Ĩ = (I−δ, I, I+δ):
- δ = local_std + k·exp(−(I_mean − I_sat)²/(2τ²)) (saturation penalty)
- Computes three barycenters (pessimistic/central/optimistic)
- Defuzzified center: y = (y_low + 2·y_best + y_high) / 4

## Project Structure

```
Project/
├── config.py                     # Centralized configuration (no DL)
├── run.py                        # CLI entry point
├── evaluate.py                   # Classical vs. fuzzy comparison
├── utils.py                      # Metrics, visualization helpers
├── requirements.txt              # Pure classical dependencies
│
├── preprocessing/
│   └── roi_extractor.py          # Otsu + connected-component ROI
│
├── extraction/
│   ├── steger.py                 # Hessian-based sub-pixel ridge detection
│   ├── profile_sampler.py        # Bilinear profile sampling along normals
│   ├── adaptive_extractor.py     # Adaptive method selection
│   └── profile_smoother.py       # Savitzky-Golay post-processing
│
├── fuzzy/
│   ├── fuzzy_pipeline.py         # Dual-path orchestrator
│   ├── fcm_segmentation.py       # Fuzzy C-Means clustering
│   ├── fuzzy_morphology.py       # Fuzzy opening/closing
│   ├── fuzzy_barycentric.py      # Fuzzy gray barycentric extraction
│   ├── type2_barycentric.py      # IT2FLS + EKM centroid
│   └── tfn_extraction.py         # TFN-based uncertainty extraction
│
├── geometry/
│   ├── triangulation.py          # Ray-plane 3D reconstruction
│   ├── seam_features.py          # Geometric feature extraction
│   └── coordinate_chain.py       # Camera → Robot transformation
│
└── pipeline/
    ├── inference.py              # End-to-end detector
    ├── visualizer.py             # Overlay and profile plots
    └── exporter.py               # JSON/CSV coordinate export
```

## Usage

### Quick Start
```bash
pip install -r requirements.txt

# Fuzzy-enhanced mode (default)
python run.py --input image.png --output results/

# Classical mode
python run.py --input image.png --output results/ --mode classical

# Side-by-side comparison
python run.py --input image.png --output results/ --mode compare

# Batch processing
python run.py --input data/test/ --output results/ --format csv
```

### Synthetic Benchmark
```bash
python evaluate.py --synthetic --output eval_results/
```

### Real Image Evaluation
```bash
python evaluate.py --image_dir data/test/images \
                   --mask_dir  data/test/masks  \
                   --output eval_results/
```

## Dependencies

- **opencv-python** ≥ 4.8 — image I/O, Gaussian derivatives, morphology
- **numpy** ≥ 1.24 — vectorized array operations
- **scipy** ≥ 1.11 — Savitzky-Golay filtering
- **scikit-image** ≥ 0.21 — image processing utilities
- **scikit-learn** ≥ 1.3 — evaluation metrics
- **matplotlib** ≥ 3.7 — visualization

No PyTorch, TensorFlow, or any deep learning framework is required.

## References

1. Steger, C. (1998). "An unbiased detector of curvilinear structures."
   *IEEE TPAMI*, 20(2), 113-125.
2. Wu, D. & Mendel, J.M. (2009). "Enhanced Karnik-Mendel algorithms."
   *IEEE Trans. Fuzzy Syst.*, 17(4), 923-934.
3. Chaira, T. & Ray, A.K. (2010). *Fuzzy Image Processing and Applications
   with MATLAB*. CRC Press.
4. Dubois, D. & Prade, H. (1980). *Fuzzy Sets and Systems: Theory and
   Applications*. Academic Press.
5. Pal, S.K. & King, R.A. (1981). "Image enhancement using smoothing with
   fuzzy sets." *IEEE Trans. SMC*, 11(7), 494-501.
