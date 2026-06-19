"""
Centralized Configuration for Offline Pre-Weld Scan Pipeline.

All hyperparameters for hardware, fuzzy logic, IT2FLS, calibration,
post-processing, and paths are defined here using frozen-friendly
dataclasses for type safety and IDE auto-completion.

This pipeline is designed for **offline, pre-weld scanning** where
latency requirements are relaxed (500 ms – 1 s per frame is
acceptable).  The system acquires and stores full scans of the
workpiece seam geometry before welding commences.

This is a **dataset-free, no-neural-network** pipeline.  All
processing is analytically grounded using Steger's algorithm,
Interval Type-2 Fuzzy Logic (IT2FLS), and classical image processing.

References:
    [1] Steger, C. (1998).
        "An unbiased detector of curvilinear structures."
        IEEE TPAMI, 20(2), pp. 113-125.
    [2] Wu, D. & Mendel, J.M. (2009).
        "Enhanced Karnik-Mendel algorithms."
        IEEE Trans. Fuzzy Syst., 17(4), pp. 923-934.
    [3] Chaira, T. & Ray, A.K. (2010).
        "Fuzzy Image Processing and Applications with MATLAB."
        CRC Press.
    [4] Zhang, Z. (2000).
        "A Flexible New Technique for Camera Calibration."
        IEEE TPAMI, 22(11), pp. 1330-1334.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@dataclass
class HardwareConfig:
    """Laser triangulation hardware parameters.

    Default values correspond to a typical industrial setup with a
    Sony IMX174 sensor and a 650 nm red-line laser.  Parameters are
    used for offline pre-weld scan geometry reconstruction.
    """

    laser_wavelength_nm: int = 650          # Red laser diode
    laser_power_mw: float = 50.0
    camera_resolution: Tuple[int, int] = (1920, 1200)
    roi_height: int = 200                   # Camera ROI rows
    sensor_pixel_size_um: float = 3.45      # Sony IMX174

    # Laser line width in pixels (optical constant, measured during setup)
    # For 2390x1792 images with typical working distance, the laser
    # stripe is ~10-15 px wide; 12 is a good default.
    w_pixels: int = 12

    @property
    def sigma(self) -> float:
        """Steger's Gaussian scale derived from physical line width.

        σ = w / (2√3), the correct value from Steger (1998) [1].
        """
        return self.w_pixels / (2 * np.sqrt(3))

    focal_length_mm: float = 16.0
    baseline_mm: float = 100.0
    triangulation_angle_deg: float = 30.0
    working_distance_mm: float = 200.0
    bandpass_filter_nm: float = 10.0        # FWHM of optical band-pass

    # --- Depth confidence model ---
    depth_confidence_z_ref_mm: float = 200.0   # Reference depth for calibration baseline
    depth_confidence_delta_p_px: float = 0.1   # Sub-pixel detection uncertainty (px)


# ---------------------------------------------------------------------------
# Fuzzy Logic (Unified — covers FCM, morphology, barycentric, TFN)
# ---------------------------------------------------------------------------
@dataclass
class FuzzyConfig:
    """Fuzzy Number Algorithm parameters [3].

    Covers Fuzzy C-Means clustering, fuzzy gray barycentric extraction,
    fuzzy morphology, and Triangular Fuzzy Number (TFN) modelling.
    """

    # --- FCM Clustering ---
    fcm_clusters: int = 3                   # background / laser / specular
    fcm_fuzziness: float = 2.0              # m parameter
    fcm_max_iter: int = 100
    fcm_tolerance: float = 1e-5

    # --- Fuzzy Gray Barycentric ---
    noise_floor_percentile: float = 10.0    # T_low auto-estimation
    saturation_threshold: int = 240         # I_sat for 8-bit images
    specular_penalty_sigma: float = 10.0    # σ for exponential decay
    intensification_iterations: int = 2

    # --- Fuzzy Morphology ---
    morph_kernel_size: int = 7
    morph_kernel: List[float] = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 1.0, 1.0, 0.5, 0.2]
    )

    # --- TFN Extraction ---
    tfn_neighborhood_size: int = 5          # Window for local noise est.

    # Saturation penalty for TFN uncertainty bounds.
    # tau prevents division-by-zero on perfectly saturated blocks;
    # k is the amplitude multiplier for the exponential penalty term.
    saturation_penalty_tau: float = 15.0
    saturation_penalty_k: float = 5.0

    # --- Adaptive Selection Thresholds ---
    flatness_threshold: float = 0.3         # Profile flatness ratio
    snr_threshold: float = 5.0              # Minimum SNR for Steger's
    uncertainty_speed_threshold: float = 2.0  # px — flag for review if exceeded

    # --- Stage toggles (for ablation studies) ---
    enable_fcm: bool = True
    enable_morphology: bool = True
    enable_fuzzy_barycentric: bool = True
    enable_tfn: bool = True


# ---------------------------------------------------------------------------
# IT2FLS (Interval Type-2 Fuzzy Logic System)
# ---------------------------------------------------------------------------
@dataclass
class IT2FLSConfig:
    """Enhanced Karnik-Mendel (EKM) algorithm parameters [2].

    The IT2FLS operates in the spatial domain along the cross-ridge
    normal direction.  Membership functions are monotonically decreasing
    in intensity — maximum weight for reliable low-intensity shoulder
    pixels, zero weight for saturated plateau pixels.
    """

    k_exponent: float = 2.0                 # Exponent for UMF power law
    epsilon: float = 0.01                   # EKM convergence tolerance
    max_iter: int = 100                     # EKM maximum iterations
    i_sat_percentile: float = 95.0          # Per-profile dynamic I_sat

    # Profile sampling parameters (along Steger normal)
    profile_half_width: int = 15            # Samples on each side of ridge
    profile_sample_step: float = 0.5        # Sub-pixel step size along n̂


# ---------------------------------------------------------------------------
# Classical Extraction
# ---------------------------------------------------------------------------
@dataclass
class ClassicalConfig:
    """Parameters for classical (non-fuzzy) extraction methods."""

    # Steger ridge detection
    # low_thresh must be high enough to reject wood grain, shadows, etc.
    # A value of 2.0 keeps only strong laser ridges.
    steger_low_thresh: float = 2.0          # Eigenvalue threshold
    steger_high_thresh: float = 10.0        # Reserved for hysteresis

    # Center of gravity (CoG)
    cog_noise_floor_percentile: float = 10.0


# ---------------------------------------------------------------------------
# ROI Extraction (Classical — no neural network)
# ---------------------------------------------------------------------------
@dataclass
class ROIConfig:
    """Classical ROI extraction parameters (replaces DL-based ROI).

    Uses chromaticity-based extraction (red dominance + HSV hue) instead
    of raw red-channel Otsu, which fails on warm-toned backgrounds.
    """

    blur_sigma: float = 3.0                 # Pre-processing Gaussian blur σ
    min_stripe_area: int = 100              # Min connected-component pixels
    roi_margin: int = 30                    # Padding around detected stripe
    red_channel_weight: float = 1.0         # Weight for red channel (red laser)

    # HSV thresholds for laser hue filtering (supplements red-dominance)
    hsv_sat_min: int = 50                   # Min saturation (0-255)
    hsv_val_min: int = 100                  # Min value/brightness (0-255)


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------
@dataclass
class PostProcessConfig:
    """Post-processing parameters for centre-line refinement."""

    threshold: float = 0.5                  # Segmentation binarisation
    min_stripe_length: int = 50             # Min connected-component px
    smoothing_window: int = 11              # Savitzky-Golay window
    smoothing_order: int = 3                # Savitzky-Golay poly order
    outlier_z_threshold: float = 3.0        # Z-score for outlier removal
    ransac_residual_threshold: float = 0.1  # mm


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    """Camera & laser-plane calibration parameters."""

    checkerboard_size: Tuple[int, int] = (8, 6)   # Inner corners
    square_size_mm: float = 25.0

    # From stage1 calib/camera.txt
    camera_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [369.35006688,   0.        , 337.96248307],
        [  0.        , 369.12689097, 249.43323907],
        [  0.        ,   0.        ,   1.        ]
    ], dtype=np.float64))

    dist_coeffs: np.ndarray = field(default_factory=lambda: np.array(
        [[-0.20266307, -0.05435428, 0.00121721, 0.00039301, 0.10316378]],
        dtype=np.float64
    ))

    # From stage1 calib/laser.txt (Ax + By + Cz + D = 0)
    laser_plane: np.ndarray = field(default_factory=lambda: np.array(
        [-0.193309, -0.080403, -0.977838, 174.601821],
        dtype=np.float64
    ))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    """Project directory layout.

    Heavy directories (data, checkpoints) are derived via ``@property``
    to keep serialisation lightweight.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / 'data'

    @property
    def calibration_dir(self) -> Path:
        return self.data_dir / 'calibration'

    @property
    def test_dir(self) -> Path:
        return self.data_dir / 'test'

    @property
    def output_dir(self) -> Path:
        return self.project_root / 'output'

    @property
    def log_dir(self) -> Path:
        return self.project_root / 'logs'


# ---------------------------------------------------------------------------
# Boundary Localization (Stage 6.5)
# ---------------------------------------------------------------------------
@dataclass
class BoundaryConfig:
    """Parameters for Stage 6.5 along-seam boundary-aware truncation.

    Two Gaussian smoothing scales are used — keep them distinct:

    * ``steger_sigma`` (Optional[float]):  per-profile LoG smoothing scale
      for the cross-ridge curvature cue (cue 4).  ``None`` means inherit
      ``HardwareConfig.sigma`` at runtime.  Use ``None`` — **not** ``0.0`` —
      to request the inherited value; ``0.0`` would skip smoothing entirely.

    * ``edge_smooth_sigma`` (float):  Gaussian applied to the along-seam
      ``mu_edge`` vector before run-length detection.  Operates along the
      seam axis (1-D), not across the laser stripe cross-section.

    OWA weight semantics
    --------------------
    ``owa_rank_weights[k]`` is applied to the *k-th largest* cue value at
    each column — NOT to a specific named cue.  This is Yager (1988) OWA.
    Do **not** assume ``owa_rank_weights[0]`` always maps to ``e_strength``;
    it maps to whichever cue is largest at that column.

    References
    ----------
    [5] Yager, R.R. (1988). On ordered weighted averaging aggregation
        operators in multicriteria decisionmaking. IEEE Trans. Syst. Man
        Cybern., 18(1), 183-190.
    [6] Marr, D. & Hildreth, E. (1980). Theory of edge detection.
        Proc. R. Soc. Lond. B, 207(1167), 187-217.
    [7] Canny, J. (1986). A computational approach to edge detection.
        IEEE TPAMI, 8(6), 679-698.
    """

    # --- Per-profile curvature cue (LoG, cue 4) ---
    steger_sigma: Optional[float] = None
    """Gaussian sigma for profile second-derivative (LoG) curvature cue.
    None → inherit HardwareConfig.sigma at runtime."""

    zc_ref: float = 2.0
    """Extra zero-crossings above 2 that saturate e_curvature to 1.0."""

    zc_eta: float = 0.5
    """Magnitude gate for zero-crossing counting: a crossing is counted
    only if |d2p| > zc_eta * std(d2p) on at least one side of the
    crossing.  Prevents noise-floor sign flips from being counted."""

    # --- MAD normalization (cues 1 & 2) ---
    z_ref: float = 3.0
    """Clamp scale for MAD-normalized z-scores; values above this map to 1.0."""

    mad_central_fraction: float = 0.80
    """Fraction of columns used to compute the MAD baseline (central
    fraction, excluding the tails which may contain edge columns)."""

    # --- OWA fusion (cue aggregation) ---
    owa_rank_weights: List[float] = field(
        default_factory=lambda: [0.35, 0.25, 0.20, 0.20]
    )
    """OWA rank-weight vector.  weight[k] applies to the k-th LARGEST
    cue value at each column (Yager 1988 convention)."""

    single_cue_damp: float = 0.5
    """Multiplier applied to mu_edge[i] when fewer than 2 cues exceed
    cue_elevation_floor — guards against single-cue false triggers."""

    cue_elevation_floor: float = 0.4
    """A cue value > this floor counts as 'elevated' for the damp test."""

    # --- Boundary run-length detection ---
    edge_smooth_sigma: float = 2.0
    """Gaussian sigma for smoothing mu_edge along the seam axis before
    run-length detection.  Different purpose from steger_sigma."""

    edge_mu_thresh: float = 0.55
    """Membership confidence floor for the run-length edge detector."""

    run_min: int = 3
    """Minimum consecutive above-threshold columns required to declare
    an off-workpiece edge run."""

    # --- Fallback guard ---
    min_seam_fraction: float = 0.50
    """If the kept fraction of columns falls below this, fall back to the
    full range and emit a structured warning with mu_edge diagnostics."""

    # --- Post-triangulation depth-curvature refinement (Section 3.6) ---
    enable_depth_refine: bool = True
    """Enable the optional Z-curvature trim pass after triangulation."""

    kappa_thresh: float = 2.0
    """MAD-normalized curvature z-score threshold for depth-refine trimming."""

    max_refine_steps: int = 10
    """Maximum columns to trim per side in the depth-refine pass."""


# ---------------------------------------------------------------------------
# Master Pipeline Config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs.

    Designed for offline pre-weld scanning; per-frame latency of
    500 ms – 1 s is acceptable.

    No neural-network dependencies.  All processing is CPU-based.

    Instantiate without arguments for sensible defaults::

        cfg = PipelineConfig()
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    it2fls: IT2FLSConfig = field(default_factory=IT2FLSConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    seed: int = 42
    processing_mode: str = 'pre_weld_scan'  # Offline scan mode


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig()
