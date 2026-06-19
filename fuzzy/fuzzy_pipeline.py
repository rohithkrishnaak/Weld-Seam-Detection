"""
Dual-Path Fuzzy Pipeline — Classical vs. IT2FLS-Enhanced Center Extraction.

Implements two parallel extraction paths for laser stripe center detection:

  **Path A — Classical (baseline)**:
      Steger 2D ridge detection + intensity-weighted Center of Gravity

  **Path B — Fuzzy-Enhanced (novelty)**:
      FCM segmentation → Fuzzy morphological cleaning →
      Steger 2D (ridge + normals) → Profile sampling along normals →
      IT2FLS (EKM) center + uncertainty interval →
      Fallback to Fuzzy Barycentric / TFN for degenerate profiles

The FuzzyResult dataclass carries both sets of centers so that
downstream modules can compare accuracy and analyze improvements.

References
----------
[1] Steger, C. (1998). An unbiased detector of curvilinear structures.
    IEEE TPAMI, 20(2), 113-125.
[2] Wu, D. & Mendel, J.M. (2009). Enhanced Karnik-Mendel algorithms.
    IEEE Trans. Fuzzy Syst., 17(4), 923-934.
[3] Chaira, T. & Ray, A.K. (2010). Fuzzy Image Processing and
    Applications with MATLAB. CRC Press.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from config import FuzzyConfig, IT2FLSConfig, HardwareConfig, ClassicalConfig
from fuzzy.fcm_segmentation import FuzzyCMeans
from fuzzy.fuzzy_morphology import FuzzyMorphology
from fuzzy.fuzzy_barycentric import FuzzyGrayBarycentric
from fuzzy.tfn_extraction import TFNCenterExtractor
from fuzzy.type2_barycentric import IT2FLSExtractor
from extraction.steger import StegerExtractor
from extraction.profile_sampler import ProfileSampler

logger = logging.getLogger(__name__)


# ======================================================================
# Result dataclass
# ======================================================================
@dataclass
class FuzzyResult:
    """Container for the outputs of the dual-path fuzzy pipeline.

    Carries both classical and fuzzy-enhanced center coordinates so
    that downstream modules can compare accuracy and uncertainty.

    Attributes
    ----------
    x_coords : np.ndarray
        Column positions of detected seam points.
    y_centers : np.ndarray
        Sub-pixel Y centers from the fuzzy-enhanced path (IT2FLS / EKM).
    y_centers_classical : np.ndarray
        Sub-pixel Y centers from the classical path (CoG).
    confidences : np.ndarray
        Per-point confidence from the fuzzy-enhanced extraction.
    uncertainties : np.ndarray
        IT2FLS interval width |y_r − y_l| per point.
    y_lower : np.ndarray
        EKM left (lower) bounds.
    y_upper : np.ndarray
        EKM right (upper) bounds.
    methods_used : List[str]
        Per-point method label ('it2fls', 'barycentric', 'tfn', 'cog').
    membership_map : np.ndarray
        FCM laser membership (H, W), values in [0, 1].
    normals_x : np.ndarray
        Steger normal x-components at each detected point.
    normals_y : np.ndarray
        Steger normal y-components at each detected point.
    """

    x_coords: np.ndarray
    y_centers: np.ndarray
    y_centers_classical: np.ndarray
    confidences: np.ndarray
    uncertainties: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    methods_used: List[str]
    membership_map: np.ndarray
    normals_x: np.ndarray
    normals_y: np.ndarray

    # --- Stage 6.5 inputs (already computed in process(), now exposed) ---
    strengths: np.ndarray
    """Per-selected-column Steger ridge strength |eigenval_max|, shape (N,).
    Same index correspondence as x_coords.  Used as cue 1 in Stage 6.5."""

    profiles: np.ndarray
    """Cross-ridge intensity profiles, shape (N, M).  Sliceable along axis 0
    (per column) in truncate_to_seam_bounds.  Axis 1 (per sample) is shared
    across columns and must NOT be sliced."""

    s_coords: np.ndarray
    """Spatial offsets along Steger normals, shape (M,).  Shared across all
    columns — NOT per-column.  truncate_to_seam_bounds leaves this unchanged."""


# ======================================================================
# Pipeline
# ======================================================================
class FuzzyPipeline:
    """Dual-path weld seam center extraction pipeline.

    Runs classical (Steger + CoG) and fuzzy-enhanced (IT2FLS + EKM)
    extraction in parallel, collecting both sets of results for
    downstream comparison and analysis.

    Parameters
    ----------
    fuzzy_config : FuzzyConfig, optional
        Configuration for FCM, morphology, barycentric, TFN.
        Defaults to ``FuzzyConfig()`` if None.
    it2fls_config : IT2FLSConfig, optional
        Configuration for the IT2FLS / EKM algorithm.
        Defaults to ``IT2FLSConfig()`` if None.
    """

    def __init__(
        self,
        fuzzy_config: Optional[FuzzyConfig] = None,
        it2fls_config: Optional[IT2FLSConfig] = None,
        hardware_config: Optional[HardwareConfig] = None,
        classical_config: Optional[ClassicalConfig] = None,
    ) -> None:
        cfg = fuzzy_config or FuzzyConfig()
        it2cfg = it2fls_config or IT2FLSConfig()
        hw = hardware_config or HardwareConfig()
        cl = classical_config or ClassicalConfig()
        self.config = cfg
        self.it2fls_config = it2cfg

        # ── Sub-modules ──
        self.fcm = FuzzyCMeans(
            n_clusters=cfg.fcm_clusters,
            m=cfg.fcm_fuzziness,
            max_iter=cfg.fcm_max_iter,
            tol=cfg.fcm_tolerance,
        )
        self.morph = FuzzyMorphology(
            kernel=np.asarray(cfg.morph_kernel, dtype=np.float64)
        )
        self.barycentric = FuzzyGrayBarycentric(
            saturation_threshold=cfg.saturation_threshold,
            specular_sigma=cfg.specular_penalty_sigma,
            noise_floor_percentile=cfg.noise_floor_percentile,
            intensification_iters=cfg.intensification_iterations,
        )
        self.tfn = TFNCenterExtractor(
            neighborhood_size=cfg.tfn_neighborhood_size,
            saturation_threshold=float(cfg.saturation_threshold),
            saturation_penalty_k=cfg.saturation_penalty_k,
            saturation_penalty_tau=cfg.saturation_penalty_tau,
        )
        self.it2fls = IT2FLSExtractor(config=it2cfg)
        # Sigma from hardware config (derived from physical laser width)
        # Steger threshold from classical config (rejects weak texture ridges)
        self.steger = StegerExtractor(
            sigma=hw.sigma,
            low_thresh=cl.steger_low_thresh,
        )
        self.sampler = ProfileSampler(
            half_width=it2cfg.profile_half_width,
            step=it2cfg.profile_sample_step,
        )

    # ------------------------------------------------------------------
    # Steger extraction with normals
    # ------------------------------------------------------------------
    def _extract_with_normals(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run Steger's algorithm and also return normal vectors.

        The standard ``StegerExtractor.extract_centers()`` does not
        expose the per-point normals (nx, ny).  This method repeats
        the Hessian computation to additionally extract them.

        Parameters
        ----------
        image : np.ndarray
            Single-channel grayscale image (float64 or uint8).
        mask : np.ndarray, optional
            Binary mask — only evaluate pixels where mask != 0.

        Returns
        -------
        x_sub : np.ndarray
            Sub-pixel column coordinates of ridge points.
        y_sub : np.ndarray
            Sub-pixel row coordinates of ridge points.
        strengths : np.ndarray
            Ridge strength (|eigenvalue|) at each point.
        nx_out : np.ndarray
            X-component of the Hessian eigenvector (cross-ridge normal).
        ny_out : np.ndarray
            Y-component of the Hessian eigenvector (cross-ridge normal).
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image.astype(np.float64)

        sigma = self.steger.sigma
        Ix, Iy, Ixx, Ixy, Iyy = StegerExtractor._compute_derivatives(img, sigma)
        eigenval_max, nx, ny, _ = StegerExtractor._compute_hessian_eigenvalues(
            Ixx, Ixy, Iyy
        )

        # Sub-pixel offset along the principal direction
        numerator = -(nx * Ix + ny * Iy)
        denominator = nx ** 2 * Ixx + 2.0 * nx * ny * Ixy + ny ** 2 * Iyy
        safe_denom = np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
        t = numerator / safe_denom

        # Acceptance criteria
        valid = np.abs(t) <= 0.5
        valid &= np.abs(eigenval_max) > self.steger.low_thresh
        valid &= eigenval_max < 0  # Ridge = local maximum in intensity
        if mask is not None:
            valid &= mask.astype(bool)

        rows, cols = np.where(valid)
        x_sub = cols.astype(np.float64) + t[valid] * nx[valid]
        y_sub = rows.astype(np.float64) + t[valid] * ny[valid]
        strengths = np.abs(eigenval_max[valid])
        nx_out = nx[valid]
        ny_out = ny[valid]

        return x_sub, y_sub, strengths, nx_out, ny_out

    # ------------------------------------------------------------------
    # Classical Center of Gravity (per-column)
    # ------------------------------------------------------------------
    @staticmethod
    def _classical_cog(
        image: np.ndarray, mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intensity-weighted Center of Gravity per column.

        For each column where the mask is active, computes:

            y_cog = Σ(y_i · I_i) / Σ(I_i)

        This is the classical baseline method.

        Parameters
        ----------
        image : np.ndarray
            Grayscale image (H, W).
        mask : np.ndarray
            Binary mask (H, W).

        Returns
        -------
        x_coords : np.ndarray
            Column indices with valid CoG.
        y_cog : np.ndarray
            Per-column CoG Y coordinates.
        """
        h, w = image.shape[:2]
        img = image.astype(np.float64)
        mask_bool = mask.astype(bool)

        # Zero out non-stripe pixels
        masked_img = np.where(mask_bool, img, 0.0)

        # Column sums
        col_sums = masked_img.sum(axis=0)  # (W,)
        valid_cols = col_sums > 1e-6

        y_idx = np.arange(h, dtype=np.float64)[:, np.newaxis]  # (H, 1)
        weighted_sums = (y_idx * masked_img).sum(axis=0)  # (W,)

        x_coords = np.where(valid_cols)[0]
        y_cog = weighted_sums[valid_cols] / col_sums[valid_cols]

        return x_coords, y_cog

    # ------------------------------------------------------------------
    # Profile classification
    # ------------------------------------------------------------------
    def _classify_profile(self, profile: np.ndarray) -> str:
        """Classify a cross-ridge profile for method selection.

        Returns
        -------
        str
            ``'gaussian'`` — suitable for IT2FLS (well-formed peak).
            ``'flat_top'`` — saturated plateau (use Fuzzy Barycentric).
            ``'noisy'``    — low SNR (use TFN).
        """
        profile = profile.astype(np.float64)
        peak = profile.max()

        if peak < 1e-6:
            return 'noisy'

        # Flatness check
        median_val = np.median(profile)
        stripe_pixels = profile > median_val
        stripe_width = max(int(stripe_pixels.sum()), 1)
        n_near_peak = int(np.sum(profile > 0.95 * peak))
        flatness = n_near_peak / stripe_width

        if flatness > self.config.flatness_threshold:
            return 'flat_top'

        # SNR check
        background = profile[profile <= median_val]
        bg_std = max(float(np.std(background)) if background.size > 0 else 1e-6, 1e-6)
        snr = peak / bg_std

        if snr < self.config.snr_threshold:
            return 'noisy'

        return 'gaussian'

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(
        self,
        roi_image: np.ndarray,
        stripe_mask: np.ndarray,
    ) -> FuzzyResult:
        """Execute the complete dual-path fuzzy processing pipeline.

        Stages
        ------
        1. **FCM Segmentation** — cluster pixels into background / laser / specular.
        2. **Fuzzy Morphological Cleaning** — remove noise, fill gaps.
        3. **Steger 2D Ridge Detection** — get ridge points + normals (nx, ny).
        4. **Classical CoG** — intensity-weighted center per column (baseline).
        5. **Profile Sampling** — sample intensity along Steger normals.
        6. **IT2FLS (EKM) Processing** — fuzzy-enhanced center + uncertainty.
           Falls back to Fuzzy Barycentric or TFN for degenerate profiles.
        7. **Result Assembly** — match column indices, package both paths.

        Parameters
        ----------
        roi_image : np.ndarray, shape (H, W)
            Grayscale ROI image.
        stripe_mask : np.ndarray, shape (H, W)
            Binary mask (non-zero where the laser stripe is).

        Returns
        -------
        FuzzyResult
            Both classical and fuzzy-enhanced centers, uncertainties,
            normals, membership map, and per-point method labels.
        """
        h, w = roi_image.shape[:2]
        gray = roi_image.astype(np.float64)
        mask_bool = stripe_mask.astype(bool)

        # ── Stage 1: FCM Segmentation ──
        if self.config.enable_fcm:
            membership_map = self.fcm.segment_stripe(
                roi_image if roi_image.dtype == np.uint8 else roi_image.astype(np.uint8),
                mask=stripe_mask,
            )
        else:
            membership_map = mask_bool.astype(np.float64)

        # ── Stage 2: Fuzzy Morphological Cleaning ──
        if self.config.enable_morphology:
            cleaned_map = self.morph.clean_stripe(membership_map)
        else:
            cleaned_map = membership_map

        # Build effective mask (cleaned membership > 0.3)
        effective_mask = (cleaned_map > 0.3).astype(np.uint8)
        # Combine with original stripe mask
        effective_mask = np.logical_and(effective_mask, mask_bool).astype(np.uint8)

        # ── Stage 3: Steger 2D Ridge Detection (with normals) ──
        ridge_x, ridge_y, strengths, nx, ny = self._extract_with_normals(
            roi_image, mask=effective_mask
        )
        logger.info("Steger ridge detection: %d points found.", len(ridge_x))

        # ── Stage 4: Classical CoG (baseline) ──
        cog_x, cog_y = self._classical_cog(gray, effective_mask)

        # If Steger found nothing, fall back to CoG-only
        if len(ridge_x) == 0:
            logger.warning("No Steger ridge points — using CoG only.")
            n = len(cog_x)
            _empty_profiles = np.zeros((n, 1), dtype=np.float64)
            _empty_s_coords = np.zeros(1, dtype=np.float64)
            return FuzzyResult(
                x_coords=cog_x.astype(np.int64),
                y_centers=cog_y.copy(),
                y_centers_classical=cog_y.copy(),
                confidences=np.ones(n, dtype=np.float64) * 0.5,
                uncertainties=np.zeros(n, dtype=np.float64),
                y_lower=cog_y.copy(),
                y_upper=cog_y.copy(),
                methods_used=['cog'] * n,
                membership_map=cleaned_map,
                normals_x=np.zeros(n, dtype=np.float64),
                normals_y=np.ones(n, dtype=np.float64),
                strengths=np.zeros(n, dtype=np.float64),
                profiles=_empty_profiles,
                s_coords=_empty_s_coords,
            )

        # ── Reduce to one point per column (strongest) ──
        ridge_cols = np.round(ridge_x).astype(np.int64)
        unique_cols = np.unique(ridge_cols)

        # For each column, pick the strongest ridge point
        best_idx = []
        for col in unique_cols:
            col_mask = ridge_cols == col
            idxs = np.where(col_mask)[0]
            best = idxs[np.argmax(strengths[idxs])]
            best_idx.append(best)
        best_idx = np.array(best_idx)

        sel_x = ridge_x[best_idx]
        sel_y = ridge_y[best_idx]
        sel_str = strengths[best_idx]
        sel_nx = nx[best_idx]
        sel_ny = ny[best_idx]
        sel_cols = unique_cols

        n_points = len(sel_x)

        # ── Stage 5: Profile Sampling along Steger normals ──
        profiles, s_coords = self.sampler.sample_profiles(
            gray, sel_x, sel_y, sel_nx, sel_ny,
        )

        # ── Stage 6: IT2FLS (EKM) Processing ──
        y_fuzzy = np.empty(n_points, dtype=np.float64)
        y_lower = np.empty(n_points, dtype=np.float64)
        y_upper = np.empty(n_points, dtype=np.float64)
        confidences = np.empty(n_points, dtype=np.float64)
        uncertainties = np.empty(n_points, dtype=np.float64)
        methods = []

        for i in range(n_points):
            profile = profiles[i]
            ptype = self._classify_profile(profile)

            if ptype == 'gaussian':
                # IT2FLS extraction
                y_c, y_l, y_r = self.it2fls.extract_center(profile, s_coords)

                if not np.isnan(y_c):
                    # Map profile-space center back to image coordinates
                    # s=0 corresponds to the ridge point, so offset = y_c
                    y_img = sel_y[i] + y_c * sel_ny[i]
                    y_l_img = sel_y[i] + y_l * sel_ny[i]
                    y_r_img = sel_y[i] + y_r * sel_ny[i]

                    y_fuzzy[i] = y_img
                    y_lower[i] = min(y_l_img, y_r_img)
                    y_upper[i] = max(y_l_img, y_r_img)
                    unc = abs(y_r_img - y_l_img)
                    uncertainties[i] = unc
                    confidences[i] = max(1.0 - unc / max(h, 1), 0.0)
                    methods.append('it2fls')
                else:
                    # IT2FLS failed — fallback to Fuzzy Barycentric
                    col_profile = gray[:, int(round(sel_x[i]))].astype(np.float64)
                    col_mask = effective_mask[:, int(round(sel_x[i]))]
                    col_profile = np.where(col_mask.astype(bool), col_profile, 0.0)
                    cy, conf = self.barycentric.extract_center(col_profile)
                    y_fuzzy[i] = cy
                    y_lower[i] = cy
                    y_upper[i] = cy
                    uncertainties[i] = 0.0
                    confidences[i] = conf
                    methods.append('barycentric')

            elif ptype == 'flat_top':
                # Fuzzy Barycentric for saturated profiles
                col_idx = int(round(sel_x[i]))
                col_idx = max(0, min(col_idx, w - 1))
                col_profile = gray[:, col_idx].astype(np.float64)
                col_mask = effective_mask[:, col_idx]
                col_profile = np.where(col_mask.astype(bool), col_profile, 0.0)
                cy, conf = self.barycentric.extract_center(col_profile)
                y_fuzzy[i] = cy
                y_lower[i] = cy
                y_upper[i] = cy
                uncertainties[i] = 0.0
                confidences[i] = conf
                methods.append('barycentric')

            else:  # 'noisy'
                # TFN for noisy profiles
                col_idx = int(round(sel_x[i]))
                col_idx = max(0, min(col_idx, w - 1))
                col_profile = gray[:, col_idx].astype(np.float64)
                col_mask = effective_mask[:, col_idx]
                col_profile = np.where(col_mask.astype(bool), col_profile, 0.0)
                cy, y_lo, y_hi, unc = self.tfn.extract_center(col_profile)
                y_fuzzy[i] = cy
                y_lower[i] = y_lo
                y_upper[i] = y_hi
                uncertainties[i] = unc
                confidences[i] = max(1.0 - min(unc / h, 1.0), 0.0)
                methods.append('tfn')

        # ── Stage 7: Match classical CoG to ridge columns ──
        y_classical = np.empty(n_points, dtype=np.float64)
        cog_lookup = dict(zip(cog_x.astype(np.int64), cog_y))
        for i, col in enumerate(sel_cols):
            if col in cog_lookup:
                y_classical[i] = cog_lookup[col]
            else:
                # No CoG for this column — use Steger ridge Y directly
                y_classical[i] = sel_y[i]

        return FuzzyResult(
            x_coords=sel_cols.astype(np.int64),
            y_centers=y_fuzzy,
            y_centers_classical=y_classical,
            confidences=confidences,
            uncertainties=uncertainties,
            y_lower=y_lower,
            y_upper=y_upper,
            methods_used=methods,
            membership_map=cleaned_map,
            normals_x=sel_nx,
            normals_y=sel_ny,
            strengths=sel_str,      # Steger ridge strength per selected column
            profiles=profiles,      # (N, M) cross-ridge profiles; axis-0 sliceable
            s_coords=s_coords,      # (M,) shared; NOT per-column, NOT sliced
        )
