"""
Profile-Adaptive Laser Center Extraction.

Automatically selects the optimal extraction method per column
based on the local intensity profile characteristics:

- **Gaussian profile** → Steger's algorithm (best precision)
- **Flat-top / saturated** → Fuzzy Gray Barycentric (handles saturation)
- **Noisy / low-SNR** → TFN Barycentric (uncertainty-aware)

The classifier examines three profile features:

1. **Flatness ratio** — fraction of the laser stripe width that is within
   5 % of the peak value (saturated region vs. total stripe width above
   the noise floor).  A high ratio indicates a saturated / flat-top profile.
2. **Peak SNR** — ratio of the peak intensity above background to the
   background noise standard deviation.  A low SNR triggers the
   uncertainty-aware TFN extractor.
3. If neither condition is met the profile is assumed Gaussian, and
   Steger's Hessian-based detector is used for maximum precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol definitions for pluggable extractors
# ---------------------------------------------------------------------------

class ColumnExtractor(Protocol):
    """Minimal interface that per-column extractors must satisfy."""

    def extract_column_center(
        self, column_profile: np.ndarray, x_coord: float
    ) -> Tuple[float, float]:
        """Return (y_center, confidence) for a single column profile."""
        ...


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Structured result of the adaptive extraction pipeline."""

    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    confidence: np.ndarray = field(default_factory=lambda: np.empty(0))
    uncertainty: np.ndarray = field(default_factory=lambda: np.empty(0))
    method: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adaptive extractor
# ---------------------------------------------------------------------------

class AdaptiveExtractor:
    """Profile-adaptive laser centre extractor.

    Parameters
    ----------
    steger : ColumnExtractor
        Steger-based sub-pixel ridge extractor (Gaussian profiles).
    fuzzy_bary : ColumnExtractor
        Fuzzy gray-barycentric extractor (saturated / flat-top profiles).
    tfn_extractor : ColumnExtractor
        TFN-barycentric extractor (noisy / low-SNR profiles).
    flatness_thresh : float
        Fraction of peak-region pixels that must be within 5 % of the
        maximum for the profile to be classified as *flat_top*.
    snr_thresh : float
        Minimum peak signal-to-noise ratio.  Profiles below this are
        classified as *noisy*.
    """

    def __init__(
        self,
        steger: ColumnExtractor,
        fuzzy_bary: ColumnExtractor,
        tfn_extractor: ColumnExtractor,
        flatness_thresh: float = 0.3,
        snr_thresh: float = 5.0,
    ) -> None:
        self.steger = steger
        self.fuzzy_bary = fuzzy_bary
        self.tfn_extractor = tfn_extractor
        self.flatness_thresh = flatness_thresh
        self.snr_thresh = snr_thresh

        # Map profile class names to extractors for dispatch
        self._extractors: Dict[str, ColumnExtractor] = {
            "gaussian": self.steger,
            "flat_top": self.fuzzy_bary,
            "noisy": self.tfn_extractor,
        }

    # ------------------------------------------------------------------
    # Profile classification
    # ------------------------------------------------------------------

    def classify_profile(
        self, profile: np.ndarray, background_std: float
    ) -> str:
        """Classify an intensity profile into one of three categories.

        The flatness ratio measures how much of the laser stripe is
        saturated (near the peak value) relative to its total width::

            T_low        = percentile_10(profile[profile > 0])
            stripe_width = count(profile > T_low)
            n_near_peak  = count(profile > 0.95 * peak)
            flatness     = n_near_peak / max(stripe_width, 1)

        A high flatness ratio indicates a saturated / flat-top profile.

        Parameters
        ----------
        profile : np.ndarray
            1-D intensity profile (float or uint8).
        background_std : float
            Standard deviation of the background noise, estimated
            externally (e.g. from a region outside the laser stripe).

        Returns
        -------
        str
            ``'gaussian'``, ``'flat_top'``, or ``'noisy'``.
        """
        profile = profile.astype(np.float64)
        peak_val = profile.max()
        background_mean = np.median(profile)

        # --- Flatness check ---
        # T_low: noise floor estimated as the 10th percentile of
        # non-zero pixels — separates the laser stripe from background.
        nonzero_pixels = profile[profile > 0]
        if nonzero_pixels.size > 0:
            t_low: float = float(np.percentile(nonzero_pixels, 10))
        else:
            t_low = 0.0

        # stripe_width: number of pixels above the noise floor
        # (the actual width of the laser line).
        stripe_width: int = int(np.count_nonzero(profile > t_low))

        # n_near_peak: number of pixels within 5 % of the peak
        # (the saturated / flat region).
        n_near_peak: int = int(np.count_nonzero(profile > 0.95 * peak_val))

        flatness_ratio: float = n_near_peak / max(stripe_width, 1)

        if flatness_ratio > self.flatness_thresh:
            return "flat_top"

        # --- SNR check ---
        signal = peak_val - background_mean
        # Guard against zero background noise
        snr = signal / background_std if background_std > 1e-6 else float("inf")

        if snr < self.snr_thresh:
            return "noisy"

        return "gaussian"

    # ------------------------------------------------------------------
    # Single-column extraction
    # ------------------------------------------------------------------

    def extract_center(
        self,
        column_profile: np.ndarray,
        x_coord: float,
        background_std: float,
    ) -> Dict[str, Any]:
        """Extract the laser centre from a single column profile.

        The profile is first classified, then the appropriate extractor
        is dispatched.

        Parameters
        ----------
        column_profile : np.ndarray
            1-D intensity profile.
        x_coord : float
            Column (x) coordinate for reference.
        background_std : float
            Background noise σ for classification.

        Returns
        -------
        dict
            ``'y_center'``, ``'confidence'``, ``'uncertainty'``,
            ``'method'``.
        """
        method_name = self.classify_profile(column_profile, background_std)
        extractor = self._extractors[method_name]

        y_center, confidence = extractor.extract_column_center(
            column_profile, x_coord
        )

        # Heuristic uncertainty estimate: inversely proportional to
        # confidence, bounded below by a minimum of 0.01 px.
        uncertainty = (
            1.0 / max(confidence, 1e-6)
            if confidence > 0
            else float("inf")
        )

        return {
            "y_center": y_center,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "method": method_name,
        }

    # ------------------------------------------------------------------
    # Full ROI extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        roi_image: np.ndarray,
        stripe_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Process an entire ROI image column-by-column.

        Parameters
        ----------
        roi_image : np.ndarray
            Grayscale ROI image (H×W, uint8 or float).
        stripe_mask : np.ndarray
            Binary mask (H×W) indicating columns that contain a laser
            stripe.  Only columns with ``np.any(stripe_mask[:, c])`` are
            processed.

        Returns
        -------
        dict
            ``'x'``, ``'y'``, ``'confidence'``, ``'uncertainty'`` as
            numpy arrays; ``'method'`` as a list of strings (one per
            accepted column).
        """
        roi = roi_image.astype(np.float64)
        h, w = roi.shape[:2]

        # Estimate global background noise from the masked-out region
        bg_pixels = roi[~stripe_mask.astype(bool)]
        background_std = float(np.std(bg_pixels)) if bg_pixels.size > 0 else 1.0

        result = ExtractionResult()

        xs: List[float] = []
        ys: List[float] = []
        confs: List[float] = []
        uncs: List[float] = []
        methods: List[str] = []

        for c in range(w):
            if not np.any(stripe_mask[:, c]):
                continue

            profile = roi[:, c]
            res = self.extract_center(profile, float(c), background_std)

            if np.isnan(res["y_center"]):
                continue

            xs.append(float(c))
            ys.append(res["y_center"])
            confs.append(res["confidence"])
            uncs.append(res["uncertainty"])
            methods.append(res["method"])

        logger.info(
            "Adaptive extraction: %d centres (gaussian=%d, flat_top=%d, noisy=%d)",
            len(xs),
            methods.count("gaussian"),
            methods.count("flat_top"),
            methods.count("noisy"),
        )

        return {
            "x": np.array(xs),
            "y": np.array(ys),
            "confidence": np.array(confs),
            "uncertainty": np.array(uncs),
            "method": methods,
        }
