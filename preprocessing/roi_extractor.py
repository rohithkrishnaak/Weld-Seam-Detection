"""
Classical ROI Extraction for Laser Stripe Detection.

Replaces the DL-based ``ROIPredictor`` with a classical image processing
pipeline that segments the 650 nm red laser stripe from the background
using **chromaticity-based** extraction and connected-component analysis.

For a red laser, the key distinguishing feature is *red dominance* — the
red channel being significantly brighter than both the green and blue
channels.  This is far more selective than a raw red-channel threshold,
because warm-lit backgrounds (wood, rust, brown metal) also have high
red channel values but are not *predominantly* red.

Pipeline:
    1. Compute red-dominance map: R − max(G, B)
    2. HSV-based laser mask: (H ∈ [0,10] ∪ [170,180]) ∧ S > S_min ∧ V > V_min
    3. Combine chromaticity + HSV via logical OR
    4. Gaussian blur + Otsu threshold on the combined signal
    5. Morphological closing to connect stripe fragments, then opening to remove noise
    6. Connected-component analysis — keep only the largest component
    7. Return a binary mask (uint8, 0 or 255)

No neural-network or deep-learning dependencies are used.

References
----------
.. [1] Otsu, N. (1979). A threshold selection method from gray-level
       histograms. IEEE Trans. Syst., Man, Cybern., 9(1), 62–66.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import ROIConfig

logger = logging.getLogger(__name__)


class ROIExtractor:
    """Chromaticity-based laser-stripe ROI extractor.

    Uses red-dominance (R − max(G, B)) combined with HSV hue filtering
    to isolate a 650 nm red laser stripe from arbitrary backgrounds
    including warm-toned surfaces (wood, rust, brown metal).

    Designed as a drop-in replacement for the DL-based ``ROIPredictor``.
    Call :meth:`predict` for backward-compatible usage.

    Parameters
    ----------
    config : ROIConfig, optional
        Configuration dataclass.  Defaults to ``ROIConfig()`` if *None*.
    """

    def __init__(self, config: Optional[ROIConfig] = None) -> None:
        self.config = config if config is not None else ROIConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to single-channel using the red channel for a 650 nm laser.

        For a red laser on reflective/wood surfaces, we use R - max(G, B)
        to suppress broadband bright areas (like wood grain) and isolate
        pure red light.

        Parameters
        ----------
        image : np.ndarray
            Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` image.

        Returns
        -------
        np.ndarray
            Single-channel image, dtype ``uint8``.
        """
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)

        if image.ndim == 3 and image.shape[2] == 3:
            b = image[:, :, 0].astype(np.int16)
            g = image[:, :, 1].astype(np.int16)
            r = image[:, :, 2].astype(np.int16)
            max_bg = np.maximum(b, g)
            laser_signal = np.clip(r - max_bg, 0, 255).astype(np.uint8)
            return laser_signal

        raise ValueError(
            f"Expected grayscale (H, W) or BGR (H, W, 3), "
            f"got shape {image.shape}"
        )

    def extract_mask(self, image: np.ndarray) -> np.ndarray:
        """Segment the laser stripe and return a binary mask.

        Parameters
        ----------
        image : np.ndarray
            Input image — either single-channel grayscale ``(H, W)`` or
            3-channel BGR ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Binary mask of shape ``(H, W)`` with dtype ``uint8``.
            Foreground (laser stripe) pixels are 255, background 0.
        """
        if image.ndim == 2:
            # Grayscale input — fall back to simple Otsu
            return self._extract_grayscale(image)

        if image.ndim == 3 and image.shape[2] == 3:
            return self._extract_bgr(image)

        raise ValueError(
            f"Expected grayscale (H, W) or BGR (H, W, 3), "
            f"got shape {image.shape}"
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Backward-compatible interface matching ``ROIPredictor.predict``.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale input image.

        Returns
        -------
        np.ndarray
            Binary mask (uint8, 0/255) of the detected laser stripe.
        """
        return self.extract_mask(image)

    # ------------------------------------------------------------------
    # BGR pipeline (primary — uses chromaticity)
    # ------------------------------------------------------------------

    def _extract_bgr(self, image: np.ndarray) -> np.ndarray:
        """Extract laser stripe from a BGR image using chromaticity.

        Combines two complementary signals:

        1. **Red dominance**: R − max(G, B) > threshold
           Isolates "pure red" from warm-toned surfaces where R, G, B
           are all moderately high.

        2. **HSV hue**: Hue ∈ [0,10] ∪ [170,180] (red range in OpenCV's
           0-180 hue space), with saturation and value guards.

        The union of both signals is thresholded and cleaned.
        """
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        # ── Signal 1: Red dominance ──
        # R − max(G, B): positive only where red is the dominant channel
        red_dominance = r - np.maximum(g, b)
        # Clip negative values to 0, scale to uint8 for Otsu
        red_dominance = np.clip(red_dominance, 0, 255).astype(np.uint8)

        # ── Signal 2: HSV hue filter ──
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Red hue in OpenCV: 0-10 and 170-180
        hue_mask = ((h <= 10) | (h >= 170))
        sat_mask = s > self.config.hsv_sat_min
        val_mask = v > self.config.hsv_val_min
        hsv_mask = (hue_mask & sat_mask & val_mask).astype(np.uint8) * 255

        # ── Combine signals ──
        # Use the stronger of the two: Otsu on red-dominance, then OR with HSV
        sigma = self.config.blur_sigma
        ksize = int(2 * np.ceil(3.0 * sigma) + 1)
        ksize = max(ksize, 3)

        # Otsu on red-dominance channel
        blurred_rd = cv2.GaussianBlur(red_dominance, (ksize, ksize), sigma)
        # Ensure there's enough foreground for Otsu
        if blurred_rd.max() < 10:
            # Red dominance is too weak — pure HSV fallback
            logger.info("Red dominance weak; falling back to HSV-only mask.")
            combined_binary = hsv_mask
        else:
            _, rd_binary = cv2.threshold(
                blurred_rd, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            # Union of Otsu(red-dominance) and HSV
            combined_binary = np.maximum(rd_binary, hsv_mask)

        # ── Morphological cleaning ──
        # Close first to connect stripe fragments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_close)

        # Open to remove small noise blobs
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        # ── Keep largest component ──
        mask = self._keep_largest_component(cleaned)

        n_stripe = int(np.count_nonzero(mask))
        total = mask.size
        pct = 100.0 * n_stripe / total
        logger.debug(
            "ROI mask (chromaticity): %d stripe pixels (%.1f%% of image)",
            n_stripe, pct,
        )

        # Sanity check: if the mask still covers > 30% of the image,
        # something is wrong — tighten the threshold adaptively.
        if pct > 30.0:
            logger.warning(
                "ROI mask covers %.1f%% of image — applying strict threshold.",
                pct,
            )
            mask = self._strict_red_dominance(red_dominance, hsv_mask)

        return mask

    def _strict_red_dominance(
        self, red_dominance: np.ndarray, hsv_mask: np.ndarray
    ) -> np.ndarray:
        """Fallback strict extraction when the primary method is too permissive.

        Uses a percentile-based threshold on the red-dominance signal
        instead of Otsu.
        """
        # Use the 90th percentile of non-zero red-dominance as threshold
        nonzero = red_dominance[red_dominance > 0]
        if len(nonzero) < 100:
            # Not enough red-dominant pixels; use HSV only
            return self._keep_largest_component(hsv_mask)

        thresh = np.percentile(nonzero, 70)
        strict_binary = (red_dominance > thresh).astype(np.uint8) * 255

        # Intersect with HSV mask for extra selectivity
        combined = np.bitwise_and(strict_binary, hsv_mask)

        # If intersection is too small, fall back to strict binary only
        if np.count_nonzero(combined) < self.config.min_stripe_area:
            combined = strict_binary

        # Morphological cleaning
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        return self._keep_largest_component(cleaned)

    # ------------------------------------------------------------------
    # Grayscale fallback
    # ------------------------------------------------------------------

    def _extract_grayscale(self, gray: np.ndarray) -> np.ndarray:
        """Fallback extraction for single-channel images.

        Uses Gaussian blur + Otsu thresholding (the original method).
        This works well when the image is already a red-dominance map
        or a properly filtered single-channel input.
        """
        gray = gray.astype(np.uint8)
        sigma = self.config.blur_sigma
        ksize = int(2 * np.ceil(3.0 * sigma) + 1)
        ksize = max(ksize, 3)

        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return self._keep_largest_component(cleaned)

    # ------------------------------------------------------------------
    # Connected-component filtering
    # ------------------------------------------------------------------

    def _keep_largest_component(self, binary: np.ndarray) -> np.ndarray:
        """Retain only the largest connected component above the area threshold.

        Parameters
        ----------
        binary : np.ndarray
            Cleaned binary mask.

        Returns
        -------
        np.ndarray
            Binary mask containing only the largest connected component
            whose area exceeds ``config.min_stripe_area``.  Returns an
            all-zero mask if no qualifying component exists.
        """
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if n_labels <= 1:
            # Only background — no foreground components found.
            logger.warning("No connected components found in binary mask.")
            return np.zeros_like(binary)

        # stats columns: [x, y, w, h, area].  Skip label 0 (background).
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1  # +1 to offset background

        if areas[largest_idx - 1] < self.config.min_stripe_area:
            logger.warning(
                "Largest component area (%d px) below threshold (%d px).",
                areas[largest_idx - 1],
                self.config.min_stripe_area,
            )
            return np.zeros_like(binary)

        mask = np.where(labels == largest_idx, np.uint8(255), np.uint8(0))
        return mask
