"""
Triangular Fuzzy Number (TFN) Based Laser Center Extraction.

Models pixel intensities as TFNs Ĩ = (I-δ, I, I+δ) instead of crisp
values, propagating uncertainty through the barycenter computation
to produce confidence intervals on the extracted center position.

The uncertainty bound δ is composed of two additive terms:

    δ_final = δ_std + k · exp(−(I_mean − I_sat)² / (2 · τ²))

where δ_std is the local standard deviation from a sliding window,
and the second term is a *saturation penalty* that injects additional
uncertainty when the local mean intensity approaches the sensor
saturation threshold I_sat.  This prevents the extractor from
reporting false high-confidence centres in specularly reflected
(all-255) regions where δ_std alone would be zero.

References:
    [1] Dubois, D. & Prade, H. (1980). Fuzzy Sets and Systems:
        Theory and Applications. Academic Press.
    [2] Zimmermann, H.J. (2001). Fuzzy Set Theory and Its Applications.
        Springer. 4th Edition.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class TFNCenterExtractor:
    """Triangular Fuzzy Number (TFN) based sub-pixel center extraction.

    Each pixel intensity *I* is modelled not as a crisp number but as a
    symmetric triangular fuzzy number  Ĩ = (I − δ, I, I + δ), where δ
    is the per-pixel uncertainty bound.

    The uncertainty δ is the sum of two components:

        δ_final = δ_std + k · exp(−(I_mean − I_sat)² / (2 · τ²))

    * **δ_std** — local standard deviation from a 1-D sliding window of
      length *neighborhood_size* (noise term).
    * **Saturation penalty** — a Gaussian bump centred at *I_sat* with
      amplitude *k* and width *τ*.  When the local mean intensity is
      close to the sensor saturation level, this term injects large
      additional uncertainty to prevent the extractor from being
      overconfident on specularly saturated pixels.

    The weighted barycenter is then computed three times — once for each
    vertex of the TFN — yielding a *fuzzy center interval*
    [y_low, y_best, y_high] that naturally quantifies positional
    uncertainty.

    The final defuzzified center uses the centroid formula for a
    triangular distribution:

        y_crisp = (y_low + 2·y_best + y_high) / 4

    Parameters
    ----------
    neighborhood_size : int
        Side length of the sliding window used to estimate local noise δ.
        Must be odd and ≥ 3.
    saturation_threshold : float
        Intensity level *I_sat* at which sensor saturation is assumed.
        The saturation penalty peaks when the local mean equals this
        value.  Default 240.
    saturation_penalty_k : float
        Amplitude *k* of the saturation penalty Gaussian.  Controls
        how much extra uncertainty is added for saturated regions.
        Default 5.0.
    saturation_penalty_tau : float
        Width *τ* (fixed constant) of the saturation penalty Gaussian.
        A smaller τ makes the penalty narrower around *I_sat*.  This is
        intentionally a fixed tuning constant — **not** the local
        standard deviation — to prevent division-by-zero when the
        local std is zero.  Default 15.0.
    """

    def __init__(
        self,
        neighborhood_size: int = 5,
        saturation_threshold: float = 240.0,
        saturation_penalty_k: float = 5.0,
        saturation_penalty_tau: float = 15.0,
    ) -> None:
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be odd and >= 3.")
        self.neighborhood_size: int = neighborhood_size
        self.saturation_threshold: float = float(saturation_threshold)
        self.saturation_penalty_k: float = float(saturation_penalty_k)
        self.saturation_penalty_tau: float = float(saturation_penalty_tau)

    # ------------------------------------------------------------------
    # Local noise estimation
    # ------------------------------------------------------------------
    def _estimate_uncertainty(
        self, profile: np.ndarray, neighborhood_size: int
    ) -> np.ndarray:
        """Compute per-pixel uncertainty δ with saturation penalty.

        The final uncertainty bound at each pixel is:

            δ_final = δ_std + k · exp(−(I_mean − I_sat)² / (2 · τ²))

        where:

        * **δ_std** is the local standard deviation computed from a
          1-D sliding window of length *neighborhood_size* using the
          identity  Var(X) = E[X²] − (E[X])².
        * **I_mean** is the local mean from the same sliding window.
        * **I_sat**, **k**, and **τ** are fixed constructor parameters
          (``saturation_threshold``, ``saturation_penalty_k``,
          ``saturation_penalty_tau``).

        The saturation penalty ensures that when the local mean
        approaches the sensor saturation level, δ is driven upward
        regardless of how small the local variance is.  This prevents
        the extractor from reporting spurious 100 % confidence on
        specularly reflected (all-255) regions.

        At the boundaries, reflect-padding is applied so the output
        has the same length as the input.

        Parameters
        ----------
        profile : np.ndarray, shape (N,)
            Intensity column profile.
        neighborhood_size : int
            Window length.

        Returns
        -------
        delta : np.ndarray, shape (N,)
            Per-pixel uncertainty bound (δ_final).
        """
        profile = profile.astype(np.float64)
        half = neighborhood_size // 2

        # Reflect-pad
        padded = np.pad(profile, pad_width=half, mode="reflect")

        # Sliding windows: (N, K)
        windows = sliding_window_view(padded, window_shape=neighborhood_size)

        # Vectorised variance: E[X²] − (E[X])²
        mean_vals = windows.mean(axis=1)
        mean_sq_vals = (windows ** 2).mean(axis=1)
        variance = np.maximum(mean_sq_vals - mean_vals ** 2, 0.0)

        delta_std = np.sqrt(variance)

        # Saturation penalty: Gaussian bump centred at I_sat
        #   penalty = k · exp(−(I_mean − I_sat)² / (2 · τ²))
        # τ is a FIXED tuning constant — NOT the local standard deviation.
        diff = mean_vals - self.saturation_threshold
        penalty = self.saturation_penalty_k * np.exp(
            -(diff ** 2) / (2.0 * self.saturation_penalty_tau ** 2)
        )

        delta = delta_std + penalty
        return delta

    # ------------------------------------------------------------------
    # TFN construction
    # ------------------------------------------------------------------
    @staticmethod
    def _create_tfn(
        intensity: np.ndarray, delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create TFN triplets (I_low, I_center, I_high) for each pixel.

        The triplet vertices are clipped to the valid intensity range
        [0, 255] to avoid negative or out-of-range values.

        Parameters
        ----------
        intensity : np.ndarray, shape (N,)
        delta : np.ndarray, shape (N,)

        Returns
        -------
        i_low : np.ndarray, shape (N,)
            Left (pessimistic) vertex:  max(0, I − δ).
        i_center : np.ndarray, shape (N,)
            Peak (best estimate):  I.
        i_high : np.ndarray, shape (N,)
            Right (optimistic) vertex:  min(255, I + δ).
        """
        i_low = np.clip(intensity - delta, 0.0, 255.0)
        i_center = np.clip(intensity, 0.0, 255.0)
        i_high = np.clip(intensity + delta, 0.0, 255.0)
        return i_low, i_center, i_high

    # ------------------------------------------------------------------
    # Single-column extraction
    # ------------------------------------------------------------------
    def extract_center(
        self, column_profile: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Extract a TFN sub-pixel center for a single column profile.

        Workflow:
            1. Estimate δ (local noise) for each pixel.
            2. Construct TFN triplets (I−δ, I, I+δ).
            3. Compute three weighted barycenters using α-cut
               decomposition at α = 0 (full support):
                 y_low  = Σ(y_i · I_low_i)  / Σ(I_low_i)    (pessimistic)
                 y_best = Σ(y_i · I_i)       / Σ(I_i)         (central)
                 y_high = Σ(y_i · I_high_i) / Σ(I_high_i)    (optimistic)
            4. Defuzzify via the TFN centroid:
                 y_crisp = (y_low + 2·y_best + y_high) / 4
            5. Uncertainty = y_high − y_low.

        Parameters
        ----------
        column_profile : np.ndarray, shape (N,)
            Vertical intensity profile.

        Returns
        -------
        y_crisp : float
            Defuzzified sub-pixel Y coordinate.
        y_low : float
            Lower (pessimistic) bound.
        y_high : float
            Upper (optimistic) bound.
        uncertainty : float
            Width of the fuzzy interval (y_high − y_low).
        """
        profile = column_profile.astype(np.float64)
        n = profile.size
        y_indices = np.arange(n, dtype=np.float64)

        # Step 1: local noise estimation
        delta = self._estimate_uncertainty(profile, self.neighborhood_size)

        # Step 2: TFN triplets
        i_low, i_center, i_high = self._create_tfn(profile, delta)

        # Step 3: weighted barycenters
        # Helper to compute weighted mean, returning midpoint on zero weight
        def _weighted_mean(weights: np.ndarray) -> float:
            s = weights.sum()
            if s < 1e-12:
                return float(n) / 2.0
            return float(np.dot(y_indices, weights) / s)

        y_low = _weighted_mean(i_low)
        y_best = _weighted_mean(i_center)
        y_high = _weighted_mean(i_high)

        # Step 4: TFN centroid defuzzification (Dubois & Prade 1980)
        y_crisp = (y_low + 2.0 * y_best + y_high) / 4.0

        # Step 5: uncertainty interval
        uncertainty = y_high - y_low

        return y_crisp, y_low, y_high, uncertainty

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------
    def extract_centers_batch(
        self,
        roi_image: np.ndarray,
        stripe_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract TFN centres for all valid columns in the ROI.

        A column is valid if it has at least one non-zero pixel in
        ``stripe_mask``.

        Parameters
        ----------
        roi_image : np.ndarray, shape (H, W)
            Grayscale ROI image.
        stripe_mask : np.ndarray, shape (H, W)
            Binary mask.

        Returns
        -------
        x_coords : np.ndarray, shape (M,)
            Valid column indices.
        y_crisps : np.ndarray, shape (M,)
            Defuzzified sub-pixel Y centres.
        y_lows : np.ndarray, shape (M,)
            Lower bounds.
        y_highs : np.ndarray, shape (M,)
            Upper bounds.
        uncertainties : np.ndarray, shape (M,)
            Interval widths.
        """
        roi = roi_image.astype(np.float64)
        mask = stripe_mask.astype(bool)

        # Valid columns: at least one masked pixel
        col_has_data = mask.any(axis=0)
        valid_cols = np.where(col_has_data)[0]

        if valid_cols.size == 0:
            empty = np.array([], dtype=np.float64)
            return (
                np.array([], dtype=np.int64),
                empty.copy(),
                empty.copy(),
                empty.copy(),
                empty.copy(),
            )

        m = valid_cols.size
        x_coords = valid_cols.copy()
        y_crisps = np.empty(m, dtype=np.float64)
        y_lows = np.empty(m, dtype=np.float64)
        y_highs = np.empty(m, dtype=np.float64)
        uncertainties = np.empty(m, dtype=np.float64)

        # Extract all valid column profiles: (H, M)
        profiles = roi[:, valid_cols]
        masks_cols = mask[:, valid_cols]
        profiles = np.where(masks_cols, profiles, 0.0)

        for idx in range(m):
            y_c, y_l, y_h, unc = self.extract_center(profiles[:, idx])
            y_crisps[idx] = y_c
            y_lows[idx] = y_l
            y_highs[idx] = y_h
            uncertainties[idx] = unc

        return x_coords, y_crisps, y_lows, y_highs, uncertainties
