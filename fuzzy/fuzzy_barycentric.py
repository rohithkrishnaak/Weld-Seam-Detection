"""
Fuzzy Gray Barycentric Method for Sub-Pixel Laser Stripe Center Extraction.

Novel contribution: Uses fuzzy membership functions to weight pixel
intensities, penalizing both noise (low membership) and specular
reflection saturation (exponential decay), achieving robust sub-pixel
accuracy on highly reflective metallic surfaces.

References:
    [1] Chaira, T. & Ray, A.K. (2010). Fuzzy Image Processing. CRC Press.
    [2] Versaci, M. et al. (2017). Fuzzy approach for image processing
        in industrial applications. LNEE, 431, 293-300.
    [3] Pal, S.K. & King, R.A. (1981). Image enhancement using smoothing
        with fuzzy sets. IEEE Trans. Systems, Man and Cybernetics, 11(7),
        494-501.  (Intensification operator)
"""

from __future__ import annotations

import numpy as np


class FuzzyGrayBarycentric:
    """Fuzzy-weighted barycentric sub-pixel center extraction.

    The core idea is to compute a *fuzzy membership* μ_laser(I) for every
    pixel intensity *I* along a vertical column profile.  The membership
    smoothly penalises:

    * **Background noise** — intensities below a data-driven threshold
      T_low receive μ = 0.
    * **Specular saturation** — intensities above the saturation point
      I_sat are penalised with an exponential decay, preventing saturated
      bright spots from biasing the center.

    The sub-pixel center is then the μ-weighted barycenter:

        y_center = Σ (y_i · μ_i · I_i) / Σ (μ_i · I_i)

    Parameters
    ----------
    saturation_threshold : int
        Intensity above which pixels are considered saturated (I_sat).
    specular_sigma : float
        Width (σ) of the Gaussian penalty applied to saturated pixels.
    noise_floor_percentile : float
        Percentile of non-zero intensities used to auto-estimate the
        noise floor T_low.
    intensification_iters : int
        Number of iterations of the Pal–King intensification operator
        applied to the membership function before computing the
        barycenter.
    """

    def __init__(
        self,
        saturation_threshold: int = 240,
        specular_sigma: float = 10.0,
        noise_floor_percentile: float = 10.0,
        intensification_iters: int = 2,
    ) -> None:
        self.saturation_threshold: int = saturation_threshold
        self.specular_sigma: float = specular_sigma
        self.noise_floor_percentile: float = noise_floor_percentile
        self.intensification_iters: int = intensification_iters

    # ------------------------------------------------------------------
    # Noise floor / peak estimation
    # ------------------------------------------------------------------
    def _compute_noise_floor(self, profile: np.ndarray) -> float:
        """Estimate the noise floor T_low from background statistics.

        Uses the ``noise_floor_percentile``-th percentile of *non-zero*
        pixels as a robust estimator of the background level.

        Parameters
        ----------
        profile : np.ndarray, shape (N,)
            Column intensity profile (float or uint8).

        Returns
        -------
        t_low : float
        """
        nonzero = profile[profile > 0]
        if nonzero.size == 0:
            return 0.0
        return float(np.percentile(nonzero, self.noise_floor_percentile))

    def _compute_peak_intensity(self, profile: np.ndarray) -> float:
        """Estimate the representative peak intensity I_peak.

        Uses the histogram mode of intensities above the median to focus
        on the stripe region rather than the dominant background.

        Parameters
        ----------
        profile : np.ndarray, shape (N,)
            Column intensity profile.

        Returns
        -------
        i_peak : float
        """
        median_val = np.median(profile)
        stripe_region = profile[profile > median_val]
        if stripe_region.size == 0:
            return float(np.max(profile))

        # Histogram with 50 bins over the stripe dynamic range
        counts, bin_edges = np.histogram(stripe_region, bins=50)
        # Mode = midpoint of the bin with the highest count
        mode_bin_idx = int(np.argmax(counts))
        i_peak = 0.5 * (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1])
        return float(i_peak)

    # ------------------------------------------------------------------
    # Fuzzy membership function
    # ------------------------------------------------------------------
    def _membership_function(
        self,
        intensities: np.ndarray,
        t_low: float,
        i_peak: float,
        i_sat: float,
        sigma: float,
    ) -> np.ndarray:
        """Compute the laser-stripe membership μ_laser(I) (vectorised).

        Piece-wise definition:
            μ = 0                                   if I < T_low
            μ = (I − T_low) / (I_peak − T_low)     if T_low ≤ I < I_peak
            μ = 1                                   if I_peak ≤ I ≤ I_sat
            μ = exp(−(I − I_sat)² / (2σ²))         if I > I_sat

        Parameters
        ----------
        intensities : np.ndarray
            Array of intensity values (any shape; operated element-wise).
        t_low, i_peak, i_sat : float
            Thresholds defining the piecewise regions.
        sigma : float
            Gaussian decay width for the specular region.

        Returns
        -------
        mu : np.ndarray, same shape as *intensities*, values in [0, 1].
        """
        I = intensities.astype(np.float64)

        # Ensure monotonic thresholds to avoid degenerate cases
        i_peak = max(i_peak, t_low + 1e-6)
        i_sat = max(i_sat, i_peak)

        # Pre-compute each piece
        ramp = (I - t_low) / (i_peak - t_low)  # linear ramp

        gaussian_decay = np.exp(
            -((I - i_sat) ** 2) / (2.0 * sigma ** 2)
        )

        # Assemble using np.where (fully vectorised, no Python loops)
        mu = np.where(
            I < t_low,
            0.0,
            np.where(
                I < i_peak,
                ramp,
                np.where(
                    I <= i_sat,
                    1.0,
                    gaussian_decay,
                ),
            ),
        )

        # Clamp to [0, 1] for safety
        return np.clip(mu, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Intensification operator
    # ------------------------------------------------------------------
    @staticmethod
    def _fuzzy_intensification(
        membership: np.ndarray, iterations: int = 2
    ) -> np.ndarray:
        """Apply the Pal & King (1981) intensification operator.

        The INT operator sharpens the membership function by pushing
        values away from 0.5 towards 0 or 1:

            μ' = 2μ²              if μ ≤ 0.5
            μ' = 1 − 2(1 − μ)²   if μ > 0.5

        Repeated application increases contrast between "definite member"
        and "definite non-member" regions.

        Parameters
        ----------
        membership : np.ndarray
            Membership values in [0, 1].
        iterations : int
            Number of times to apply the operator.

        Returns
        -------
        mu : np.ndarray, same shape, values in [0, 1].
        """
        mu = membership.copy()
        for _ in range(iterations):
            mu = np.where(
                mu <= 0.5,
                2.0 * mu ** 2,
                1.0 - 2.0 * (1.0 - mu) ** 2,
            )
        return mu

    # ------------------------------------------------------------------
    # Single-column extraction
    # ------------------------------------------------------------------
    def extract_center(
        self, column_profile: np.ndarray
    ) -> tuple[float, float]:
        """Extract the sub-pixel center for a single column profile.

        Steps:
            1. Compute noise floor T_low and peak intensity I_peak.
            2. Compute fuzzy membership μ for every pixel.
            3. Apply the Pal–King intensification operator.
            4. Compute the fuzzy-weighted barycenter:
                 y = Σ(y_i · μ_i · I_i) / Σ(μ_i · I_i)
            5. Return (center_y, confidence) where confidence = max(μ).

        Parameters
        ----------
        column_profile : np.ndarray, shape (N,)
            Vertical intensity profile.

        Returns
        -------
        center_y : float
            Sub-pixel Y coordinate of the stripe centre.
        confidence : float
            Maximum membership value — a quality indicator in [0, 1].
        """
        profile = column_profile.astype(np.float64)
        n = profile.size

        # Step 1: estimate thresholds
        t_low = self._compute_noise_floor(profile)
        i_peak = self._compute_peak_intensity(profile)
        i_sat = float(self.saturation_threshold)

        # Step 2: membership
        mu = self._membership_function(
            profile, t_low, i_peak, i_sat, self.specular_sigma
        )

        # Step 3: intensification
        mu = self._fuzzy_intensification(mu, self.intensification_iters)

        # Step 4: fuzzy-weighted barycenter
        y_indices = np.arange(n, dtype=np.float64)
        weights = mu * profile  # μ_i · I_i

        weight_sum = weights.sum()
        if weight_sum < 1e-12:
            # No valid signal — return midpoint with zero confidence
            return float(n) / 2.0, 0.0

        center_y = float(np.dot(y_indices, weights) / weight_sum)

        # Step 5: confidence
        confidence = float(np.max(mu))

        return center_y, confidence

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------
    def extract_centers_batch(
        self,
        roi_image: np.ndarray,
        stripe_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process an entire ROI and extract centres for all valid columns.

        A column is considered valid if its corresponding stripe_mask
        has at least one non-zero pixel.

        Parameters
        ----------
        roi_image : np.ndarray, shape (H, W)
            Grayscale ROI image.
        stripe_mask : np.ndarray, shape (H, W)
            Binary mask (non-zero where the stripe is expected).

        Returns
        -------
        x_coords : np.ndarray, shape (M,)
            Column indices with valid extractions.
        y_centers : np.ndarray, shape (M,)
            Sub-pixel Y centre coordinates.
        confidences : np.ndarray, shape (M,)
            Per-column confidence values.
        """
        roi = roi_image.astype(np.float64)
        mask = stripe_mask.astype(bool)
        h, w = roi.shape[:2]

        # Identify columns that have at least one masked pixel
        col_has_data = mask.any(axis=0)  # (W,)
        valid_cols = np.where(col_has_data)[0]

        if valid_cols.size == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        # Pre-compute per-column profiles (vectorised column slicing)
        # Build arrays to hold results
        x_coords = valid_cols.copy()
        y_centers = np.empty(valid_cols.size, dtype=np.float64)
        confidences = np.empty(valid_cols.size, dtype=np.float64)

        # --- Batch membership computation (vectorised across columns) ---
        # Extract all valid column profiles at once: (H, M)
        profiles = roi[:, valid_cols]  # (H, M)
        masks_cols = mask[:, valid_cols]  # (H, M)  boolean

        # Zero out pixels outside the mask
        profiles = np.where(masks_cols, profiles, 0.0)

        # Compute thresholds per column (vectorised)
        # Noise floor: percentile of non-zero values per column
        # We process column-by-column for threshold estimation because
        # percentile over masked subsets is inherently ragged.
        for idx, col_idx in enumerate(valid_cols):
            profile = profiles[:, idx]
            center_y, conf = self.extract_center(profile)
            y_centers[idx] = center_y
            confidences[idx] = conf

        return x_coords, y_centers, confidences

    # ------------------------------------------------------------------
    # ColumnExtractor protocol
    # ------------------------------------------------------------------
    def extract_column_center(
        self, column_profile: np.ndarray, x_coord: float
    ) -> tuple[float, float]:
        """Extract centre from a column profile (ColumnExtractor protocol).

        Thin wrapper around :meth:`extract_center` that accepts an
        ``x_coord`` argument for compatibility with the
        :class:`ColumnExtractor` protocol used by the adaptive pipeline.

        Parameters
        ----------
        column_profile : np.ndarray, shape (N,)
            Vertical intensity profile.
        x_coord : float
            Column (x) coordinate (unused, present for protocol).

        Returns
        -------
        y_center : float
            Sub-pixel Y coordinate of the stripe centre.
        confidence : float
            Maximum fuzzy membership — quality indicator in [0, 1].
        """
        return self.extract_center(column_profile)

