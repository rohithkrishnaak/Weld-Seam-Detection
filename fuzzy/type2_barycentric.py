"""
Interval Type-2 Fuzzy Logic System (IT2FLS) for Laser Center Extraction.

Novel contribution: Uses the Enhanced Karnik-Mendel (EKM) algorithm to
compute left and right centroid bounds. The membership functions are
monotonically decreasing with intensity, penalizing near-saturated pixels
to prevent spurious detections on highly reflective metallic surfaces.

References:
    [1] Wu, H. & Mendel, J. M. (2009). Enhanced Karnik-Mendel Algorithms.
        IEEE Trans. Fuzzy Systems, 17(4), 923-934.
"""

from __future__ import annotations

import numpy as np

from config import IT2FLSConfig


class IT2FLSExtractor:
    """Interval Type-2 Fuzzy Logic sub-pixel center extraction.

    Uses the EKM algorithm to compute left and right bounds [y_l, y_r].

    Parameters
    ----------
    config : IT2FLSConfig, optional
        Configuration dataclass.  When *None*, default parameters are
        used (``IT2FLSConfig()``).
    """

    def __init__(self, config: IT2FLSConfig | None = None) -> None:
        cfg = config or IT2FLSConfig()
        self.k: float = float(cfg.k_exponent)
        self.epsilon: float = float(cfg.epsilon)
        self.max_iter: int = cfg.max_iter
        self.i_sat_percentile: float = float(cfg.i_sat_percentile)

    def _compute_memberships(self, profile: np.ndarray, i_sat: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute Upper and Lower Membership Functions (UMF, LMF)."""
        # Ensure i_sat > 0 to avoid division by zero
        i_sat = max(i_sat, 1e-6)
        
        # UMF: monotonically decreasing function of intensity
        clamped_I = np.minimum(profile, i_sat)
        mu_upper = 1.0 - (clamped_I / i_sat) ** self.k
        mu_upper = np.clip(mu_upper, 0.0, 1.0)
        
        # LMF: strictly zero for saturated pixels
        mu_lower = np.where(profile >= i_sat, 0.0, mu_upper)
        
        return mu_upper, mu_lower

    def _ekm_algorithm(self, s: np.ndarray, w_l: np.ndarray, w_u: np.ndarray) -> tuple[float, float]:
        """Enhanced Karnik-Mendel (EKM) algorithm for interval centroid.
        
        Parameters
        ----------
        s : np.ndarray
            Spatial coordinates (must be strictly increasing).
        w_l : np.ndarray
            Lower membership weights.
        w_u : np.ndarray
            Upper membership weights.
            
        Returns
        -------
        y_l, y_r : float
            Left and right centroid bounds.
        """
        N = len(s)
        
        # -------------------------------------------------------------
        # 1. Compute Left Bound (y_l)
        # -------------------------------------------------------------
        k_l = int(round(N / 2.4))  # Initialization heuristic from Wu & Mendel (2009)
        k_l = max(0, min(k_l, N - 1))
        
        for _ in range(self.max_iter):
            # weights for left bound: w_u for i <= k, w_l for i > k
            w_left = np.where(np.arange(N) <= k_l, w_u, w_l)
            sum_w = w_left.sum()
            y_l_prime = np.dot(s, w_left) / sum_w if sum_w > 1e-12 else s[N//2]
            
            # Find new switch point k_prime where s[k] <= y_l_prime < s[k+1]
            k_l_prime = np.searchsorted(s, y_l_prime) - 1
            k_l_prime = max(0, min(k_l_prime, N - 2))
            
            if k_l_prime == k_l:
                y_l = y_l_prime
                break
            k_l = k_l_prime
        else:
            y_l = y_l_prime  # Fallback if max_iter reached
            
        # -------------------------------------------------------------
        # 2. Compute Right Bound (y_r)
        # -------------------------------------------------------------
        k_r = int(round(N / 1.7))
        k_r = max(0, min(k_r, N - 1))
        
        for _ in range(self.max_iter):
            # weights for right bound: w_l for i <= k, w_u for i > k
            w_right = np.where(np.arange(N) <= k_r, w_l, w_u)
            sum_w = w_right.sum()
            y_r_prime = np.dot(s, w_right) / sum_w if sum_w > 1e-12 else s[N//2]
            
            k_r_prime = np.searchsorted(s, y_r_prime) - 1
            k_r_prime = max(0, min(k_r_prime, N - 2))
            
            if k_r_prime == k_r:
                y_r = y_r_prime
                break
            k_r = k_r_prime
        else:
            y_r = y_r_prime

        return float(y_l), float(y_r)

    def extract_center(
        self, profile: np.ndarray, s_coords: np.ndarray | None = None
    ) -> tuple[float, float, float]:
        """Extract IT2FLS center for a 1-D intensity profile.

        Parameters
        ----------
        profile : np.ndarray, shape (N,)
            Intensity values along the profile.
        s_coords : np.ndarray, shape (N,), optional
            Spatial coordinates corresponding to each intensity sample.
            Defaults to ``np.arange(N)``.

        Returns
        -------
        y_center : float
            Defuzzified centre position (midpoint of EKM bounds).
        y_l : float
            Left (lower) centroid bound from EKM.
        y_r : float
            Right (upper) centroid bound from EKM.
        """
        profile = profile.astype(np.float64)
        N = len(profile)

        # --- Input validation: degenerate profiles ---
        if N == 0 or np.all(profile == 0):
            return float("nan"), float("nan"), float("nan")
        if np.ptp(profile) < 1e-12:
            # All-same-value profile — no structure to extract
            return float("nan"), float("nan"), float("nan")

        if s_coords is None:
            s_coords = np.arange(N, dtype=np.float64)

        # Per-profile dynamic I_sat
        i_sat = float(np.percentile(profile, self.i_sat_percentile))

        mu_upper, mu_lower = self._compute_memberships(profile, i_sat)
        
        # --- Degenerate Case Handlers ---
        sum_mu_lower = mu_lower.sum()
        sum_mu_upper = mu_upper.sum()
        
        # Case 1: All saturated (no reliable information)
        if sum_mu_lower < 1e-6 and sum_mu_upper < 1e-6:
            return float('nan'), float('nan'), float('nan')
            
        # Case 2: Only UMF is zero-sum
        if sum_mu_upper < 1e-6:
            return float('nan'), float('nan'), float('nan')
            
        # Case 3: Single sample above LMF
        nonzero_lower_indices = np.where(mu_lower > 0)[0]
        if len(nonzero_lower_indices) == 1:
            val = float(s_coords[nonzero_lower_indices[0]])
            return val, val, val
            
        # EKM Iteration
        y_l, y_r = self._ekm_algorithm(s_coords, mu_lower, mu_upper)
        
        # Case 4: EKM collapses to Type-1
        if abs(y_r - y_l) < self.epsilon:
            return y_l, y_l, y_l
            
        y_center = (y_l + y_r) / 2.0

        return y_center, y_l, y_r

    # ------------------------------------------------------------------
    # ColumnExtractor protocol
    # ------------------------------------------------------------------
    def extract_column_center(
        self, column_profile: np.ndarray, x_coord: float
    ) -> tuple[float, float]:
        """Extract centre from a column profile (ColumnExtractor protocol).

        Wraps :meth:`extract_center` and returns ``(y_center, confidence)``
        where confidence is derived from the EKM interval width::

            confidence = 1.0 - |y_r - y_l| / max(len(profile), 1)

        A narrow interval (y_r ≈ y_l) yields confidence close to 1;
        a wide interval indicates high positional uncertainty.

        Parameters
        ----------
        column_profile : np.ndarray, shape (N,)
            1-D intensity profile.
        x_coord : float
            Column (x) coordinate (unused, present for protocol).

        Returns
        -------
        y_center : float
            Sub-pixel centre position.
        confidence : float
            Confidence score in [0, 1].
        """
        y_center, y_l, y_r = self.extract_center(column_profile)
        if np.isnan(y_center):
            return float("nan"), 0.0
        confidence = 1.0 - abs(y_r - y_l) / max(len(column_profile), 1)
        return y_center, max(confidence, 0.0)
