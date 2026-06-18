"""
Fuzzy C-Means (FCM) Clustering for Laser Stripe Segmentation.

Segments pixel intensities within a masked region into *n* clusters
(default 3: background, laser stripe, specular highlight) and returns
a soft membership map for the laser cluster.

The implementation is fully vectorised with NumPy — no Python-level
loops iterate over individual pixels.

Algorithm
---------
1. Extract intensities from the masked region and flatten to 1-D.
2. Initialise cluster centres from percentiles of the intensity
   distribution (10th, 50th, 90th for *n* = 3).
3. Iterate until convergence:
   a. Compute squared distances from each sample to each centre.
   b. Update the membership matrix::

          u_ij = 1 / Σ_k (d_ij / d_ik) ^ (2 / (m - 1))

   c. Update centres::

          c_j = Σ (u_ij^m · x_i) / Σ (u_ij^m)

   d. Stop when max |ΔU| < tolerance.
4. Identify the *laser* cluster as the one whose centre is the
   median value (not background, not specular).
5. Reshape the laser membership back to ``(H, W)`` and return.

References
----------
.. [1] Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective
       Function Algorithms*. Springer.
.. [3] Chaira, T. & Ray, A. K. (2010). *Fuzzy Image Processing and
       Applications with MATLAB*. CRC Press.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FuzzyCMeans:
    """Fuzzy C-Means clustering for laser stripe segmentation.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.  Typically 3 (background, laser, specular).
    m : float
        Fuzziness exponent.  Must be > 1.  Standard choice is 2.0.
    max_iter : int
        Maximum number of FCM iterations.
    tol : float
        Convergence tolerance on the membership matrix (L∞ norm).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        if m <= 1.0:
            raise ValueError(f"Fuzziness exponent m must be > 1, got {m}")
        if n_clusters < 2:
            raise ValueError(
                f"Need at least 2 clusters, got {n_clusters}"
            )

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

        # Populated after fitting
        self.centers_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_stripe(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Cluster pixel intensities and return the laser membership map.

        Parameters
        ----------
        image : np.ndarray
            Single-channel image of shape ``(H, W)``, any numeric dtype.
        mask : np.ndarray, optional
            Binary mask ``(H, W)`` — nonzero entries mark pixels to
            include.  If *None*, all pixels are used.

        Returns
        -------
        np.ndarray
            Membership map of shape ``(H, W)``, dtype ``float64``,
            with values in [0, 1] representing the degree of
            membership in the *laser* cluster.  Pixels outside the
            mask are set to 0.

        Raises
        ------
        ValueError
            If *image* is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(
                f"Expected 2-D image, got shape {image.shape}"
            )

        H, W = image.shape
        image_f = image.astype(np.float64, copy=False)

        # Build mask indices
        if mask is not None:
            mask_bool = mask.astype(bool, copy=False)
        else:
            mask_bool = np.ones((H, W), dtype=bool)

        indices = np.nonzero(mask_bool)
        n_pixels = indices[0].size

        # --- Edge cases ---
        if n_pixels == 0:
            logger.warning("Empty mask — returning zero membership map.")
            return np.zeros((H, W), dtype=np.float64)

        x = image_f[indices]  # (N,)

        if np.all(x == x[0]):
            logger.warning(
                "All masked pixels have the same intensity (%.1f). "
                "Returning uniform membership.",
                x[0],
            )
            membership_map = np.zeros((H, W), dtype=np.float64)
            membership_map[indices] = 1.0 / self.n_clusters
            return membership_map

        # --- Run FCM ---
        centers, U = self._fit(x)
        self.centers_ = centers

        # --- Identify the laser cluster (middle centre value) ---
        laser_idx = self._find_laser_cluster(centers)
        laser_membership = U[laser_idx]  # (N,)

        # Reconstruct the spatial map
        membership_map = np.zeros((H, W), dtype=np.float64)
        membership_map[indices] = laser_membership

        logger.debug(
            "FCM converged in %d iterations. Centres: %s. Laser cluster: %d.",
            self.n_iter_,
            np.array2string(centers, precision=1),
            laser_idx,
        )
        return membership_map

    # ------------------------------------------------------------------
    # Core FCM algorithm
    # ------------------------------------------------------------------

    def _fit(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the FCM optimisation loop.

        Parameters
        ----------
        x : np.ndarray
            1-D array of pixel intensities, shape ``(N,)``.

        Returns
        -------
        centers : np.ndarray
            Final cluster centres, shape ``(C,)``.
        U : np.ndarray
            Final membership matrix, shape ``(C, N)``.
        """
        C = self.n_clusters
        N = x.size
        power = 2.0 / (self.m - 1.0)

        # Initialise centres from evenly spaced percentiles
        percentiles = np.linspace(10, 90, C)
        centers = np.percentile(x, percentiles)  # (C,)

        U = np.zeros((C, N), dtype=np.float64)

        for iteration in range(self.max_iter):
            # 3a. Squared distances: (C, N)
            diff = centers[:, np.newaxis] - x[np.newaxis, :]  # (C, N)
            dist_sq = diff ** 2 + 1e-12  # avoid exact zero

            # 3b. Update memberships
            #     u_ij = 1 / Σ_k (d_ij / d_ik)^power
            # Reshape for broadcasting: dist_sq is (C, N)
            # Ratio matrix: (C, C, N) → sum over k → (C, N)
            # More efficient: compute each u_ij directly.
            #   ratio_ijk = (dist_sq[j] / dist_sq[k])^(power/2)
            # but dist_sq already squared, so (d_ij/d_ik)^2 = dist_sq[j]/dist_sq[k]
            # exponent on the ratio is 2/(m-1), applied to d_ij/d_ik,
            # i.e. (dist_sq[j]/dist_sq[k])^(1/(m-1))
            inv_power = 1.0 / (self.m - 1.0)
            U_new = np.zeros_like(U)
            # Vectorised: for each cluster j, sum_k (dist_sq[j]/dist_sq[k])^inv_power
            for j in range(C):
                ratio_sum = np.sum(
                    (dist_sq[j:j+1, :] / dist_sq) ** inv_power,
                    axis=0,
                )  # (N,)
                U_new[j] = 1.0 / ratio_sum

            # 3d. Convergence check
            delta = np.max(np.abs(U_new - U))
            U = U_new

            # 3c. Update centres
            Um = U ** self.m  # (C, N)
            centers = np.dot(Um, x) / np.sum(Um, axis=1)  # (C,)

            if delta < self.tol:
                self.n_iter_ = iteration + 1
                return centers, U

        self.n_iter_ = self.max_iter
        logger.debug(
            "FCM did not converge within %d iterations (delta=%.2e).",
            self.max_iter,
            delta,
        )
        return centers, U

    # ------------------------------------------------------------------
    # Cluster identification
    # ------------------------------------------------------------------

    @staticmethod
    def _find_laser_cluster(centers: np.ndarray) -> int:
        """Identify the laser cluster as the one with the median centre.

        For 3 clusters the ordering is typically:
            background (low) < laser (mid) < specular (high).

        For *n* clusters the cluster whose centre rank is ⌊n/2⌋ is
        selected.

        Parameters
        ----------
        centers : np.ndarray
            Cluster centre values, shape ``(C,)``.

        Returns
        -------
        int
            Index into *centers* for the laser cluster.
        """
        sorted_indices = np.argsort(centers)
        median_rank = len(centers) // 2  # middle for odd, upper-mid for even
        return int(sorted_indices[median_rank])
