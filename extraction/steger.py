"""
Steger's Algorithm for Sub-Pixel Laser Stripe Center Extraction.

Uses second-order derivatives (Hessian matrix) to find ridge lines
in intensity images with sub-pixel accuracy.

At each pixel the 2×2 Hessian matrix of the image intensity is
formed from second-order Gaussian derivatives:

    H = [[Ixx, Ixy],
         [Ixy, Iyy]]

The eigenvector corresponding to the eigenvalue of largest absolute
value gives the direction *perpendicular* to the ridge.  A Taylor
expansion along this direction yields the sub-pixel offset:

    t = -(nx·Ix + ny·Iy) / (nx²·Ixx + 2·nx·ny·Ixy + ny²·Iyy)

A pixel is accepted as a ridge point when:
    * |t| < 0.5   (the offset lies within the pixel)
    * The larger eigenvalue exceeds a strength threshold.

All operations are fully vectorised with NumPy — no per-pixel Python
loops.

References:
    [1] Steger, C. (1998). An unbiased detector of curvilinear structures.
        IEEE TPAMI, 20(2), 113-125.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StegerExtractor:
    """Sub-pixel ridge (laser-stripe centre) detector using Hessian analysis.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel used to compute
        image derivatives.  Controls the scale of the detected ridges.
    low_thresh : float
        Lower threshold on the absolute eigenvalue for ridge acceptance.
    high_thresh : float
        Upper threshold (not used for rejection — reserved for hysteresis
        linking in future extensions).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        low_thresh: float = 0.5,
        high_thresh: float = 2.0,
    ) -> None:
        self.sigma = sigma
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    # ------------------------------------------------------------------
    # Gaussian derivative computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_derivatives(
        image: np.ndarray, sigma: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute first- and second-order Gaussian image derivatives.

        A Gaussian blur at scale *sigma* is applied first, then Sobel
        operators approximate the partial derivatives.

        Parameters
        ----------
        image : np.ndarray
            Single-channel float64 image.
        sigma : float
            Gaussian kernel standard deviation.

        Returns
        -------
        Ix, Iy : np.ndarray
            First-order partial derivatives.
        Ixx, Ixy, Iyy : np.ndarray
            Second-order partial derivatives.
        """
        # Gaussian kernel size: must be odd and ≥ 6σ+1 for adequate support
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd

        # Smooth first to set the derivative scale (Steger [1], §3)
        smoothed = cv2.GaussianBlur(
            image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma
        )

        # First-order derivatives via Sobel (kernel size 3)
        Ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

        # Second-order derivatives
        Ixx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)

        return Ix, Iy, Ixx, Ixy, Iyy

    # ------------------------------------------------------------------
    # Hessian eigenvalue / eigenvector analysis (fully vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hessian_eigenvalues(
        Ixx: np.ndarray, Ixy: np.ndarray, Iyy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Hessian eigenvalues and the principal eigenvector direction.

        For a symmetric 2×2 matrix [[a, b], [b, c]] the eigenvalues are:

            λ = 0.5 * (a + c ± sqrt((a - c)² + 4b²))

        The eigenvector for the larger |λ| gives the direction
        perpendicular to the ridge (nx, ny).

        Parameters
        ----------
        Ixx, Ixy, Iyy : np.ndarray
            Second-order derivative images (same shape).

        Returns
        -------
        eigenval_max : np.ndarray
            Absolute-value-larger eigenvalue at each pixel (ridge strength).
        nx, ny : np.ndarray
            Components of the unit eigenvector associated with *eigenval_max*.
        discriminant : np.ndarray
            sqrt((Ixx-Iyy)² + 4·Ixy²) — useful for diagnostics.
        """
        # Discriminant of the characteristic equation
        diff = Ixx - Iyy
        discriminant = np.sqrt(diff ** 2 + 4.0 * Ixy ** 2)

        # Two eigenvalues
        half_sum = 0.5 * (Ixx + Iyy)
        lambda1 = half_sum + 0.5 * discriminant
        lambda2 = half_sum - 0.5 * discriminant

        # Select the eigenvalue with the larger absolute value
        abs1 = np.abs(lambda1)
        abs2 = np.abs(lambda2)
        use_first = abs1 >= abs2

        eigenval_max = np.where(use_first, lambda1, lambda2)

        # Eigenvector for the selected eigenvalue:
        # For [[a, b],[b, c]], eigenvector for λ is proportional to
        #   (b, λ - a)  or equivalently  (λ - c, b)
        # We use (Ixy, λ - Ixx) and normalise.
        raw_nx = Ixy
        raw_ny = np.where(use_first, lambda1 - Ixx, lambda2 - Ixx)

        norm = np.sqrt(raw_nx ** 2 + raw_ny ** 2)
        # Avoid division by zero where the Hessian is isotropic
        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        nx = raw_nx / safe_norm
        ny = raw_ny / safe_norm

        # Enforce eigenvector sign convention: nx >= 0
        # This ensures all profiles are sampled in a consistent direction
        flip_mask = nx < 0
        nx = np.where(flip_mask, -nx, nx)
        ny = np.where(flip_mask, -ny, ny)

        return eigenval_max, nx, ny, discriminant

    # ------------------------------------------------------------------
    # Sub-pixel centre extraction
    # ------------------------------------------------------------------

    def extract_centers(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract sub-pixel ridge centres from a single-channel image.

        Implements the full Steger pipeline [1]:

        1. Compute Gaussian image derivatives (Ix, Iy, Ixx, Ixy, Iyy).
        2. At every pixel, build the Hessian and find the eigenvector
           (nx, ny) of the largest-magnitude eigenvalue.
        3. Compute the sub-pixel offset along (nx, ny):

               t = -(nx·Ix + ny·Iy) / (nx²·Ixx + 2·nx·ny·Ixy + ny²·Iyy)

        4. Accept the pixel if |t| < 0.5 **and** the eigenvalue magnitude
           exceeds ``low_thresh``.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale, uint8 or float).
        mask : Optional[np.ndarray]
            Binary mask (same HxW). Only pixels where mask != 0 are evaluated.

        Returns
        -------
        x_coords : np.ndarray
            Sub-pixel x (column) coordinates of ridge points.
        y_coords : np.ndarray
            Sub-pixel y (row) coordinates of ridge points.
        strengths : np.ndarray
            Eigenvalue magnitude (ridge strength) at each point.
        """
        # Ensure float64 grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image.astype(np.float64)

        # Step 1: derivatives
        Ix, Iy, Ixx, Ixy, Iyy = self._compute_derivatives(img, self.sigma)

        # Step 2: Hessian eigenvalues / eigenvectors
        eigenval_max, nx, ny, _ = self._compute_hessian_eigenvalues(Ixx, Ixy, Iyy)

        # Step 3: sub-pixel offset along the principal direction
        #   t = -(nx·Ix + ny·Iy) / (nx²·Ixx + 2·nx·ny·Ixy + ny²·Iyy)
        numerator = -(nx * Ix + ny * Iy)
        denominator = nx ** 2 * Ixx + 2.0 * nx * ny * Ixy + ny ** 2 * Iyy

        # Guard against division by zero
        safe_denom = np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
        t = numerator / safe_denom

        # Step 4: acceptance criteria
        valid = np.abs(t) <= 0.5                            # sub-pixel constraint (ridge is within pixel)
        valid &= np.abs(eigenval_max) > self.low_thresh     # ridge strength
        valid &= eigenval_max < 0                           # polarity check (ridge is a maximum, not a valley)
        if mask is not None:
            valid &= mask.astype(bool)

        # Pixel grid coordinates
        rows, cols = np.where(valid)

        # Sub-pixel positions: shift the integer pixel by t·(nx, ny)
        x_sub = cols.astype(np.float64) + t[valid] * nx[valid]
        y_sub = rows.astype(np.float64) + t[valid] * ny[valid]
        strengths = np.abs(eigenval_max[valid])

        logger.info("Steger extraction: %d ridge points found.", len(x_sub))
        return x_sub, y_sub, strengths

    # ------------------------------------------------------------------
    # Single-column extraction (compatibility API)
    # ------------------------------------------------------------------

    def extract_column_center(
        self,
        column_profile: np.ndarray,
        x_coord: float,
    ) -> Tuple[float, float]:
        """Extract the ridge centre from a single column intensity profile.

        This is a convenience wrapper for the fuzzy/adaptive pipeline that
        works on individual column profiles rather than full 2-D images.

        Parameters
        ----------
        column_profile : np.ndarray
            1-D intensity profile along one image column.
        x_coord : float
            The column (x) coordinate associated with this profile.

        Returns
        -------
        y_center : float
            Sub-pixel row coordinate of the ridge centre, or ``NaN`` if
            no valid ridge is found.
        confidence : float
            Ridge strength (absolute eigenvalue), or 0.0 on failure.
        """
        # Reshape the 1-D profile into a single-column image (H, 1)
        col_img = column_profile.astype(np.float64).reshape(-1, 1)

        x_sub, y_sub, strengths = self.extract_centers(col_img)

        if len(y_sub) == 0:
            return float("nan"), 0.0

        # If multiple ridge points found, pick the strongest
        best_idx = np.argmax(strengths)
        return float(y_sub[best_idx]), float(strengths[best_idx])
