"""
Profile Sampler — Intensity Profiles along Steger Normal Directions.

Bridges Steger ridge detection and IT2FLS processing by sampling
cross-ridge intensity profiles using bilinear interpolation.

For each detected ridge point ``(x_i, y_i)`` with outward normal
``(nx_i, ny_i)``, the sampler evaluates the image intensity at
equidistant positions along the normal, producing a 1-D profile
that captures the laser stripe cross-section.

All sampling is fully vectorised with NumPy broadcasting — a single
batch bilinear interpolation covers **all** ridge points and sample
offsets simultaneously, avoiding any Python-level loop over pixels.

Coordinate convention
---------------------
Image coordinates follow OpenCV / NumPy convention:

- ``x`` → column index (horizontal)
- ``y`` → row index (vertical)
- ``image[y, x]`` selects the intensity value.

Bilinear interpolation formula::

    I(px, py) = (1 - dx)(1 - dy) · I[y0, x0]
              + dx · (1 - dy)    · I[y0, x0+1]
              + (1 - dx) · dy    · I[y0+1, x0]
              + dx · dy           · I[y0+1, x0+1]

where ``x0 = ⌊px⌋``, ``y0 = ⌊py⌋``, ``dx = px − x0``,
``dy = py − y0``.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProfileSampler:
    """Sample intensity profiles along ridge normals via bilinear interpolation.

    Parameters
    ----------
    half_width : int
        Number of pixels to sample on each side of the ridge point
        along the normal direction.
    step : float
        Sub-pixel step size between consecutive samples along the
        normal.

    Examples
    --------
    >>> sampler = ProfileSampler(half_width=15, step=0.5)
    >>> profiles, s = sampler.sample_profiles(img, rx, ry, nx, ny)
    >>> profiles.shape  # (N_ridge_points, M_samples)
    >>> s.shape         # (M_samples,)
    """

    def __init__(self, half_width: int = 15, step: float = 0.5) -> None:
        if half_width < 1:
            raise ValueError(
                f"half_width must be >= 1, got {half_width}"
            )
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")

        self.half_width = half_width
        self.step = step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_profiles(
        self,
        image: np.ndarray,
        ridge_x: np.ndarray,
        ridge_y: np.ndarray,
        normals_x: np.ndarray,
        normals_y: np.ndarray,
        half_width: int | None = None,
        step: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample cross-ridge intensity profiles for all ridge points.

        Parameters
        ----------
        image : np.ndarray
            Single-channel image of shape ``(H, W)``, any numeric dtype.
        ridge_x : np.ndarray
            Sub-pixel column coordinates of ridge points, shape ``(N,)``.
        ridge_y : np.ndarray
            Sub-pixel row coordinates of ridge points, shape ``(N,)``.
        normals_x : np.ndarray
            X-component of the unit normal at each ridge point, ``(N,)``.
        normals_y : np.ndarray
            Y-component of the unit normal at each ridge point, ``(N,)``.
        half_width : int, optional
            Override the instance ``half_width`` for this call.
        step : float, optional
            Override the instance ``step`` for this call.

        Returns
        -------
        profiles : np.ndarray
            Sampled intensity profiles, shape ``(N, M)`` where *N* is
            the number of ridge points and *M* is the number of samples
            per profile.  dtype ``float64``.
        s_coords : np.ndarray
            Spatial offset coordinates along the normal direction,
            shape ``(M,)``.  Negative values are on one side of the
            ridge, positive on the other, and 0 is the ridge centre.

        Raises
        ------
        ValueError
            If input arrays have inconsistent shapes or image is not 2-D.
        """
        # --- Resolve parameters ---
        hw = half_width if half_width is not None else self.half_width
        ds = step if step is not None else self.step

        # --- Input validation ---
        if image.ndim != 2:
            raise ValueError(
                f"Expected 2-D image, got shape {image.shape}"
            )

        ridge_x = np.asarray(ridge_x, dtype=np.float64).ravel()
        ridge_y = np.asarray(ridge_y, dtype=np.float64).ravel()
        normals_x = np.asarray(normals_x, dtype=np.float64).ravel()
        normals_y = np.asarray(normals_y, dtype=np.float64).ravel()

        N = ridge_x.size
        if not (ridge_y.size == normals_x.size == normals_y.size == N):
            raise ValueError(
                f"All ridge/normal arrays must have the same length. "
                f"Got ridge_x={N}, ridge_y={ridge_y.size}, "
                f"normals_x={normals_x.size}, normals_y={normals_y.size}."
            )

        if N == 0:
            logger.warning("No ridge points provided — returning empty profiles.")
            s_coords = np.arange(-hw, hw + ds, ds)
            return np.empty((0, s_coords.size), dtype=np.float64), s_coords

        # --- Build sample offset array ---
        s_coords = np.arange(-hw, hw + ds, ds)  # (M,)
        M = s_coords.size

        # --- Compute sample coordinates (vectorised) ---
        # ridge_x: (N,), s_coords: (M,), normals_x: (N,)
        # px[i, j] = ridge_x[i] + s_coords[j] * normals_x[i]
        px = ridge_x[:, np.newaxis] + s_coords[np.newaxis, :] * normals_x[:, np.newaxis]  # (N, M)
        py = ridge_y[:, np.newaxis] + s_coords[np.newaxis, :] * normals_y[:, np.newaxis]  # (N, M)

        # --- Batch bilinear interpolation ---
        profiles = self._bilinear_interpolate(image, px, py)

        logger.debug(
            "Sampled %d profiles × %d samples (hw=%d, step=%.2f).",
            N, M, hw, ds,
        )
        return profiles, s_coords

    # ------------------------------------------------------------------
    # Bilinear interpolation (vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _bilinear_interpolate(
        image: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
    ) -> np.ndarray:
        """Batch bilinear interpolation at sub-pixel coordinates.

        Parameters
        ----------
        image : np.ndarray
            Source image, shape ``(H, W)``.
        px : np.ndarray
            Column (x) coordinates, arbitrary shape (broadcast-safe).
        py : np.ndarray
            Row (y) coordinates, same shape as *px*.

        Returns
        -------
        np.ndarray
            Interpolated intensities, same shape as *px*, dtype
            ``float64``.

        Notes
        -----
        Out-of-bounds coordinates are clamped to the nearest valid
        pixel (border replication).
        """
        H, W = image.shape
        img = image.astype(np.float64, copy=False)

        # Clamp to valid interpolation range [0, dim-1]
        px_c = np.clip(px, 0.0, W - 1.0)
        py_c = np.clip(py, 0.0, H - 1.0)

        # Integer parts (floor)
        x0 = np.floor(px_c).astype(np.intp)
        y0 = np.floor(py_c).astype(np.intp)

        # Ensure x0+1 and y0+1 stay within bounds
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)

        # Fractional parts
        dx = px_c - x0.astype(np.float64)
        dy = py_c - y0.astype(np.float64)

        # Gather four neighbours (advanced indexing)
        Ia = img[y0, x0]      # top-left
        Ib = img[y0, x1]      # top-right
        Ic = img[y1, x0]      # bottom-left
        Id = img[y1, x1]      # bottom-right

        # Bilinear combination
        result = (
            (1.0 - dx) * (1.0 - dy) * Ia
            + dx * (1.0 - dy) * Ib
            + (1.0 - dx) * dy * Ic
            + dx * dy * Id
        )
        return result
