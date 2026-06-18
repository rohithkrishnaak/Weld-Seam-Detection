"""
Profile Smoothing and Outlier Removal for Extracted Laser Centers.

Applies Savitzky-Golay filtering and statistical outlier removal
to produce a smooth, continuous path suitable for robot motion planning.

The pipeline consists of three stages:

1. **Outlier removal** — Z-score thresholding on the y-coordinates
   removes isolated spikes caused by specular reflections or detection
   artefacts.
2. **Gap interpolation** — Small gaps (missing columns) are filled
   with linear interpolation to maintain path continuity.
3. **Savitzky-Golay smoothing** — A polynomial least-squares filter
   suppresses high-frequency noise while preserving the shape of the
   weld seam profile.

References:
    [1] Savitzky, A. & Golay, M.J.E. (1964). Smoothing and
        differentiation of data by simplified least squares procedures.
        Analytical Chemistry, 36(8), 1627-1639.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def remove_outliers(
    x: np.ndarray,
    y: np.ndarray,
    z_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers from a laser-centre profile using Z-score filtering.

    A point is classified as an outlier if its *y*-coordinate deviates
    from the local mean by more than *z_threshold* standard deviations.

    Parameters
    ----------
    x : np.ndarray
        1-D array of column (x) coordinates.
    y : np.ndarray
        1-D array of row (y) centre coordinates.
    z_threshold : float
        Number of standard deviations beyond which a point is rejected.

    Returns
    -------
    x_clean : np.ndarray
        Filtered x-coordinates.
    y_clean : np.ndarray
        Filtered y-coordinates.
    """
    if len(y) == 0:
        return x.copy(), y.copy()

    mean_y = np.mean(y)
    std_y = np.std(y)

    if std_y < 1e-12:
        # Degenerate case: all points identical — nothing to remove
        return x.copy(), y.copy()

    # Z-score for each point
    z_scores = np.abs((y - mean_y) / std_y)
    inlier_mask = z_scores < z_threshold

    n_removed = int(np.sum(~inlier_mask))
    if n_removed > 0:
        logger.info("Removed %d outliers (Z > %.1f).", n_removed, z_threshold)

    return x[inlier_mask].copy(), y[inlier_mask].copy()


def smooth_path(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 11,
    order: int = 3,
) -> np.ndarray:
    """Smooth laser-centre y-coordinates with a Savitzky-Golay filter.

    The Savitzky-Golay filter [1] fits successive sub-sets of adjacent
    data points with a low-degree polynomial by least squares, providing
    noise reduction without distorting the signal shape.

    Parameters
    ----------
    x : np.ndarray
        1-D array of column coordinates (used only for length validation).
    y : np.ndarray
        1-D array of row centre coordinates to smooth.
    window : int
        Window length (must be odd and > order).
    order : int
        Polynomial order for the filter.

    Returns
    -------
    y_smooth : np.ndarray
        Smoothed y-coordinates (same length as *y*).
    """
    if len(y) < window:
        # Not enough points for the requested window — fall back to
        # a smaller window or return unmodified.
        fallback_window = max(order + 2, 3)
        if fallback_window % 2 == 0:
            fallback_window += 1
        if len(y) < fallback_window:
            logger.warning(
                "Too few points (%d) for Savitzky-Golay smoothing.", len(y)
            )
            return y.copy()
        window = fallback_window
        logger.info("Reduced Savitzky-Golay window to %d.", window)

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    return savgol_filter(y, window_length=window, polyorder=order)


def interpolate_gaps(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill small gaps in the extracted profile via linear interpolation.

    A "gap" is a range of consecutive missing column indices between two
    existing points.  Only gaps of width ≤ *max_gap* are filled;
    larger gaps are left empty to avoid hallucinating data over large
    occlusions.

    Parameters
    ----------
    x : np.ndarray
        1-D array of existing column coordinates (must be integer-like).
    y : np.ndarray
        1-D array of corresponding y-centres.
    max_gap : int
        Maximum number of consecutive missing columns to interpolate.

    Returns
    -------
    x_filled : np.ndarray
        Column coordinates with gaps filled.
    y_filled : np.ndarray
        Interpolated y-centres.
    """
    if len(x) < 2:
        return x.copy(), y.copy()

    # Round x to integer column indices
    x_int = np.round(x).astype(int)

    # Determine the full range of columns
    x_min, x_max = x_int.min(), x_int.max()
    full_x = np.arange(x_min, x_max + 1)

    # Map existing values onto the full grid
    y_full = np.full(full_x.shape, np.nan)
    # Use the relative index within the full range
    indices = x_int - x_min
    y_full[indices] = y

    # Identify gap runs and interpolate only short ones
    is_nan = np.isnan(y_full)
    # Label connected NaN regions
    diff = np.diff(is_nan.astype(int))
    gap_starts = np.where(diff == 1)[0] + 1   # start of NaN run
    gap_ends = np.where(diff == -1)[0] + 1     # first non-NaN after run

    # Handle edge cases where the profile starts or ends with NaN
    if is_nan[0]:
        gap_starts = np.insert(gap_starts, 0, 0)
    if is_nan[-1]:
        gap_ends = np.append(gap_ends, len(y_full))

    for gs, ge in zip(gap_starts, gap_ends):
        gap_length = ge - gs
        if gap_length > max_gap:
            continue
        # Need valid neighbours on both sides for linear interpolation
        if gs == 0 or ge == len(y_full):
            continue
        # Linear interpolation between boundary values
        y_left = y_full[gs - 1]
        y_right = y_full[ge]
        y_full[gs:ge] = np.linspace(y_left, y_right, gap_length + 2)[1:-1]

    # Return only filled positions
    filled_mask = ~np.isnan(y_full)
    return full_x[filled_mask].astype(np.float64), y_full[filled_mask]


def full_smoothing_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    z_threshold: float = 3.0,
    window: int = 11,
    order: int = 3,
    max_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the complete smoothing pipeline on extracted laser centres.

    Stages:
    1. Z-score outlier removal.
    2. Linear gap interpolation.
    3. Savitzky-Golay smoothing.

    Parameters
    ----------
    x, y : np.ndarray
        Raw extracted centre coordinates.
    z_threshold : float
        Z-score threshold for outlier removal.
    window : int
        Savitzky-Golay window length.
    order : int
        Savitzky-Golay polynomial order.
    max_gap : int
        Maximum gap width (in columns) to interpolate.

    Returns
    -------
    x_out : np.ndarray
        Processed x-coordinates.
    y_out : np.ndarray
        Processed y-coordinates.
    """
    logger.info(
        "Smoothing pipeline: %d input points, z=%.1f, window=%d, order=%d, "
        "max_gap=%d.",
        len(x), z_threshold, window, order, max_gap,
    )

    # Step 1: outlier removal
    x_clean, y_clean = remove_outliers(x, y, z_threshold)

    # Step 2: gap interpolation
    x_filled, y_filled = interpolate_gaps(x_clean, y_clean, max_gap)

    # Step 3: Savitzky-Golay smoothing
    y_smooth = smooth_path(x_filled, y_filled, window, order)

    logger.info(
        "Smoothing pipeline complete: %d → %d points.", len(x), len(x_filled)
    )
    return x_filled, y_smooth
