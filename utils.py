"""
Utility functions for the Weld Seam Detection Pipeline.

Provides evaluation metrics (Dice, IoU, Boundary-F1), reproducibility
helpers, visualisation routines, and lightweight I/O for experiment
tracking.

References:
    [1] Milletari, F., Navab, N. & Ahmadi, S.-A. (2016).
        "V-Net: Fully Convolutional Neural Networks for Volumetric
        Medical Image Segmentation."  3DV, pp. 565-571.
        — Dice coefficient formulation.
    [2] Csurka, G. et al. (2013).
        "What is a good evaluation measure for semantic segmentation?"
        BMVC.
        — Boundary-F1 metric.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — side-effect import
from scipy.ndimage import binary_dilation


# -----------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------

def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Compute the Sørensen–Dice coefficient [1].

    .. math::
        \\text{Dice} = \\frac{2 |P \\cap T|}{|P| + |T|}

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask (H, W), values in {0, 1}.
    target : np.ndarray
        Binary ground-truth mask (H, W), values in {0, 1}.
    smooth : float
        Laplace smoothing to avoid division by zero.

    Returns
    -------
    float
        Dice score in [0, 1].
    """
    pred_flat = pred.astype(np.float64).ravel()
    target_flat = target.astype(np.float64).ravel()

    intersection = np.dot(pred_flat, target_flat)
    return float(
        (2.0 * intersection + smooth)
        / (pred_flat.sum() + target_flat.sum() + smooth)
    )


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection-over-Union (Jaccard Index).

    .. math::
        \\text{IoU} = \\frac{|P \\cap T|}{|P \\cup T|}

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask (H, W).
    target : np.ndarray
        Binary ground-truth mask (H, W).
    smooth : float
        Laplace smoothing.

    Returns
    -------
    float
        IoU score in [0, 1].
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    union = np.logical_or(pred_bool, target_bool).sum()
    return float((intersection + smooth) / (union + smooth))


def boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Boundary F1 score (BF1) [2].

    Extracts contour pixels from both masks, then computes precision
    and recall with a spatial tolerance (dilation radius).

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask (H, W).
    target : np.ndarray
        Binary ground-truth mask (H, W).
    tolerance : int
        Pixel tolerance for boundary matching (dilation radius).

    Returns
    -------
    float
        Boundary F1 score in [0, 1].
    """
    # Extract boundary pixels via morphological gradient
    struct = np.ones((3, 3), dtype=bool)

    pred_boundary = pred.astype(bool) ^ binary_dilation(
        pred.astype(bool), structure=struct, iterations=1
    )
    target_boundary = target.astype(bool) ^ binary_dilation(
        target.astype(bool), structure=struct, iterations=1
    )

    # Dilate boundaries by tolerance for soft matching
    pred_dilated = binary_dilation(
        pred_boundary, structure=struct, iterations=tolerance
    )
    target_dilated = binary_dilation(
        target_boundary, structure=struct, iterations=tolerance
    )

    # Count matched boundary pixels
    pred_boundary_sum = float(pred_boundary.sum())
    target_boundary_sum = float(target_boundary.sum())

    if pred_boundary_sum == 0 and target_boundary_sum == 0:
        return 1.0  # Both empty → perfect match

    # Precision: fraction of predicted boundary within tolerance of GT
    precision = (
        float(np.logical_and(pred_boundary, target_dilated).sum())
        / max(pred_boundary_sum, 1e-6)
    )

    # Recall: fraction of GT boundary within tolerance of prediction
    recall = (
        float(np.logical_and(target_boundary, pred_dilated).sum())
        / max(target_boundary_sum, 1e-6)
    )

    if precision + recall < 1e-8:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Seeds Python and NumPy random number generators.

    Parameters
    ----------
    seed : int
        Master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------

def overlay_seam(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask on an image with semi-transparent colour.

    Parameters
    ----------
    image : np.ndarray
        Base image, shape (H, W) or (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary mask, shape (H, W), values in {0, 1} or bool.
    color : tuple of int
        BGR colour for the overlay.
    alpha : float
        Blending factor in [0, 1] (0 = image only, 1 = mask only).

    Returns
    -------
    np.ndarray
        Blended BGR image, shape (H, W, 3), dtype uint8.
    """
    # Ensure 3-channel base
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    # Build solid-colour overlay where mask is active
    overlay = base.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    # Alpha-blend
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    return blended


def plot_profile(
    profile: np.ndarray,
    centers: Optional[np.ndarray] = None,
    title: str = 'Laser Stripe Profile',
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot a 1-D cross-section intensity profile with detected centres.

    Parameters
    ----------
    profile : np.ndarray
        1-D intensity array (length N).
    centers : np.ndarray or None
        Array of detected sub-pixel centre positions to mark.
    title : str
        Plot title.
    save_path : str or Path, optional
        If given, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile, linewidth=1.0, label='Intensity')

    if centers is not None and len(centers) > 0:
        # Mark each detected centre with a vertical dashed line
        for c in centers:
            ax.axvline(
                x=c, color='red', linestyle='--', linewidth=0.8, alpha=0.7
            )
        # Add a single legend entry for the centre markers
        ax.axvline(
            x=centers[0],
            color='red',
            linestyle='--',
            linewidth=0.8,
            alpha=0.0,
            label='Detected centre',
        )

    ax.set_xlabel('Pixel position')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_3d_path(
    coords_3d: np.ndarray,
    title: str = '3-D Weld Path',
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Scatter-plot a 3-D weld seam trajectory.

    Parameters
    ----------
    coords_3d : np.ndarray
        Array of shape (N, 3) with columns [X, Y, Z] in mm.
    title : str
        Plot title.
    save_path : str or Path, optional
        If given, save figure to this path instead of showing.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=coords_3d[:, 2],     # Colour by Z (height)
        cmap='viridis',
        s=2,
        alpha=0.8,
    )

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------

def save_metrics(
    metrics_dict: Dict[str, float],
    filepath: Union[str, Path],
) -> None:
    """Persist an evaluation metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics_dict : dict
        Mapping of metric names to scalar values.
    filepath : str or Path
        Destination JSON file path.  Parent directories are created
        automatically.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(metrics_dict, fh, indent=2, ensure_ascii=False)


def load_metrics(filepath: Union[str, Path]) -> Dict[str, float]:
    """Load an evaluation metrics dictionary from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Source JSON file path.

    Returns
    -------
    dict
        Mapping of metric names to scalar values.
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as fh:
        return json.load(fh)
