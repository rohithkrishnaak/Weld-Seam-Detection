"""
Visualization Utilities for Weld Seam Detection.

Provides overlay rendering, profile plots, and 3D path visualization
for diagnostic and publication-quality figures.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class Visualizer:
    """Visualization tools for seam detection results."""

    def draw_seam_overlay(
        self,
        image: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detected seam line on image with confidence coloring.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).
        x_coords, y_coords : np.ndarray
            Seam pixel coordinates.
        confidences : np.ndarray, optional
            Per-point confidence in [0, 1]. Green = high, red = low.
        color : tuple
            Fallback BGR color when confidences are not provided.
        thickness : int
            Line thickness in pixels.

        Returns
        -------
        np.ndarray
            BGR image with overlay.
        """
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if len(x_coords) < 2:
            return vis

        points = np.column_stack([x_coords, y_coords]).astype(np.int32)

        if confidences is not None and len(confidences) == len(x_coords):
            for i in range(len(points) - 1):
                conf = float(confidences[i])
                r = int(255 * (1 - conf))
                g = int(255 * conf)
                cv2.line(
                    vis, tuple(points[i]), tuple(points[i + 1]),
                    (0, g, r), thickness,
                )
        else:
            cv2.polylines(vis, [points], False, color, thickness)

        return vis

    def draw_method_map(
        self,
        image: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        methods: list,
        thickness: int = 2,
    ) -> np.ndarray:
        """Color-code seam by extraction method used.

        Color scheme:
        - Steger's (gaussian) → Green
        - Fuzzy Barycentric (flat_top) → Yellow
        - TFN (noisy) → Red
        """
        method_colors = {
            'steger': (0, 255, 0),
            'gaussian': (0, 255, 0),
            'fuzzy_barycentric': (0, 255, 255),
            'flat_top': (0, 255, 255),
            'tfn': (0, 0, 255),
            'noisy': (0, 0, 255),
        }
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        points = np.column_stack([x_coords, y_coords]).astype(np.int32)

        for i in range(len(points) - 1):
            m = methods[i] if i < len(methods) else 'steger'
            c = method_colors.get(m, (255, 255, 255))
            cv2.line(vis, tuple(points[i]), tuple(points[i + 1]), c, thickness)

        return vis

    def plot_laser_profile(
        self,
        profile: np.ndarray,
        center_y: Optional[float] = None,
        membership: Optional[np.ndarray] = None,
        title: str = "Laser Profile",
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Plot cross-sectional intensity profile with optional membership.

        Parameters
        ----------
        profile : np.ndarray
            1-D intensity array.
        center_y : float, optional
            Extracted center position.
        membership : np.ndarray, optional
            Fuzzy membership values to overlay.
        title : str
            Plot title.
        save_path : str, optional
            If provided, save figure to disk.

        Returns
        -------
        np.ndarray or None
            RGB image of the plot.
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.plot(profile, 'b-', linewidth=1.5, label='Intensity')

        if membership is not None:
            ax2 = ax1.twinx()
            ax2.plot(membership, 'g--', linewidth=1.0, alpha=0.7, label='μ')
            ax2.set_ylabel('Membership μ', color='g')
            ax2.set_ylim(-0.05, 1.1)

        if center_y is not None:
            ax1.axvline(
                center_y, color='r', linestyle='--',
                label=f'Center={center_y:.2f}',
            )

        ax1.set_xlabel('Pixel Position')
        ax1.set_ylabel('Intensity')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        return img

    def plot_3d_seam(
        self,
        coords_3d: np.ndarray,
        title: str = "3D Weld Path",
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Plot 3-D weld seam path colored by Z-height."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            c=coords_3d[:, 2], cmap='viridis', s=2,
        )
        ax.plot(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            'r-', linewidth=0.5, alpha=0.5,
        )
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        return img

    def create_dashboard(
        self,
        image: np.ndarray,
        seam_overlay: np.ndarray,
        method_map: np.ndarray,
        profile_plot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create a 2×2 dashboard layout combining multiple visualizations."""
        h, w = 400, 600

        def _resize(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (w, h))

        top = np.hstack([_resize(image), _resize(seam_overlay)])
        if profile_plot is not None:
            bottom = np.hstack([_resize(method_map), _resize(profile_plot)])
        else:
            bottom = np.hstack([
                _resize(method_map),
                np.zeros((h, w, 3), dtype=np.uint8),
            ])
        return np.vstack([top, bottom])
