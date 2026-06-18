"""
Full Coordinate Transformation Chain: Pixel → Camera → Robot → World.

Orchestrates the complete transformation pipeline from 2D detected
laser pixels to 3D robot coordinates, wrapping ``LaserTriangulator``
with an optional hand-eye calibration transform.

The chain is:
    pixel (u, v)  ──undistort──▷  normalised coords  ──ray-plane──▷
    camera frame  ──T_cam2robot──▷  robot / world frame

References:
    [1] Hartley, R. & Zisserman, A. (2004). Multiple View Geometry
        in Computer Vision. Cambridge University Press. 2nd Edition.
    [2] Tsai, R. Y. & Lenz, R. K. (1989). A New Technique for Fully
        Autonomous and Efficient 3-D Robotics Hand/Eye Calibration.
        IEEE Trans. Robotics and Automation, 5(3), 345-358.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from geometry.triangulation import LaserTriangulator


class CoordinateChain:
    """End-to-end pixel → robot coordinate transformer.

    Parameters
    ----------
    camera_matrix : np.ndarray
        3×3 camera intrinsic matrix K.
    dist_coeffs : np.ndarray
        OpenCV-format distortion coefficients.
    plane_coeffs : np.ndarray | list
        Laser plane [A, B, C, D] in camera frame.
    T_cam2robot : np.ndarray | None
        Optional 4×4 camera-to-robot homogeneous transform.  Can be
        supplied later via :py:meth:`set_robot_transform`.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        plane_coeffs: np.ndarray,
        T_cam2robot: Optional[np.ndarray] = None,
    ) -> None:
        self._triangulator = LaserTriangulator(
            camera_matrix, dist_coeffs, plane_coeffs
        )
        self._T_cam2robot: Optional[np.ndarray] = None
        if T_cam2robot is not None:
            self.set_robot_transform(T_cam2robot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_robot_transform(self, T_cam2robot: np.ndarray) -> None:
        """Update (or set) the hand-eye calibration transform.

        Parameters
        ----------
        T_cam2robot : np.ndarray
            4×4 homogeneous transform from camera frame to robot frame.
            Must satisfy Tsai & Lenz [2] AX = XB calibration or
            equivalent.
        """
        T = np.asarray(T_cam2robot, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T_cam2robot must be a 4×4 matrix.")
        self._T_cam2robot = T

    def pixel_to_robot(self, u: float, v: float) -> np.ndarray:
        """Full chain: single pixel (u, v) → robot-frame 3D point.

        Parameters
        ----------
        u, v : float
            Pixel coordinates.

        Returns
        -------
        np.ndarray
            3D point in robot frame, shape (3,).
        """
        p_cam = self._triangulator.pixel_to_3d(u, v)  # (3,)

        if self._T_cam2robot is not None:
            p_robot = LaserTriangulator.transform_to_robot(
                p_cam.reshape(1, 3), self._T_cam2robot
            )
            return p_robot.ravel()
        return p_cam

    def pixels_to_robot_batch(self, pixels: np.ndarray) -> np.ndarray:
        """Full chain: batch pixels (Nx2) → robot-frame 3D points (Nx3).

        This is fully vectorized — no Python-level loops.

        Parameters
        ----------
        pixels : np.ndarray
            Nx2 array of pixel coordinates [[u1, v1], ...].

        Returns
        -------
        np.ndarray
            Nx3 array of 3D points in robot (or camera) frame.
        """
        points_cam = self._triangulator.pixels_to_3d_batch(pixels)  # (N, 3)

        if self._T_cam2robot is not None:
            return LaserTriangulator.transform_to_robot(
                points_cam, self._T_cam2robot
            )
        return points_cam

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_path(
        self,
        robot_coords: np.ndarray,
        filepath: Union[str, Path],
        format: str = "json",  # noqa: A002 — shadows builtin, but matches spec
    ) -> None:
        """Export robot-frame coordinates to a file.

        Parameters
        ----------
        robot_coords : np.ndarray
            Nx3 array of 3D points.
        filepath : str | Path
            Output file path.
        format : str
            ``'json'`` or ``'csv'``.
        """
        filepath = Path(filepath)
        robot_coords = np.asarray(robot_coords, dtype=np.float64)
        if robot_coords.ndim == 1:
            robot_coords = robot_coords.reshape(1, 3)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            data = {
                "num_points": int(robot_coords.shape[0]),
                "points": [
                    {"x": float(row[0]), "y": float(row[1]), "z": float(row[2])}
                    for row in robot_coords
                ],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format.lower() == "csv":
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                for row in robot_coords:
                    writer.writerow([float(row[0]), float(row[1]), float(row[2])])

        else:
            raise ValueError(f"Unsupported export format '{format}'. Use 'json' or 'csv'.")
