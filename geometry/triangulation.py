"""
2D-to-3D Triangulation via Ray-Plane Intersection.

Converts detected laser center pixels (u, v) to 3D points in camera
frame by intersecting the camera ray with the calibrated laser plane.

The laser plane is parameterized as Ax + By + Cz + D = 0 in camera
coordinates. Each pixel (u, v) defines a ray from the camera origin
through the image plane; the intersection of this ray with the laser
plane yields the 3D point.

References:
    [1] Hartley, R. & Zisserman, A. (2004). Multiple View Geometry
        in Computer Vision. Cambridge University Press. 2nd Edition.
    [2] Zhang, Z. (2000). A Flexible New Technique for Camera
        Calibration. IEEE TPAMI, 22(11), 1330-1334.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class LaserTriangulator:
    """Triangulates 2D laser stripe pixels to 3D camera-frame points.

    Given the camera intrinsics (K), lens distortion coefficients, and
    a calibrated laser plane equation Ax + By + Cz + D = 0, this class
    converts pixel coordinates to 3D points via ray-plane intersection.

    Parameters
    ----------
    camera_matrix : np.ndarray
        3×3 camera intrinsic matrix K.
    dist_coeffs : np.ndarray
        Distortion coefficients in OpenCV format (k1, k2, p1, p2[, k3 ...]).
    plane_coeffs : np.ndarray | list
        Laser plane coefficients [A, B, C, D] such that
        Ax + By + Cz + D = 0 in the camera coordinate frame.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        plane_coeffs: np.ndarray,
    ) -> None:
        # Store the 3×3 intrinsic matrix
        self.K: np.ndarray = np.asarray(camera_matrix, dtype=np.float64)
        # Distortion coefficients (OpenCV convention)
        self.dist: np.ndarray = np.asarray(dist_coeffs, dtype=np.float64)
        # Laser plane: Ax + By + Cz + D = 0
        self.plane: np.ndarray = np.asarray(plane_coeffs, dtype=np.float64)
        if self.plane.shape != (4,):
            raise ValueError("plane_coeffs must have exactly 4 elements [A, B, C, D]")

        # Pre-extract plane normal [A, B, C] and offset D for clarity
        self.A: float = float(self.plane[0])
        self.B: float = float(self.plane[1])
        self.C: float = float(self.plane[2])
        self.D: float = float(self.plane[3])

    # ------------------------------------------------------------------
    # Single-pixel triangulation
    # ------------------------------------------------------------------
    def pixel_to_3d(self, u: float, v: float) -> np.ndarray:
        """Convert a single pixel (u, v) to a 3D point in camera frame.

        Steps (following Hartley & Zisserman [1], §6.2):
            1. Undistort the pixel to obtain normalized image coordinates.
            2. Form the camera ray direction d = [x_n, y_n, 1].
            3. Compute the ray parameter λ = -D / (A·x_n + B·y_n + C).
            4. The 3D point is P_cam = λ · d.

        Parameters
        ----------
        u, v : float
            Pixel coordinates (column, row).

        Returns
        -------
        np.ndarray
            3D point [X, Y, Z] in the camera coordinate frame, shape (3,).
        """
        # Step 1: Undistort pixel to normalized image coordinates.
        # cv2.undistortPoints expects shape (N, 1, 2) and returns (N, 1, 2)
        pixel = np.array([[[u, v]]], dtype=np.float64)
        normalized = cv2.undistortPoints(pixel, self.K, self.dist)  # (1, 1, 2)
        xn: float = float(normalized[0, 0, 0])
        yn: float = float(normalized[0, 0, 1])

        # Step 2: Camera ray direction in normalized coordinates
        # The ray from the camera origin through the pixel is d = [xn, yn, 1]^T
        # (in normalized / metric image coordinates).

        # Step 3: Ray-plane intersection parameter
        # Plane: A·X + B·Y + C·Z + D = 0
        # Point on ray: P = λ·[xn, yn, 1]
        # Substituting: A·λ·xn + B·λ·yn + C·λ + D = 0
        #   =>  λ = -D / (A·xn + B·yn + C)
        denominator: float = self.A * xn + self.B * yn + self.C
        if abs(denominator) < 1e-12:
            raise ValueError(
                f"Ray through pixel ({u}, {v}) is (nearly) parallel to "
                f"the laser plane — no finite intersection."
            )
        lam: float = -self.D / denominator

        # Step 4: 3D point in camera frame
        p_cam = lam * np.array([xn, yn, 1.0], dtype=np.float64)
        return p_cam

    # ------------------------------------------------------------------
    # Vectorized batch triangulation
    # ------------------------------------------------------------------
    def pixels_to_3d_batch(self, pixels: np.ndarray) -> np.ndarray:
        """Convert N pixels to 3D points in camera frame (vectorized).

        This method avoids Python-level loops by leveraging OpenCV's
        batch undistortion and NumPy broadcasting.

        Parameters
        ----------
        pixels : np.ndarray
            Nx2 array of pixel coordinates [[u1, v1], ...].

        Returns
        -------
        np.ndarray
            Nx3 array of 3D points [[X, Y, Z], ...] in camera frame.

        Raises
        ------
        ValueError
            If any ray is parallel to the laser plane.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        n = pixels.shape[0]

        # --- Step 1: Batch undistortion ---------------------------------
        # cv2.undistortPoints wants (N, 1, 2) float64
        pts_in = pixels.reshape(n, 1, 2)
        normalized = cv2.undistortPoints(pts_in, self.K, self.dist)  # (N, 1, 2)
        xn = normalized[:, 0, 0]  # (N,)
        yn = normalized[:, 0, 1]  # (N,)

        # --- Step 2: Form ray directions --------------------------------
        # Each ray: d_i = [xn_i, yn_i, 1]

        # --- Step 3: Vectorized ray-plane intersection ------------------
        # λ_i = -D / (A·xn_i + B·yn_i + C)
        denominator = self.A * xn + self.B * yn + self.C  # (N,)

        # Check for (near-)parallel rays
        parallel_mask = np.abs(denominator) < 1e-12
        if np.any(parallel_mask):
            bad_indices = np.where(parallel_mask)[0]
            raise ValueError(
                f"Rays at pixel indices {bad_indices.tolist()} are "
                f"(nearly) parallel to the laser plane."
            )

        lam = -self.D / denominator  # (N,)

        # --- Step 4: Compute 3D points P_cam = λ · [xn, yn, 1] ---------
        points_3d = np.column_stack([xn, yn, np.ones(n)]) * lam[:, np.newaxis]
        return points_3d  # (N, 3)

    def pixels_to_3d_interval(
        self,
        p_ridge: np.ndarray,
        y_c: float,
        y_l: float,
        y_r: float,
        n_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Steger/EKM sub-pixel offsets to 3D confidence interval.
        
        Maps the profile offsets back to full image coordinates along
        the cross-ridge normal, undistorts them, and intersects the
        resulting rays with the laser plane.
        
        Parameters
        ----------
        p_ridge : np.ndarray
            Base Steger pixel coordinate [u, v].
        y_c, y_l, y_r : float
            EKM centroid, left bound, and right bound (offset along normal).
        n_hat : np.ndarray
            Steger normal vector [nx, ny] at p_ridge.
            
        Returns
        -------
        P_c, P_l, P_r : np.ndarray
            3D points for the center, left bound, and right bound.
        """
        p_ridge = np.asarray(p_ridge, dtype=np.float64)
        n_hat = np.asarray(n_hat, dtype=np.float64)
        
        # Step 1: Map profile offset to image coordinates
        p_c_img = p_ridge + y_c * n_hat
        p_l_img = p_ridge + y_l * n_hat
        p_r_img = p_ridge + y_r * n_hat
        
        pts_in = np.array([p_c_img, p_l_img, p_r_img], dtype=np.float64)
        
        # Use existing batch method (handles Step 2, 3, 4)
        pts_3d = self.pixels_to_3d_batch(pts_in)
        
        return pts_3d[0], pts_3d[1], pts_3d[2]

    # ------------------------------------------------------------------
    # Coordinate frame transformation
    # ------------------------------------------------------------------
    @staticmethod
    def transform_to_robot(
        points_cam: np.ndarray,
        T_cam2robot: np.ndarray,
    ) -> np.ndarray:
        """Apply a 4×4 homogeneous transform to convert camera-frame
        points into robot-frame points.

        Parameters
        ----------
        points_cam : np.ndarray
            Nx3 array of points in camera coordinates.
        T_cam2robot : np.ndarray
            4×4 homogeneous transformation matrix from camera to robot.

        Returns
        -------
        np.ndarray
            Nx3 array of points in robot coordinates.
        """
        points_cam = np.asarray(points_cam, dtype=np.float64)
        if points_cam.ndim == 1:
            points_cam = points_cam.reshape(1, 3)
        n = points_cam.shape[0]

        T = np.asarray(T_cam2robot, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T_cam2robot must be 4×4")

        # Build homogeneous coordinates [X, Y, Z, 1]^T
        ones = np.ones((n, 1), dtype=np.float64)
        pts_h = np.hstack([points_cam, ones])  # (N, 4)

        # Apply transform: P_robot = T · P_cam (transpose trick for Nx4)
        pts_robot_h = (T @ pts_h.T).T  # (N, 4)

        return pts_robot_h[:, :3]

    # ------------------------------------------------------------------
    # Depth confidence metric
    # ------------------------------------------------------------------
    def compute_depth_confidence(
        self,
        points_3d: np.ndarray,
        baseline_mm: float = 100.0,
        focal_length_mm: float = 16.0,
        delta_p_px: float = 0.1,
        z_ref_mm: float = 200.0,
    ) -> np.ndarray:
        """Compute a per-point depth confidence that models the non-linear
        degradation of depth resolution with distance.

        Depth resolution of a triangulation system degrades quadratically
        with distance:

            δZ = Z² / (f · B) · δp          (theoretical depth uncertainty)

        where *Z* is the depth, *f* the focal length, *B* the baseline
        between the camera and the laser emitter, and *δp* the sub-pixel
        detection uncertainty.  The confidence metric is defined as the
        inverse relationship normalised to [0, 1]:

            C(Z) = z_ref² / Z²              (clipped to [0, 1])

        At the reference calibration depth *z_ref*, confidence is 1.0 and
        it falls off quadratically for greater depths.

        Parameters
        ----------
        points_3d : np.ndarray
            Nx3 array of 3D points; the Z (depth) column is used.
        baseline_mm : float, optional
            Stereo baseline between camera and laser emitter in mm.
        focal_length_mm : float, optional
            Effective focal length of the camera in mm.
        delta_p_px : float, optional
            Sub-pixel detection uncertainty in pixels.
        z_ref_mm : float, optional
            Reference / calibration depth at which confidence = 1.0.

        Returns
        -------
        np.ndarray
            Shape (N,) array of confidence values in [0, 1].
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)

        # Depth is the Z component (column index 2)
        z = points_3d[:, 2]  # (N,)

        # Guard against zero / near-zero depth to avoid division by zero
        z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)

        # C(Z) = z_ref^2 / Z^2, clipped to [0, 1]
        confidence = (z_ref_mm ** 2) / (z_safe ** 2)
        confidence = np.clip(confidence, 0.0, 1.0)

        return confidence

    # ------------------------------------------------------------------
    # Surface normal computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_surface_normals(
        points_3d: np.ndarray,
        up_hint: np.ndarray = None,
    ) -> np.ndarray:
        """Compute local surface normals along an ordered 3D trajectory.

        Each normal is derived from the cross product of the local
        tangent vector with a user-supplied *up_hint* direction.  Tangent
        vectors are estimated via central differences for interior points
        and one-sided (forward / backward) differences at the two
        endpoints.

        Algorithm
        ---------
        1. tangent[i] = points[i+1] − points[i−1]   (central difference)
           - tangent[0]   = points[1] − points[0]    (forward difference)
           - tangent[N-1] = points[N-1] − points[N-2] (backward difference)
        2. normal[i] = cross(tangent[i], up_hint), then normalise to
           unit length.
        3. If any resulting normal has near-zero magnitude (< 1e-12),
           it is replaced by the *up_hint* vector (normalised).

        Parameters
        ----------
        points_3d : np.ndarray
            Nx3 array of ordered 3D trajectory points.
        up_hint : np.ndarray, optional
            3-element direction vector used as the second operand of the
            cross product.  Defaults to [0, 0, 1].

        Returns
        -------
        np.ndarray
            Nx3 array of unit normal vectors.
        """
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0])
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        up_hint = np.asarray(up_hint, dtype=np.float64)

        n = points_3d.shape[0]

        # --- Step 1: Compute tangent vectors ----------------------------
        tangents = np.empty_like(points_3d)  # (N, 3)

        if n == 1:
            # Single point: tangent is undefined; use a zero vector
            # (will be replaced by up_hint in the fallback step).
            tangents[0] = 0.0
        else:
            # Forward difference for the first point
            tangents[0] = points_3d[1] - points_3d[0]
            # Backward difference for the last point
            tangents[-1] = points_3d[-1] - points_3d[-2]
            # Central differences for interior points (vectorised)
            if n > 2:
                tangents[1:-1] = points_3d[2:] - points_3d[:-2]

        # --- Step 2: Cross product with up_hint -------------------------
        normals = np.cross(tangents, up_hint)  # (N, 3)

        # --- Step 3: Normalise to unit length ---------------------------
        magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)  # (N, 1)

        # Identify degenerate normals (near-zero magnitude)
        degenerate_mask = (magnitudes < 1e-12).squeeze()  # (N,)

        # Safe-divide: avoid division by zero for degenerate rows
        safe_magnitudes = np.where(magnitudes < 1e-12, 1.0, magnitudes)
        normals = normals / safe_magnitudes

        # Fall back to normalised up_hint for degenerate entries
        if np.any(degenerate_mask):
            up_norm = up_hint / np.linalg.norm(up_hint)
            normals[degenerate_mask] = up_norm

        return normals
