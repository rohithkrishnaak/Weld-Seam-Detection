"""
Seam Geometry Feature Extraction from 3D Laser Profiles.

Extracts geometric features (root point, groove angle, gap width, step
height, corner point) from different weld joint types by analysing
the 2D cross-sectional profile (x, z) obtained from laser triangulation.

Supported joint types:
- V-Groove: root point, groove angle, gap width, depth
- Lap Joint: step edge, step height, overlap distance
- Butt Joint: gap center, gap width, plate mismatch
- Fillet Joint: corner point, leg lengths

References:
    [1] Zou, Y., Chen, J., & Wei, X. (2020). Research on a real-time
        weld seam tracking method. J. Manuf. Processes, 56, 538-551.
    [2] Fischler, M. A. & Bolles, R. C. (1981). Random Sample
        Consensus: A Paradigm for Model Fitting. CACM, 24(6), 381-395.
    [3] Savitzky, A. & Golay, M. J. E. (1964). Smoothing and
        Differentiation of Data by Simplified Least Squares
        Procedures. Analytical Chemistry, 36(8), 1627-1639.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor


# ======================================================================
# Helper utilities
# ======================================================================

def _smooth_profile(
    z: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a 1-D profile.

    References:
        [3] Savitzky & Golay (1964).

    Parameters
    ----------
    z : np.ndarray
        1-D height profile.
    window_length : int
        SG filter window (must be odd and > polyorder).
    polyorder : int
        Polynomial order for local fitting.

    Returns
    -------
    np.ndarray
        Smoothed profile, same length as *z*.
    """
    # Clamp window_length to profile length (must stay odd)
    wl = min(window_length, len(z))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, polyorder + 2)
    if wl % 2 == 0:
        wl += 1
    return savgol_filter(z, window_length=wl, polyorder=polyorder)


def _numerical_derivative(
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Central-difference first derivative dz/dx.

    Parameters
    ----------
    x, z : np.ndarray
        Profile abscissa and ordinate (same length).

    Returns
    -------
    np.ndarray
        dz/dx array of same length as *x* (forward/backward at edges).
    """
    return np.gradient(z, x)


def _ransac_line_fit(
    x: np.ndarray,
    z: np.ndarray,
    residual_threshold: float = 0.1,
) -> Tuple[float, float]:
    """Fit a line z = m*x + b via RANSAC and return (slope, intercept).

    Uses scikit-learn's RANSACRegressor [2] for outlier-robust fitting.

    Parameters
    ----------
    x, z : np.ndarray
        1-D arrays of matching length.
    residual_threshold : float
        Maximum residual for an inlier (in the same units as *z*).

    Returns
    -------
    tuple[float, float]
        (slope m, intercept b).
    """
    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(
        residual_threshold=residual_threshold,
        random_state=42,
    )
    ransac.fit(X, z)
    m = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)
    return m, b


def compute_3d_approach_vector(
    normal_L: np.ndarray,
    normal_R: np.ndarray,
) -> np.ndarray:
    """Compute the 3D torch approach vector from left and right surface normals.

    Uses the bisector computation method. The approach vector is perpendicular
    to the seam direction and lies in the bisector plane.

    Parameters
    ----------
    normal_L, normal_R : np.ndarray
        Unit normal vectors of the left and right metal plates (or groove walls).

    Returns
    -------
    np.ndarray
        Unit 3D approach vector.
    """
    n_L = np.asarray(normal_L, dtype=np.float64)
    n_R = np.asarray(normal_R, dtype=np.float64)
    
    norm_L = np.linalg.norm(n_L)
    norm_R = np.linalg.norm(n_R)
    if norm_L < 1e-12 or norm_R < 1e-12:
        raise ValueError("Normals must have non-zero magnitude.")
        
    n_L /= norm_L
    n_R /= norm_R

    # 1. Seam direction (intersection of the two planes)
    d_seam = np.cross(n_L, n_R)
    norm_d = np.linalg.norm(d_seam)
    if norm_d < 1e-12:
        raise ValueError("Normals are parallel; cannot compute seam direction.")
    d_seam /= norm_d

    # 2. Bisector plane normal
    n_B = n_L + n_R
    norm_B = np.linalg.norm(n_B)
    if norm_B < 1e-12:
        raise ValueError("Normals are opposite; cannot compute bisector.")
    n_B /= norm_B

    # 3. Approach vector (perpendicular to seam, in bisector plane)
    v_approach = np.cross(d_seam, n_B)
    v_approach /= np.linalg.norm(v_approach)

    # Assuming +Z is "up" towards the camera, the torch should point "down" (-Z)
    if v_approach[2] > 0:
        v_approach = -v_approach

    return v_approach


# ======================================================================
# Main feature extractor
# ======================================================================

class SeamFeatureExtractor:
    """Extracts weld-seam geometry features from 2D laser profiles.

    Parameters
    ----------
    ransac_threshold : float
        Residual threshold (mm) for RANSAC line fitting.
    smooth_window : int
        Savitzky-Golay smoothing window length (must be odd).
    smooth_poly : int
        Savitzky-Golay polynomial order.
    """

    def __init__(
        self,
        ransac_threshold: float = 0.1,
        smooth_window: int = 11,
        smooth_poly: int = 3,
    ) -> None:
        self.ransac_threshold = ransac_threshold
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

    # ------------------------------------------------------------------
    # V-Groove
    # ------------------------------------------------------------------
    def extract_v_groove(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
    ) -> Dict[str, object]:
        """Extract V-groove joint features.

        Algorithm (following Zou et al. [1]):
            1. Smooth the height profile with Savitzky-Golay.
            2. Find the root (minimum-z point).
            3. RANSAC-fit lines to the left and right groove walls.
            4. Compute the root from line intersection.
            5. Derive groove angle and gap width at a reference height.

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.

        Returns
        -------
        dict
            {'root_point': (x, z), 'groove_angle_deg': float,
             'gap_width_mm': float, 'depth_mm': float,
             'left_wall_slope': float, 'right_wall_slope': float}
        """
        x = np.asarray(profile_x, dtype=np.float64)
        z = np.asarray(profile_z, dtype=np.float64)

        # Step 1: Smooth profile
        z_smooth = _smooth_profile(z, self.smooth_window, self.smooth_poly)

        # Step 2: Initial root estimate — global minimum of smoothed z
        root_idx = int(np.argmin(z_smooth))

        # Step 3: RANSAC line fits to left wall and right wall
        # Left wall: indices [0, root_idx)
        # Right wall: indices (root_idx, end)
        left_x, left_z = x[:root_idx], z_smooth[:root_idx]
        right_x, right_z = x[root_idx + 1:], z_smooth[root_idx + 1:]

        # Guard against degenerate partitions
        if len(left_x) < 3 or len(right_x) < 3:
            raise ValueError(
                "Insufficient points for left/right wall RANSAC fitting. "
                "The profile may not contain a clear V-groove."
            )

        m_L, b_L = _ransac_line_fit(left_x, left_z, self.ransac_threshold)
        m_R, b_R = _ransac_line_fit(right_x, right_z, self.ransac_threshold)

        # Step 4: Refined root from line intersection
        # Left line:  z = m_L * x + b_L
        # Right line: z = m_R * x + b_R
        # Intersection: m_L*x + b_L = m_R*x + b_R
        #   => x_root = (b_R - b_L) / (m_L - m_R)
        slope_diff = m_L - m_R
        if abs(slope_diff) < 1e-12:
            raise ValueError("Left and right wall slopes are parallel; cannot intersect.")
        x_root = (b_R - b_L) / slope_diff
        z_root = m_L * x_root + b_L

        # Step 5: Groove angle (angle between the two walls)
        # Each wall makes angle α with horizontal: α = arctan(slope)
        # Groove opening angle = π - |α_L - α_R|
        alpha_L = np.arctan(m_L)
        alpha_R = np.arctan(m_R)
        groove_angle_rad = np.pi - abs(alpha_L - alpha_R)
        groove_angle_deg = float(np.degrees(groove_angle_rad))

        # Step 6: Gap width at a reference height (midpoint between root
        # and the average surface height)
        surface_z = max(z_smooth[0], z_smooth[-1])
        ref_z = (z_root + surface_z) / 2.0
        # x on left line at ref_z: ref_z = m_L * x + b_L => x = (ref_z - b_L) / m_L
        if abs(m_L) > 1e-12 and abs(m_R) > 1e-12:
            x_left_at_ref = (ref_z - b_L) / m_L
            x_right_at_ref = (ref_z - b_R) / m_R
            gap_width = float(abs(x_right_at_ref - x_left_at_ref))
        else:
            gap_width = 0.0

        # Depth: surface height minus root height
        depth = float(surface_z - z_root)

        return {
            "root_point": (float(x_root), float(z_root)),
            "groove_angle_deg": groove_angle_deg,
            "gap_width_mm": gap_width,
            "depth_mm": depth,
            "left_wall_slope": float(m_L),
            "right_wall_slope": float(m_R),
        }

    # ------------------------------------------------------------------
    # Lap Joint
    # ------------------------------------------------------------------
    def extract_lap_joint(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
    ) -> Dict[str, object]:
        """Extract lap-joint features.

        Algorithm:
            1. Compute first derivative dz/dx.
            2. Step edge = location of max |dz/dx|.
            3. Fit horizontal lines to left and right plateaus.
            4. Step height = |z_top − z_bottom|.

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.

        Returns
        -------
        dict
            {'step_point': (x, z), 'step_height_mm': float,
             'weld_position': (x, z)}
        """
        x = np.asarray(profile_x, dtype=np.float64)
        z = np.asarray(profile_z, dtype=np.float64)

        z_smooth = _smooth_profile(z, self.smooth_window, self.smooth_poly)

        # Step 1: First derivative of height profile
        dz_dx = _numerical_derivative(x, z_smooth)

        # Step 2: Step edge — maximum absolute gradient
        step_idx = int(np.argmax(np.abs(dz_dx)))
        step_point = (float(x[step_idx]), float(z_smooth[step_idx]))

        # Step 3: Fit horizontal lines (constant z) to left and right
        # plateaus. We use the median of each side as a robust estimate.
        margin = max(3, len(x) // 10)  # avoid transition zone
        left_plateau_z = float(np.median(z_smooth[:max(step_idx - margin, 1)]))
        right_plateau_z = float(np.median(z_smooth[min(step_idx + margin, len(z_smooth) - 1):]))

        # Step 4: Step height
        step_height = float(abs(left_plateau_z - right_plateau_z))

        # Weld position: the step edge itself is the nominal weld point
        weld_position = step_point

        return {
            "step_point": step_point,
            "step_height_mm": step_height,
            "weld_position": weld_position,
        }

    # ------------------------------------------------------------------
    # Butt Joint
    # ------------------------------------------------------------------
    def extract_butt_joint(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
    ) -> Dict[str, object]:
        """Extract butt-joint features.

        Algorithm:
            1. Find the gap valley (global minimum = gap centre).
            2. Determine gap edges via height-threshold analysis.
            3. Gap width = right_edge − left_edge.
            4. Mismatch = height difference between left and right plates.

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.

        Returns
        -------
        dict
            {'gap_center': (x, z), 'gap_width_mm': float,
             'mismatch_mm': float}
        """
        x = np.asarray(profile_x, dtype=np.float64)
        z = np.asarray(profile_z, dtype=np.float64)

        z_smooth = _smooth_profile(z, self.smooth_window, self.smooth_poly)

        # Step 1: Gap valley — global minimum
        valley_idx = int(np.argmin(z_smooth))
        gap_center = (float(x[valley_idx]), float(z_smooth[valley_idx]))

        # Step 2: Threshold-based gap edge detection
        # Surface level estimated from the outer quarters of the profile
        quarter = max(1, len(z_smooth) // 4)
        left_surface = float(np.median(z_smooth[:quarter]))
        right_surface = float(np.median(z_smooth[-quarter:]))
        surface_level = (left_surface + right_surface) / 2.0

        # Gap threshold: points below (surface − 0.5 * depth) are "in the gap"
        depth = surface_level - gap_center[1]
        threshold_z = surface_level - 0.5 * depth

        in_gap = z_smooth < threshold_z  # boolean mask
        gap_indices = np.where(in_gap)[0]

        if len(gap_indices) > 0:
            left_edge_idx = int(gap_indices[0])
            right_edge_idx = int(gap_indices[-1])
            gap_width = float(x[right_edge_idx] - x[left_edge_idx])
        else:
            gap_width = 0.0

        # Step 3: Plate mismatch — height difference between the two sides
        mismatch = float(abs(left_surface - right_surface))

        return {
            "gap_center": gap_center,
            "gap_width_mm": gap_width,
            "mismatch_mm": mismatch,
        }

    # ------------------------------------------------------------------
    # Fillet Joint
    # ------------------------------------------------------------------
    def extract_fillet_joint(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
    ) -> Dict[str, object]:
        """Extract fillet-joint features.

        Algorithm:
            1. Segment profile into horizontal and vertical portions
               using the gradient magnitude.
            2. RANSAC-fit a line to each segment.
            3. Corner = intersection of the two fitted lines.

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.

        Returns
        -------
        dict
            {'corner_point': (x, z), 'horizontal_slope': float,
             'vertical_slope': float}
        """
        x = np.asarray(profile_x, dtype=np.float64)
        z = np.asarray(profile_z, dtype=np.float64)

        z_smooth = _smooth_profile(z, self.smooth_window, self.smooth_poly)

        # Step 1: Compute gradient to classify horizontal vs. vertical
        dz_dx = _numerical_derivative(x, z_smooth)
        abs_grad = np.abs(dz_dx)

        # Threshold: points with |dz/dx| < median are "horizontal",
        # those above are "vertical".
        grad_threshold = float(np.median(abs_grad))

        horizontal_mask = abs_grad <= grad_threshold
        vertical_mask = abs_grad > grad_threshold

        # Require a minimum number of points for each segment
        h_x, h_z = x[horizontal_mask], z_smooth[horizontal_mask]
        v_x, v_z = x[vertical_mask], z_smooth[vertical_mask]

        if len(h_x) < 3 or len(v_x) < 3:
            raise ValueError(
                "Cannot segment profile into horizontal/vertical "
                "portions — profile may not be a fillet joint."
            )

        # Step 2: RANSAC line fit to each segment
        m_h, b_h = _ransac_line_fit(h_x, h_z, self.ransac_threshold)
        m_v, b_v = _ransac_line_fit(v_x, v_z, self.ransac_threshold)

        # Step 3: Corner = intersection of the two lines
        # m_h * x + b_h = m_v * x + b_v  =>  x = (b_v - b_h) / (m_h - m_v)
        slope_diff = m_h - m_v
        if abs(slope_diff) < 1e-12:
            raise ValueError("Horizontal and vertical fits are parallel.")
        x_corner = (b_v - b_h) / slope_diff
        z_corner = m_h * x_corner + b_h

        return {
            "corner_point": (float(x_corner), float(z_corner)),
            "horizontal_slope": float(m_h),
            "vertical_slope": float(m_v),
        }

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------
    def auto_detect_joint_type(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
    ) -> str:
        """Heuristic classification of the weld joint type.

        Heuristics (applied sequentially):
            - **V-groove**: Clear valley/minimum with two rising walls
              on either side (high gradient symmetry).
            - **Lap**: Single dominant step (large monotonic gradient
              change).
            - **Butt**: Two nearly co-planar plateaus separated by a
              narrow gap (small valley, low overall gradient).
            - **Fillet**: L-shaped profile (one segment ~horizontal,
              one segment ~vertical).

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.

        Returns
        -------
        str
            One of ``'v_groove'``, ``'lap'``, ``'butt'``, ``'fillet'``.
        """
        x = np.asarray(profile_x, dtype=np.float64)
        z = np.asarray(profile_z, dtype=np.float64)
        z_smooth = _smooth_profile(z, self.smooth_window, self.smooth_poly)

        n = len(z_smooth)
        dz_dx = _numerical_derivative(x, z_smooth)
        abs_grad = np.abs(dz_dx)

        # --- Feature metrics -------------------------------------------
        # Valley prominence: how deep is the minimum relative to edges?
        min_idx = int(np.argmin(z_smooth))
        edge_z = (z_smooth[0] + z_smooth[-1]) / 2.0
        valley_depth = edge_z - z_smooth[min_idx]

        # Profile height range
        z_range = float(z_smooth.max() - z_smooth.min())
        if z_range < 1e-12:
            z_range = 1e-12  # avoid division by zero

        # Normalised valley depth
        norm_depth = valley_depth / z_range

        # Maximum gradient magnitude (step indicator)
        max_grad = float(abs_grad.max())
        mean_grad = float(abs_grad.mean())
        grad_ratio = max_grad / mean_grad if mean_grad > 1e-12 else 1.0

        # Symmetry of gradient around the minimum
        left_grad_mean = float(np.mean(dz_dx[:max(min_idx, 1)]))
        right_grad_mean = float(np.mean(dz_dx[min(min_idx + 1, n - 1):]))
        # V-groove: left slope negative, right slope positive (opposite signs)
        sign_product = left_grad_mean * right_grad_mean

        # Relative position of minimum (how centred is the valley?)
        relative_min_pos = min_idx / max(n - 1, 1)

        # --- Decision tree ---------------------------------------------
        # V-groove: deep centred valley with opposite-sign wall gradients
        if (
            norm_depth > 0.3
            and 0.15 < relative_min_pos < 0.85
            and sign_product < 0
        ):
            return "v_groove"

        # Lap joint: dominant single step (very high gradient spike)
        if grad_ratio > 3.0 and norm_depth < 0.3:
            return "lap"

        # Butt joint: shallow valley (narrow gap) between co-planar plates
        if norm_depth > 0.1 and abs(z_smooth[0] - z_smooth[-1]) / z_range < 0.3:
            return "butt"

        # Fillet: L-shaped — one part low gradient, one part high gradient
        return "fillet"

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------
    def extract_features(
        self,
        profile_x: np.ndarray,
        profile_z: np.ndarray,
        joint_type: str = "auto",
    ) -> Dict[str, object]:
        """Extract seam features for the given (or auto-detected) joint type.

        Parameters
        ----------
        profile_x, profile_z : np.ndarray
            Profile abscissa (mm) and height (mm), 1-D.
        joint_type : str
            ``'v_groove'``, ``'lap'``, ``'butt'``, ``'fillet'``, or
            ``'auto'`` (default) to automatically detect.

        Returns
        -------
        dict
            Feature dictionary (contents depend on joint type).
            Always includes ``'joint_type'`` key.
        """
        if joint_type == "auto":
            joint_type = self.auto_detect_joint_type(profile_x, profile_z)

        _extractors = {
            "v_groove": self.extract_v_groove,
            "lap": self.extract_lap_joint,
            "butt": self.extract_butt_joint,
            "fillet": self.extract_fillet_joint,
        }

        if joint_type not in _extractors:
            raise ValueError(
                f"Unknown joint type '{joint_type}'. "
                f"Must be one of {list(_extractors.keys())} or 'auto'."
            )

        features = _extractors[joint_type](profile_x, profile_z)
        features["joint_type"] = joint_type
        return features
