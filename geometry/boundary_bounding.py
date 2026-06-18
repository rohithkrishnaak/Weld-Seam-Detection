"""
Geometric Seam Boundary Localization and Truncation (Stage 3.5).

Implements the two-stage bounding architecture described in the
"Bounding the Weld Seam" technical report (Round 3 Supplement).

This module addresses the X-axis boundary failure: the pipeline previously
had no explicit model of plate extent along the seam-length axis, causing:

  - **False positives**: the extracted seam extends past the physical metal
    plate edges into background, fixture, or specular-glint pixels.
  - **False negatives**: the seam is truncated before the true endpoints
    because ridge strength drops at chamfers/cut-edges and no recovery
    mechanism exists for terminal gaps.

The fix is two clearly separable stages:

Stage A — Plate-Presence Gating (``compute_plate_presence``, ``gate_columns``)
    A per-column scalar Φ(c) — the plate-presence likelihood — computed
    independently of Steger ridge strength so it cannot be fooled by
    the same specular artefacts that fool the ridge detector.  Folded into
    column admission as a multiplicative gate.

Stage B — Geometric Boundary Localization (``localize_seam_boundaries``)
    Operates on the assembled per-column (x, y) arrays after smoothing
    and before triangulation.  Uses:

    1. Mamdani FIS with Sugeno defuzzification to fuse three evidence
       channels (mask fill-ratio, ridge indicator, local continuity) into
       a single presence profile Φ(c).
    2. Gaussian smoothing + central-difference derivatives of Φ(c).
    3. Argmax-anchored bidirectional zero-crossing search to locate
       c_start* and c_end* — removes the unjustified center-of-frame prior
       of the original bisection approach.
    4. Physically-scaled Hampel/MAD continuity filter with window size
       derived from the calibrated pixel pitch at the reference depth —
       invariant to sensor resolution and working-distance changes.
    5. GTrFN asymmetric confidence envelope per located boundary
       (diagnostic only; does not alter the crisp truncation boundary).

Integration point in ``WeldSeamDetector.detect()``::

    # After full_smoothing_pipeline, before pixels_to_3d_batch:
    P_mask, P_chroma, _ = compute_plate_presence(
        gray, effective_mask, hardware_cfg.w_pixels)
    P_ridge = build_ridge_indicator(x_steger_cols, gray.shape[1])
    x_bounded, y_bounded, c_start, c_end, env_s, env_e = \\
        localize_seam_boundaries(
            x_smooth, y_smooth, P_mask, P_ridge, W=gray.shape[1],
            hardware_cfg=hardware_cfg,
            Z_ref=hardware_cfg.depth_confidence_z_ref_mm)

References
----------
[1] Canny, J. (1986). A Computational Approach to Edge Detection.
    IEEE TPAMI, 8(6), 679-698.
[2] Marr, D. & Hildreth, E. (1980). Theory of Edge Detection.
    Proc. Royal Society B, 207(1167), 187-217.
[3] Hampel, F.R. (1974). The Influence Curve and its Role in Robust
    Estimation. JASA, 69(346), 383-393.
[8] Mamdani, E.H. & Assilian, S. (1975). An experiment in linguistic
    synthesis with a fuzzy logic controller. IJMMS, 7(1), 1-13.
[10] Chen, S.H. & Hsieh, C.H. (2000). Representation, ranking, distance,
     and similarity of L-R type fuzzy number and application.
     Australian J. Intelligent Info. Processing Sys., 6(4), 217-229.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from config import HardwareConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ============================================================
# Stage A — Plate-Presence Gating
# ============================================================

def compute_plate_presence(
    gray: np.ndarray,
    effective_mask: np.ndarray,
    expected_stripe_height: int,
    bgr: Optional[np.ndarray] = None,
    tau_contrast: float = 10.0,
    s_contrast: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-column plate-presence evidence arrays (Stage A).

    Two complementary signals are returned, both independent of Steger
    ridge strength, so they cannot be fooled by the same specular
    artefacts that trigger false ridge acceptances.

    Parameters
    ----------
    gray : np.ndarray, shape (H, W)
        Float64 grayscale image (red channel for a 650 nm laser).
    effective_mask : np.ndarray, shape (H, W)
        Binary mask from FCM + fuzzy morphology (non-zero = stripe).
    expected_stripe_height : int
        Nominal laser-stripe pixel height (``HardwareConfig.w_pixels``).
        Used to normalise the mask-fill ratio to [0, 1].
    bgr : np.ndarray, shape (H, W, 3), optional
        Original BGR image.  When provided, the red-dominance signal
        R − max(G, B) is computed from it for a more precise
        chromaticity column coherence score.  When None, ``gray`` is
        used as a proxy (single-channel fallback).
    tau_contrast : float
        Sigmoid centre for the chromaticity contrast ratio.
    s_contrast : float
        Sigmoid softness for the chromaticity contrast ratio.

    Returns
    -------
    P_mask : np.ndarray, shape (W,)
        Column mask-fill ratio clipped to [0, 1].
    P_chroma : np.ndarray, shape (W,)
        Column chromaticity coherence score in [0, 1].
    peak_row : np.ndarray, shape (W,)
        Row index of the chromaticity peak per column (diagnostic).
    """
    H, W = gray.shape[:2]
    expected_stripe_height = max(expected_stripe_height, 1)

    # ── P_mask: column mask-fill ratio ──
    col_mask_count = effective_mask.astype(np.float64).sum(axis=0)   # (W,)
    P_mask = np.clip(col_mask_count / expected_stripe_height, 0.0, 1.0)

    # ── Chromaticity column coherence ──
    if bgr is not None and bgr.ndim == 3:
        r = bgr[:, :, 2].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        b = bgr[:, :, 0].astype(np.float32)
        red_dom = (r - np.maximum(g, b)).astype(np.float64)
    else:
        # Single-channel fallback: use raw intensity
        red_dom = gray.astype(np.float64)

    peak_val = red_dom.max(axis=0)            # (W,)
    peak_row = red_dom.argmax(axis=0)         # (W,)
    bg_med = np.median(red_dom, axis=0)       # (W,)

    contrast = peak_val - bg_med              # (W,)
    # Sigmoid: 0 when contrast ≪ tau_contrast, 1 when contrast ≫ tau_contrast
    P_chroma = 1.0 / (1.0 + np.exp(-(contrast - tau_contrast) / (s_contrast + _EPS)))

    return P_mask, P_chroma, peak_row.astype(np.float64)


def gate_columns(
    valid_pixel_mask: np.ndarray,
    P_mask: np.ndarray,
    P_chroma: np.ndarray,
    tau_phi: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the Stage A column gate to Steger's pixel-level valid mask.

    The gate is strictly additive from the pipeline's perspective: it
    can only *remove* columns that the existing Hessian criteria would
    have accepted; it never adds new ones.

    Parameters
    ----------
    valid_pixel_mask : np.ndarray, shape (H, W), dtype bool
        Per-pixel validity from Steger's acceptance criteria.
    P_mask : np.ndarray, shape (W,)
        Column mask-fill ratio (from ``compute_plate_presence``).
    P_chroma : np.ndarray, shape (W,)
        Column chromaticity coherence (from ``compute_plate_presence``).
    tau_phi : float
        Gate threshold.  Columns with Φ(c) < tau_phi are suppressed.

    Returns
    -------
    gated_mask : np.ndarray, shape (H, W), dtype bool
        Tightened pixel mask.
    Phi : np.ndarray, shape (W,)
        Per-column gate score Φ(c) ∈ [0, 1].
    """
    Phi = 0.5 * P_mask + 0.5 * P_chroma          # (W,)
    column_gate = Phi >= tau_phi                   # (W,) boolean
    # Broadcast gate across rows
    gated_mask = valid_pixel_mask & column_gate[np.newaxis, :]
    return gated_mask, Phi


def build_ridge_indicator(
    x_steger_cols: np.ndarray,
    W: int,
) -> np.ndarray:
    """Build a per-column ridge indicator P_ridge(c) ∈ {0, 1}.

    A column receives value 1 if at least one Steger ridge point was
    accepted in it; 0 otherwise.

    Parameters
    ----------
    x_steger_cols : np.ndarray
        Integer (or float, rounded) column indices of accepted ridge points.
    W : int
        Total image width.

    Returns
    -------
    P_ridge : np.ndarray, shape (W,)
    """
    P_ridge = np.zeros(W, dtype=np.float64)
    cols = np.round(x_steger_cols).astype(int)
    cols = cols[(cols >= 0) & (cols < W)]
    P_ridge[cols] = 1.0
    return P_ridge


# ============================================================
# Stage B — Boundary Localization
# ============================================================

# ── Helper: triangular membership function ──────────────────

def _trimf(v: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function TRIMF(v, a, b, c).

    Returns max(0, min((v-a)/(b-a), (c-v)/(c-b))).
    Handles degenerate cases (a==b or b==c) gracefully.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        left = np.where(
            np.abs(b - a) > _EPS,
            np.where(b - a > _EPS, (v - a) / (b - a), 0.0),
            np.where(v >= b, 1.0, 0.0),
        )
        right = np.where(
            np.abs(c - b) > _EPS,
            np.where(c - b > _EPS, (c - v) / (c - b), 0.0),
            np.where(v <= b, 1.0, 0.0),
        )
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def fuzzy_fuse_presence(
    P_mask: np.ndarray,
    P_ridge: np.ndarray,
    P_continuity: np.ndarray,
) -> np.ndarray:
    """Mamdani FIS with Sugeno defuzzification: fuse three evidence channels.

    Implements the 5-rule base from §3.2.1 of the technical report:

    R1: P_mask=HIGH ∧ P_ridge=HIGH ∧ P_continuity=HIGH → Φ = HIGH (0.9)
    R2: P_mask=LOW  ∧ P_ridge=HIGH ∧ P_continuity=HIGH → Φ = MED  (0.5)
    R3: P_mask=HIGH ∧ P_ridge=LOW  ∧ P_continuity=HIGH → Φ = MED  (0.5)
    R4: P_mask=LOW  ∧ P_ridge=LOW  ∧ P_continuity=HIGH → Φ = LOW  (0.1)
    R5: P_mask=LOW  ∧ P_ridge=LOW  ∧ P_continuity=LOW  → Φ = LOW  (0.1)

    Antecedents connected by fuzzy AND = min.
    Defuzzification: Sugeno weighted average (closed-form, no area integration).

    Parameters
    ----------
    P_mask, P_ridge, P_continuity : np.ndarray, shape (W,)
        All three inputs must be in [0, 1].

    Returns
    -------
    Phi : np.ndarray, shape (W,)
        Fused plate-presence likelihood, in (0, 1).
    """
    # Shared triangular membership functions (all inputs normalized to [0,1])
    mu_low  = lambda v: _trimf(v, 0.0, 0.0, 0.5)   # noqa: E731
    mu_med  = lambda v: _trimf(v, 0.2, 0.5, 0.8)   # noqa: E731
    mu_high = lambda v: _trimf(v, 0.5, 1.0, 1.0)   # noqa: E731

    m_lo = mu_low(P_mask);   m_md = mu_med(P_mask);   m_hi = mu_high(P_mask)
    r_lo = mu_low(P_ridge);  r_md = mu_med(P_ridge);  r_hi = mu_high(P_ridge)
    k_lo = mu_low(P_continuity); k_md = mu_med(P_continuity); k_hi = mu_high(P_continuity)

    # Crisp Sugeno consequents (singletons)
    z_LOW, z_MED, z_HIGH = 0.1, 0.5, 0.9

    # Rule firing strengths (fuzzy AND = min), vectorized
    mu1 = np.minimum(np.minimum(m_hi, r_hi), k_hi)   # R1 → HIGH
    mu2 = np.minimum(np.minimum(m_lo, r_hi), k_hi)   # R2 → MED
    mu3 = np.minimum(np.minimum(m_hi, r_lo), k_hi)   # R3 → MED
    mu4 = np.minimum(np.minimum(m_lo, r_lo), k_hi)   # R4 → LOW
    mu5 = np.minimum(np.minimum(m_lo, r_lo), k_lo)   # R5 → LOW

    num = mu1 * z_HIGH + mu2 * z_MED + mu3 * z_MED + mu4 * z_LOW + mu5 * z_LOW
    den = mu1 + mu2 + mu3 + mu4 + mu5

    # Sugeno weighted average; fall back to 0 where no rules fire
    with np.errstate(divide='ignore', invalid='ignore'):
        Phi = np.where(den > _EPS, num / den, 0.0)
    return Phi


def build_gtrfn_envelope(
    Phi_smooth: np.ndarray,
    c_star: float,
    half_decay: float = 0.5,
) -> Tuple[float, float, float, float, float, float]:
    """Fit an asymmetric Generalised Trapezoidal Fuzzy Number around c_star.

    The GTrFN Ã = (a1, a2, a3, a4; h) encodes the boundary confidence
    envelope.  The crisp core a2=a3=c* is the located zero-crossing;
    the spreads a1 and a4 are fit from the empirical decay rate of
    Phi_smooth on each side (the column offset at which Phi_smooth first
    falls below half_decay * Phi_smooth(c*)).

    This is a **diagnostic** — it does not alter the crisp truncation
    boundary used in localize_seam_boundaries.

    Parameters
    ----------
    Phi_smooth : np.ndarray, shape (W,)
        Smoothed presence profile.
    c_star : float
        Sub-pixel boundary column (e.g. c_start* or c_end*).
    half_decay : float
        Fraction of Phi_smooth(c*) used as the spread stopping criterion.

    Returns
    -------
    (a1, a2, a3, a4, h, centroid) : tuple of float
    """
    W = len(Phi_smooth)
    c_int = int(np.clip(round(c_star), 0, W - 1))
    peak_val = float(Phi_smooth[c_int])
    target = half_decay * peak_val

    # Walk inward (toward centre) — gradual background re-entry side
    delta_in = 0
    while (c_int - delta_in) > 0 and Phi_smooth[c_int - delta_in] > target:
        delta_in += 1

    # Walk outward (away from centre) — sharp plate-exit side
    delta_out = 0
    while (c_int + delta_out) < W - 1 and Phi_smooth[c_int + delta_out] > target:
        delta_out += 1

    a1 = c_star - delta_in
    a2 = c_star
    a3 = c_star
    a4 = c_star + delta_out
    max_phi = float(Phi_smooth.max()) if Phi_smooth.max() > _EPS else 1.0
    h = peak_val / max_phi
    centroid = ((a1 + a2 + a3 + a4) / 4.0) * h   # closed-form GTrFN centroid

    return a1, a2, a3, a4, h, centroid


def localize_seam_boundaries(
    x_cols: np.ndarray,
    y_vals: np.ndarray,
    P_mask: np.ndarray,
    P_ridge: np.ndarray,
    W: int,
    sigma_b: float = 4.0,
    mad_k: float = 3.0,
    L_w_mm: float = 2.0,
    hardware_cfg: Optional[HardwareConfig] = None,
    Z_ref: Optional[float] = None,
    y_scale: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float, float,
           Tuple, Tuple]:
    """Locate physical plate boundaries and truncate the seam array (Stage B).

    Parameters
    ----------
    x_cols : np.ndarray, shape (M,)
        Sorted integer column indices from the smoothed seam path.
    y_vals : np.ndarray, shape (M,)
        Corresponding sub-pixel Y centres.
    P_mask : np.ndarray, shape (W,)
        Column mask-fill ratio (Stage A output).
    P_ridge : np.ndarray, shape (W,)
        Binary column ridge indicator (from ``build_ridge_indicator``).
    W : int
        Full image width.
    sigma_b : float
        Gaussian smoothing scale (columns) for the presence profile.
    mad_k : float
        MAD rejection multiplier (≈ 3, matching the existing z_threshold
        convention in profile_smoother.py).
    L_w_mm : float
        Physical Hampel window length in mm (default 2 mm).
    hardware_cfg : HardwareConfig, optional
        Provides ``sensor_pixel_size_um`` and ``focal_length_mm`` for
        pixel-pitch computation.  When None a fallback of 7 columns is used.
    Z_ref : float, optional
        Reference depth in mm for pixel-pitch computation.  When None,
        defaults to ``hardware_cfg.depth_confidence_z_ref_mm`` if
        hardware_cfg is provided, else 200 mm.
    y_scale : float
        Characteristic seam curvature scale (px) for the continuity signal.

    Returns
    -------
    x_bounded : np.ndarray
        Truncated column indices within the validated plate extent.
    y_bounded : np.ndarray
        Corresponding Y centres.
    c_start : float
        Sub-pixel left (start) boundary column.
    c_end : float
        Sub-pixel right (end) boundary column.
    env_start : tuple
        GTrFN envelope (a1, a2, a3, a4, h, centroid) for the start boundary.
    env_end : tuple
        GTrFN envelope for the end boundary.
    """
    M = len(x_cols)

    # ── Step 1: Continuity signal & Fuzzy FIS fusion ──────────────────
    P_continuity = np.zeros(W, dtype=np.float64)
    x_int = np.round(x_cols).astype(int)
    for i in range(1, M - 1):
        dy_left  = abs(float(y_vals[i]) - float(y_vals[i - 1]))
        dy_right = abs(float(y_vals[i + 1]) - float(y_vals[i]))
        col = int(np.clip(x_int[i], 0, W - 1))
        P_continuity[col] = float(
            np.exp(-(dy_left + dy_right) / (2.0 * max(y_scale, _EPS)))
        )

    Phi = fuzzy_fuse_presence(P_mask, P_ridge, P_continuity)

    # ── Step 2: Gaussian smooth + derivatives ─────────────────────────
    Phi_smooth = gaussian_filter1d(Phi.astype(np.float64), sigma=sigma_b)

    # Central differences (pad edges with nearest value)
    d1Phi = np.gradient(Phi_smooth)
    d2Phi = np.gradient(d1Phi)

    # ── Step 3: Argmax-anchored bidirectional zero-crossing search ────
    c_peak = int(np.argmax(Phi_smooth))

    def _find_zero_crossing(d2: np.ndarray, start: int, stop: int, step: int):
        """Traverse from start toward stop (step = ±1); return first k where
        sign(d2[k]) ≠ sign(d2[k+step]), or None if not found."""
        indices = range(start, stop, step)
        for k in indices:
            k_next = k + step
            if 0 <= k_next < len(d2):
                if d2[k] * d2[k_next] < 0:   # sign change
                    return k
        return None

    c_start_int = _find_zero_crossing(d2Phi, c_peak, 0, -1)
    c_end_int   = _find_zero_crossing(d2Phi, c_peak, W - 1, +1)

    # Fallback: if no crossing found, use the edges of the detected point range
    if c_start_int is None:
        c_start_int = max(int(x_int.min()) - 1, 0)
        logger.warning("No left zero-crossing found; falling back to min column %d.", c_start_int)
    if c_end_int is None:
        c_end_int = min(int(x_int.max()) + 1, W - 2)
        logger.warning("No right zero-crossing found; falling back to max column %d.", c_end_int)

    # Sub-pixel refinement (1st-order Taylor expansion of d2Phi about zero)
    def _subpixel_refine(d2: np.ndarray, k: int, step: int) -> float:
        k_next = k + step
        if 0 <= k_next < len(d2):
            denom = d2[k_next] - d2[k]
            if abs(denom) > _EPS:
                return float(k) - float(d2[k]) / denom
        return float(k)

    c_start = _subpixel_refine(d2Phi, c_start_int, +1)   # leftward result
    c_end   = _subpixel_refine(d2Phi, c_end_int,   +1)   # rightward result

    # ── Safety clamp: boundaries must lie within the data range ──────
    # The FIS operates over the full image width but may produce a very
    # narrow peak if the Steger output is sparse (e.g. few ridge points).
    # Clamp boundaries to [data_min, data_max] so we never over-truncate.
    data_col_min = float(x_int.min())
    data_col_max = float(x_int.max())
    data_span = data_col_max - data_col_min

    # Guarantee ordering (c_start ≤ c_end) — by construction but defensive
    if c_start > c_end:
        c_start, c_end = c_end, c_start

    # If the located window is narrower than 10% of the data span, the
    # boundary finder likely converged on a spurious local feature rather
    # than the true plate edges — fall back to the data range.
    located_span = c_end - c_start
    if located_span < max(data_span * 0.10, 10):
        logger.warning(
            "Boundary window (%.1f columns) is implausibly narrow relative to "
            "data span (%.1f columns) — falling back to data range.",
            located_span, data_span,
        )
        c_start = data_col_min
        c_end   = data_col_max
    else:
        # Soft-clamp: do not cut into the data range
        c_start = min(c_start, data_col_min)
        c_end   = max(c_end,   data_col_max)

    logger.info(
        "Boundary localization: c_start*=%.2f  c_peak=%d  c_end*=%.2f",
        c_start, c_peak, c_end,
    )

    # ── Step 4: Physically-scaled Hampel / MAD continuity filter ─────
    window_columns = _compute_hampel_window(hardware_cfg, Z_ref, L_w_mm)
    logger.info(
        "Hampel window: %d columns (L_w=%.1f mm).", window_columns, L_w_mm
    )

    # Select points within the validated window
    window_mask = (x_int >= int(np.floor(c_start))) & (x_int <= int(np.ceil(c_end)))
    x_win = x_cols[window_mask]
    y_win = y_vals[window_mask]

    if len(y_win) > window_columns:
        inlier_mask = _hampel_filter(y_win, window_columns, mad_k)
        # Run-length gate: reject inlier runs shorter than window_columns
        inlier_mask = _reject_short_runs(inlier_mask, min_run=window_columns)
        x_bounded = x_win[inlier_mask]
        y_bounded = y_win[inlier_mask]
    else:
        # Too few points for Hampel — keep all
        x_bounded = x_win.copy()
        y_bounded = y_win.copy()

    if len(x_bounded) == 0:
        logger.warning(
            "Hampel filter removed all points — returning full window as fallback."
        )
        x_bounded = x_win.copy()
        y_bounded = y_win.copy()

    # ── Step 5: GTrFN confidence envelopes (diagnostic) ───────────────
    env_start = build_gtrfn_envelope(Phi_smooth, c_start)
    env_end   = build_gtrfn_envelope(Phi_smooth, c_end)

    logger.info(
        "Boundary truncation: %d → %d points (%.1f%% retained).",
        M, len(x_bounded),
        100.0 * len(x_bounded) / max(M, 1),
    )

    return x_bounded, y_bounded, c_start, c_end, env_start, env_end


# ============================================================
# Private helpers
# ============================================================

def _compute_hampel_window(
    hardware_cfg: Optional[HardwareConfig],
    Z_ref: Optional[float],
    L_w_mm: float,
) -> int:
    """Convert the physical Hampel window length to an odd integer column count.

    Uses the calibrated pixel pitch at the reference depth:

        pixel_pitch_mm(Z) = (sensor_pixel_size_um * Z) / (focal_length_mm * 1000)
        window_columns    = round(L_w_mm / pixel_pitch_mm(Z_ref))  [odd, ≥ 3]

    Falls back to 7 columns (the original hard-coded value) if hardware
    configuration is unavailable.
    """
    FALLBACK = 7
    if hardware_cfg is None:
        return FALLBACK

    z = Z_ref if Z_ref is not None else getattr(
        hardware_cfg, 'depth_confidence_z_ref_mm', 200.0
    )
    sensor_px_um = getattr(hardware_cfg, 'sensor_pixel_size_um', None)
    focal_mm     = getattr(hardware_cfg, 'focal_length_mm', None)

    if sensor_px_um is None or focal_mm is None or focal_mm < _EPS:
        return FALLBACK

    pixel_pitch_mm = (sensor_px_um * z) / (focal_mm * 1000.0)
    if pixel_pitch_mm < _EPS:
        return FALLBACK

    w = int(round(L_w_mm / pixel_pitch_mm))
    w = max(w, 3)
    if w % 2 == 0:
        w += 1
    return w


def _hampel_filter(
    y: np.ndarray,
    window: int,
    k: float,
) -> np.ndarray:
    """Rolling Hampel / MAD inlier filter.

    For each point i, the local median and MAD are computed over a window
    centred at i.  Points with |y[i] - median| > k * MAD are flagged as
    outliers.

    Parameters
    ----------
    y : np.ndarray
        Y-centre values (1-D).
    window : int
        Odd rolling-window length.
    k : float
        Rejection multiplier (≈ 3, matching z_threshold convention).

    Returns
    -------
    inlier_mask : np.ndarray, shape (len(y),), dtype bool
    """
    n = len(y)
    inlier = np.ones(n, dtype=bool)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = y[lo:hi]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        if mad < _EPS:
            continue
        if abs(float(y[i]) - med) > k * mad:
            inlier[i] = False

    return inlier


def _reject_short_runs(
    inlier_mask: np.ndarray,
    min_run: int,
) -> np.ndarray:
    """Reject contiguous inlier runs shorter than *min_run* columns.

    This implements the physical run-length gate described in §3.2.4:
    even if individual points pass the Hampel test, a very short run of
    inliers surrounded by outliers is likely a specular artefact rather
    than a real plate segment.

    Parameters
    ----------
    inlier_mask : np.ndarray, dtype bool
    min_run : int
        Minimum inlier run length to keep.

    Returns
    -------
    updated_mask : np.ndarray, dtype bool
    """
    result = inlier_mask.copy()
    n = len(inlier_mask)
    i = 0
    while i < n:
        if inlier_mask[i]:
            # Find end of this run
            j = i
            while j < n and inlier_mask[j]:
                j += 1
            run_length = j - i
            if run_length < min_run:
                result[i:j] = False
            i = j
        else:
            i += 1
    return result
