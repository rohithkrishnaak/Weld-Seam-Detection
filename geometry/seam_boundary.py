"""
Stage 6.5 — Along-Seam Boundary-Aware Truncation.

Operates between Stage 6 (IT2FLS/Barycentric/TFN per-column extraction)
and Stage 7 (smoothing), truncating all parallel coordinate arrays to
the physical workpiece extent BEFORE smoothing.

The core idea is that a true plate edge simultaneously activates multiple
independent radiometric/uncertainty cues, while each individual cue's
false triggers are largely uncorrelated with the others.  An OWA-fused
"off-workpiece" membership exploits that decorrelation; any single
threshold cannot.

Algorithm summary
-----------------
Step A — compute_edge_evidence:  four per-column evidence scalars
    1. Ridge-strength drop (cue 1): MAD-normalized deviation below median
    2. IT2FLS interval widening  (cue 2): MAD-normalized deviation above median
    3. EKM left/right asymmetry (cue 3): |left_half - right_half| / total
    4. Cross-ridge 2nd-derivative zero-crossing density (cue 4): LoG with
       magnitude gate to suppress noise-floor sign flips

Step B — fuse_edge_membership: OWA aggregation + single-cue damping

Step C — locate_seam_boundaries_radiometric: run-length detection on the
    smoothed membership curve, with sub-column linear interpolation

Step D — truncate_to_seam_bounds: slice ALL parallel FuzzyResult arrays
    (functional update via dataclasses.replace; s_coords left unchanged)

Optional — refine_boundary_with_depth: post-triangulation Z-curvature trim

References
----------
[1] Yager, R.R. (1988). On ordered weighted averaging aggregation operators
    in multicriteria decisionmaking. IEEE Trans. Syst. Man Cybern., 18(1),
    183-190.
[2] Marr, D. & Hildreth, E. (1980). Theory of edge detection.
    Proc. R. Soc. Lond. B, 207(1167), 187-217.
[3] Canny, J. (1986). A computational approach to edge detection.
    IEEE TPAMI, 8(6), 679-698.
[4] Steger, C. (1998). An unbiased detector of curvilinear structures.
    IEEE TPAMI, 20(2), 113-125.
[5] Wu, D. & Mendel, J.M. (2009). Enhanced Karnik-Mendel algorithms.
    IEEE Trans. Fuzzy Syst., 17(4), 923-934.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from fuzzy.fuzzy_pipeline import FuzzyResult
    from config import BoundaryConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ======================================================================
# Step A — Per-column edge-evidence features
# ======================================================================

def compute_edge_evidence(
    s: np.ndarray,
    uncertainties: np.ndarray,
    y_l: np.ndarray,
    y_c: np.ndarray,
    y_r: np.ndarray,
    profiles: np.ndarray,
    s_coords: np.ndarray,
    steger_sigma: float,
    config: "BoundaryConfig",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute four per-column off-workpiece evidence scalars.

    Parameters
    ----------
    s : np.ndarray, shape (N,)
        Steger ridge strengths (``FuzzyResult.strengths``).
    uncertainties : np.ndarray, shape (N,)
        IT2FLS interval widths (``FuzzyResult.uncertainties``).
        Passed directly — NOT recomputed from y_lower/y_upper.
    y_l : np.ndarray, shape (N,)
        EKM left bounds (``FuzzyResult.y_lower``).
    y_c : np.ndarray, shape (N,)
        EKM center positions (``FuzzyResult.y_centers``).
    y_r : np.ndarray, shape (N,)
        EKM right bounds (``FuzzyResult.y_upper``).
    profiles : np.ndarray, shape (N, M)
        Cross-ridge intensity profiles (``FuzzyResult.profiles``).
    s_coords : np.ndarray, shape (M,)
        Spatial offsets along Steger normals (``FuzzyResult.s_coords``).
    steger_sigma : float
        Gaussian smoothing scale for the LoG curvature cue (cue 4).
        Caller resolves ``None`` to ``HardwareConfig.sigma`` before passing.
    config : BoundaryConfig
        Boundary localization parameters.

    Returns
    -------
    e_strength : np.ndarray, shape (N,)  — cue 1
    e_interval : np.ndarray, shape (N,)  — cue 2
    e_asymmetry : np.ndarray, shape (N,)  — cue 3
    e_curvature : np.ndarray, shape (N,)  — cue 4
        All values in [0, 1]; higher = stronger evidence this column
        is off-workpiece / at an edge transition.
    """
    s = np.asarray(s, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    y_l = np.asarray(y_l, dtype=np.float64)
    y_c = np.asarray(y_c, dtype=np.float64)
    y_r = np.asarray(y_r, dtype=np.float64)
    N = len(s)

    if N == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty

    # --- Central-fraction window for MAD baselines (item J) ---
    # Excludes edge columns from corrupting the reference statistic.
    half_tail = int(N * (1.0 - config.mad_central_fraction) / 2)
    lo = max(0, half_tail)
    hi = min(N, N - half_tail)
    if hi <= lo:
        lo, hi = 0, N   # degenerate: too few points, use all

    # ── Cue 1: Ridge-strength evidence ──────────────────────────────────
    # High e_strength → ridge is weaker than the median ridge on this seam
    s_central = s[lo:hi]
    s_med = float(np.median(s_central))
    s_mad = float(np.median(np.abs(s_central - s_med))) * 1.4826
    if s_mad < _EPS:
        e_strength = np.zeros(N, dtype=np.float64)
    else:
        z_s = (s_med - s) / s_mad      # positive → weaker than median
        e_strength = np.clip(z_s / config.z_ref, 0.0, 1.0)

    # ── Cue 2: IT2FLS interval-width evidence ────────────────────────────
    # uncertainties already = |y_upper - y_lower|; passed directly (item A)
    u_central = uncertainties[lo:hi]
    u_med = float(np.median(u_central))
    u_mad = float(np.median(np.abs(u_central - u_med))) * 1.4826
    if u_mad < _EPS:
        e_interval = np.zeros(N, dtype=np.float64)
    else:
        z_u = (uncertainties - u_med) / u_mad   # positive → wider than median
        e_interval = np.clip(z_u / config.z_ref, 0.0, 1.0)

    # ── Cue 3: EKM asymmetry ─────────────────────────────────────────────
    # A clipped/occluded profile at the bevel skews EKM bounds asymmetrically.
    # Mid-plate noise produces nearly symmetric bounds.
    left_half = np.abs(y_c - y_l)
    right_half = np.abs(y_r - y_c)
    denom = left_half + right_half
    diff  = np.abs(left_half - right_half)
    e_asymmetry = np.zeros(N, dtype=np.float64)
    valid = denom > _EPS
    np.divide(diff, denom, out=e_asymmetry, where=valid)
    e_asymmetry = np.clip(e_asymmetry, 0.0, 1.0)

    # ── Cue 4: Cross-ridge LoG curvature / zero-crossing density ─────────
    # A clean Gaussian profile has exactly 2 zero-crossings in d2p
    # (the two inflection points).  Edge-clipped or double-reflection
    # profiles produce extra zero-crossings.  Magnitude gate (item I)
    # prevents noise-floor sign flips from being counted.
    e_curvature = np.zeros(N, dtype=np.float64)
    sigma_eff = max(steger_sigma, 0.5)   # guard against sigma=0

    for i in range(N):
        p = profiles[i, :]
        if np.all(p < _EPS):
            continue
        p_smooth = gaussian_filter1d(p.astype(np.float64), sigma=sigma_eff)
        # Second derivative via two passes of np.gradient (uses s_coords spacing)
        d1 = np.gradient(p_smooth, s_coords)
        d2 = np.gradient(d1, s_coords)
        d2_std = float(np.std(d2))
        if d2_std < _EPS:
            continue
        # Count magnitude-gated sign changes (item I)
        sign_d2 = np.sign(d2)
        # np.where returns indices where sign changes
        change_idx = np.where(np.diff(sign_d2) != 0)[0]
        valid_crossings = 0
        for k in change_idx:
            # At least one side of the crossing must exceed the magnitude floor
            if (abs(d2[k]) > config.zc_eta * d2_std
                    or abs(d2[min(k + 1, len(d2) - 1)]) > config.zc_eta * d2_std):
                valid_crossings += 1
        extra = max(valid_crossings - 2, 0)
        e_curvature[i] = float(np.clip(extra / max(config.zc_ref, 1.0), 0.0, 1.0))

    logger.debug(
        "Stage 6.5 evidence: e_str=[%.3f,%.3f] e_int=[%.3f,%.3f] "
        "e_asym=[%.3f,%.3f] e_curv=[%.3f,%.3f]",
        e_strength.min(), e_strength.max(),
        e_interval.min(), e_interval.max(),
        e_asymmetry.min(), e_asymmetry.max(),
        e_curvature.min(), e_curvature.max(),
    )
    return e_strength, e_interval, e_asymmetry, e_curvature


# ======================================================================
# Step B — OWA fuzzy fusion → mu_edge
# ======================================================================

def fuse_edge_membership(
    e_strength: np.ndarray,
    e_interval: np.ndarray,
    e_asymmetry: np.ndarray,
    e_curvature: np.ndarray,
    config: "BoundaryConfig",
) -> np.ndarray:
    """Fuse four per-column edge-evidence scalars into a single membership.

    Uses Yager (1988) OWA aggregation: ``config.owa_rank_weights[k]``
    applies to the *k-th largest* cue value at each column (NOT to a
    specific named cue).  A single-cue damping factor reduces ``mu_edge``
    when fewer than 2 cues are elevated, guarding against false triggers.

    Parameters
    ----------
    e_strength, e_interval, e_asymmetry, e_curvature : np.ndarray, shape (N,)
        Per-column evidence scalars in [0, 1] from ``compute_edge_evidence``.
    config : BoundaryConfig

    Returns
    -------
    mu_edge : np.ndarray, shape (N,)
        Per-column "off-workpiece" membership in [0, 1].
        High values near the true plate edges, low values mid-plate.
    """
    e_strength  = np.asarray(e_strength,  dtype=np.float64)
    e_interval  = np.asarray(e_interval,  dtype=np.float64)
    e_asymmetry = np.asarray(e_asymmetry, dtype=np.float64)
    e_curvature = np.asarray(e_curvature, dtype=np.float64)
    N = len(e_strength)

    w = np.asarray(config.owa_rank_weights, dtype=np.float64)
    if len(w) < 4:
        # Pad with zeros if fewer than 4 weights provided (defensive)
        w = np.pad(w, (0, 4 - len(w)))
    w = w / w.sum()   # normalise to ensure weights sum to 1.0

    # Stack cues into (N, 4) for vectorised sorting
    cues = np.column_stack([e_strength, e_interval, e_asymmetry, e_curvature])

    # Sort each row descending (OWA: largest cue gets w[0])
    sorted_cues = np.sort(cues, axis=1)[:, ::-1]   # (N, 4) descending

    # OWA aggregation: dot product of sorted cue row with weight vector
    mu_edge = sorted_cues @ w                        # (N,)

    # Single-cue damping: dampen columns where < 2 cues are elevated
    # "< 2" is strictly less-than; exactly 2 elevated cues = no damping (item H)
    n_elevated = np.sum(cues > config.cue_elevation_floor, axis=1)  # (N,)
    damp_mask = n_elevated < 2
    mu_edge[damp_mask] *= config.single_cue_damp

    mu_edge = np.clip(mu_edge, 0.0, 1.0)

    logger.debug(
        "Stage 6.5 mu_edge: min=%.3f  max=%.3f  mean=%.3f  "
        "n_elevated>=2: %d/%d",
        float(mu_edge.min()), float(mu_edge.max()), float(mu_edge.mean()),
        int(np.sum(~damp_mask)), N,
    )
    return mu_edge


# ======================================================================
# Step C — Derivative-based boundary localization
# ======================================================================

def locate_seam_boundaries_radiometric(
    mu_edge: np.ndarray,
    x_coords: np.ndarray,
    config: "BoundaryConfig",
) -> Tuple[int, int, float, float]:
    """Locate the two plate boundaries in the along-seam mu_edge signal.

    Scans for sustained runs of above-threshold membership from both ends
    (Canny-style run-length gating applied to the 1-D seam axis).
    Sub-column interpolation refines boundary positions to fractional
    column coordinates.

    Parameters
    ----------
    mu_edge : np.ndarray, shape (N,)
        Per-column off-workpiece membership from ``fuse_edge_membership``.
    x_coords : np.ndarray, shape (N,)
        Column indices corresponding to each entry (``FuzzyResult.x_coords``).
    config : BoundaryConfig

    Returns
    -------
    i_start : int
        Integer index (into ``mu_edge`` / all FuzzyResult arrays) of the
        first on-workpiece column.
    i_end : int
        Integer index of the last on-workpiece column (inclusive).
    frac_start : float
        Sub-column boundary position on the left edge, in x_coords units.
    frac_end : float
        Sub-column boundary position on the right edge, in x_coords units.

    Notes
    -----
    If the detected interior region is narrower than
    ``config.min_seam_fraction * N``, the function falls back to the full
    range (0, N-1) and emits a structured WARNING with mu_edge diagnostics
    for post-hoc debugging (item K).
    """
    mu_edge  = np.asarray(mu_edge,  dtype=np.float64)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    N = len(mu_edge)

    if N < 2:
        return 0, max(0, N - 1), float(x_coords[0]) if N > 0 else 0.0, float(x_coords[-1]) if N > 0 else 0.0

    # --- Smooth mu_edge along the seam axis before run detection ---
    mu_smooth = gaussian_filter1d(mu_edge, sigma=config.edge_smooth_sigma)

    thresh = config.edge_mu_thresh
    run_min = config.run_min

    def _find_run_end_from_left(arr: np.ndarray) -> int:
        """Return the first index AFTER a sustained above-threshold run."""
        run = 0
        for idx in range(len(arr)):
            if arr[idx] > thresh:
                run += 1
                if run >= run_min:
                    # Found the run; return the column after the run ends
                    return idx + 1
            else:
                run = 0
        return 0   # no sustained run → no off-workpiece leading segment

    def _find_run_end_from_right(arr: np.ndarray) -> int:
        """Return the last index BEFORE a sustained above-threshold run from the right."""
        run = 0
        for idx in range(len(arr) - 1, -1, -1):
            if arr[idx] > thresh:
                run += 1
                if run >= run_min:
                    return idx - 1
            else:
                run = 0
        return len(arr) - 1   # no sustained run → no off-workpiece trailing segment

    i_start = _find_run_end_from_left(mu_smooth)
    i_end   = _find_run_end_from_right(mu_smooth)

    # Clamp to valid index range
    i_start = max(0, min(i_start, N - 1))
    i_end   = max(0, min(i_end,   N - 1))

    # If inversion (start > end), reset to full range
    if i_start > i_end:
        i_start, i_end = 0, N - 1

    # --- Sub-column interpolation of the threshold crossing (item F) ---
    def _interp_crossing(arr: np.ndarray, idx_before: int, idx_after: int) -> float:
        """Linear interpolation: exact fractional index where arr crosses thresh."""
        idx_before = max(0, min(idx_before, N - 1))
        idx_after  = max(0, min(idx_after,  N - 1))
        v0, v1 = float(arr[idx_before]), float(arr[idx_after])
        if abs(v1 - v0) < _EPS:
            return float(idx_before)
        t = (thresh - v0) / (v1 - v0)
        frac_idx = idx_before + t
        # Map fractional index back to x_coords space via linear interpolation
        frac_idx_clamped = np.clip(frac_idx, 0, N - 1)
        return float(np.interp(frac_idx_clamped, np.arange(N), x_coords))

    frac_start = _interp_crossing(mu_smooth, i_start - 1, i_start)
    frac_end   = _interp_crossing(mu_smooth, i_end,       i_end + 1)

    # --- Sanity / fallback guard with structured diagnostics (item K) ---
    seam_len = i_end - i_start + 1
    kept_fraction = seam_len / N

    if kept_fraction < config.min_seam_fraction or seam_len < 2:
        logger.warning(
            "Stage 6.5 boundary detector fallback: "
            "kept %.1f%% of %d columns (i_start=%d, i_end=%d). "
            "mu_edge diagnostics — min=%.3f  max=%.3f  mean=%.3f  "
            "n_above_thresh=%d. "
            "Returning full range; flag for manual review.",
            100.0 * kept_fraction, N, i_start, i_end,
            float(mu_edge.min()), float(mu_edge.max()), float(mu_edge.mean()),
            int(np.sum(mu_smooth > thresh)),
        )
        return 0, N - 1, float(x_coords[0]), float(x_coords[-1])

    logger.info(
        "Stage 6.5 boundaries: i_start=%d, i_end=%d  "
        "(kept %.1f%% of %d columns; sub-pixel: [%.2f, %.2f]).",
        i_start, i_end, 100.0 * kept_fraction, N, frac_start, frac_end,
    )
    return i_start, i_end, frac_start, frac_end


# ======================================================================
# Step D — Truncation of parallel FuzzyResult arrays
# ======================================================================

def truncate_to_seam_bounds(
    fuzzy_result: "FuzzyResult",
    i_start: int,
    i_end: int,
) -> "FuzzyResult":
    """Slice all per-column parallel arrays in FuzzyResult to [i_start, i_end].

    Uses ``dataclasses.replace`` for a functional (non-mutating) update.

    Slicing contract
    ----------------
    * 1-D arrays  (x_coords, y_centers, …): ``arr[i_start:i_end+1]``
    * profiles (N, M): ``arr[i_start:i_end+1, :]``  — axis 0 only
    * s_coords (M,): **UNCHANGED** — shared across all columns (item C)
    * membership_map (H, W): unchanged — full image membership map
    * methods_used (list): ``lst[i_start:i_end+1]``

    Parameters
    ----------
    fuzzy_result : FuzzyResult
        Input result from the dual-path fuzzy pipeline.
    i_start : int
        Inclusive start index.
    i_end : int
        Inclusive end index.

    Returns
    -------
    FuzzyResult
        New FuzzyResult with all per-column arrays sliced.
    """
    sl = slice(i_start, i_end + 1)
    return dataclasses.replace(
        fuzzy_result,
        x_coords            = fuzzy_result.x_coords[sl],
        y_centers           = fuzzy_result.y_centers[sl],
        y_centers_classical = fuzzy_result.y_centers_classical[sl],
        confidences         = fuzzy_result.confidences[sl],
        uncertainties       = fuzzy_result.uncertainties[sl],
        y_lower             = fuzzy_result.y_lower[sl],
        y_upper             = fuzzy_result.y_upper[sl],
        normals_x           = fuzzy_result.normals_x[sl],
        normals_y           = fuzzy_result.normals_y[sl],
        strengths           = fuzzy_result.strengths[sl],
        profiles            = fuzzy_result.profiles[sl, :],  # axis 0 only (item C)
        s_coords            = fuzzy_result.s_coords,         # shared — NOT sliced
        methods_used        = fuzzy_result.methods_used[i_start:i_end + 1],
        membership_map      = fuzzy_result.membership_map,   # 2-D image — unchanged
    )


# ======================================================================
# Optional Step E — Post-triangulation depth-curvature refinement
# ======================================================================

def refine_boundary_with_depth(
    coords_3d: np.ndarray,
    i_start: int,
    i_end: int,
    config: "BoundaryConfig",
) -> Tuple[int, int]:
    """Trim boundary indices inward using 3D Z-curvature.

    After a first triangulation pass, the Z coordinates provide the most
    physically authoritative edge cue (true plate edge = depth
    discontinuity).  This function can only tighten (never loosen) the
    boundary found by ``locate_seam_boundaries_radiometric``.

    Parameters
    ----------
    coords_3d : np.ndarray, shape (N', 3)
        Already-truncated 3D points from ``LaserTriangulator.pixels_to_3d_batch``.
    i_start : int
        Current start index within ``coords_3d`` (typically 0 after Stage 6.5).
    i_end : int
        Current end index within ``coords_3d`` (typically N'-1).
    config : BoundaryConfig

    Returns
    -------
    i_start, i_end : int, int
        Updated (possibly trimmed-inward) boundary indices.
    """
    coords_3d = np.asarray(coords_3d, dtype=np.float64)
    N = len(coords_3d)
    if N < 4:
        return i_start, i_end

    z = coords_3d[:, 2]

    # Smooth Z with Savitzky-Golay before computing curvature proxy
    win = min(9, N if N % 2 == 1 else N - 1)
    if win < 3:
        return i_start, i_end
    z_smooth = savgol_filter(z, window_length=win, polyorder=2)

    # Second-derivative curvature proxy (Savitzky, Golay 1964)
    kappa = np.gradient(np.gradient(z_smooth))

    # MAD-normalized baseline
    kappa_med = float(np.median(kappa))
    kappa_mad = float(np.median(np.abs(kappa - kappa_med))) * 1.4826
    if kappa_mad < _EPS:
        return i_start, i_end

    # Walk inward from both ends; stop at first column within tolerance
    max_steps = min(config.max_refine_steps, (i_end - i_start) // 4)

    for _ in range(max_steps):
        if i_start >= i_end:
            break
        if abs(kappa[i_start] - kappa_med) / kappa_mad > config.kappa_thresh:
            i_start += 1
        else:
            break

    for _ in range(max_steps):
        if i_end <= i_start:
            break
        if abs(kappa[i_end] - kappa_med) / kappa_mad > config.kappa_thresh:
            i_end -= 1
        else:
            break

    logger.debug(
        "Stage 6.5 depth-refine: trimmed to [%d, %d] (kappa_thresh=%.2f).",
        i_start, i_end, config.kappa_thresh,
    )
    return i_start, i_end
