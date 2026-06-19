"""
Stage B — Geometric Depth-Gating for Weld Seam Boundary Localization.

Replaces the prior 1-D intensity-derivative and Mamdani-FIS boundary
methods with a purely geometric pipeline that operates on the sub-pixel
coordinate arrays produced by Steger + EKM (Stage A).

The algorithm has four composable stages:

    B.1  Minimum-density pre-filter — rejects isolated glare speckles
         too short to represent physical plate material.
    B.2  RANSAC line fit — extracts the maximal inlier set consistent
         with a linear plate model, discarding background scatter.
    B.3  Jump-distance clustering — partitions the inlier sequence at
         spatial discontinuities that correspond to the weld gap.
    B.4  Plate-cluster selection — identifies the two largest clusters
         (Plate L, Plate R), derives outer bounds, and annotates the
         gap interval.

The entry-point ``localize_seam_boundaries_geometric`` orchestrates
all four stages and returns the topology-preserving truncated arrays
together with boundary metadata consumed by ``SeamFeatureExtractor``.

Failure Modes Addressed
-----------------------
* **False negative at weld gap** — the prior derivative method fired
  at the gap (no signal → large derivative) before reaching the far
  plate.  Jump-distance clustering distinguishes gap discontinuities
  from terminal boundaries: points across the gap are retained in the
  output because the truncation window spans *both* plates.
* **False positive into background** — glare highlights with rounded
  intensity profiles can satisfy the Steger ridge criterion and extend
  the coordinate array beyond the physical metal.  RANSAC rejects these
  as geometric outliers because they lie off the plate line by more than
  the physically derived inlier tolerance δ_plate.

References
----------
[1] Fischler, M. A. & Bolles, R. C. (1981). Random sample consensus:
    A paradigm for model fitting with applications to image analysis
    and automated cartography. *Communications of the ACM*, 24(6),
    381-395.
[2] Bogoslavskyi, I. & Stachniss, C. (2016). Fast range image-based
    segmentation of sparse 3D laser scans for online operation.
    *Proceedings of IEEE/RSJ IROS*, 163-169.
[3] Steger, C. (1998). An unbiased detector of curvilinear structures.
    *IEEE TPAMI*, 20(2), 113-125.
[4] Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in
    Computer Vision* (2nd ed.). Cambridge University Press.
[5] Zou, Y., Chen, J. & Wei, X. (2020). Research on a real-time weld
    seam tracking method for medium-thick plates. *Journal of
    Manufacturing Processes*, 56, 538-551.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Public result dataclass
# ======================================================================

@dataclass
class BoundaryResult:
    """Structured output of the geometric boundary localization pipeline.

    Attributes
    ----------
    u_bounded : np.ndarray, shape (M,)
        Column coordinates truncated to the physical seam extent.
    v_bounded : np.ndarray, shape (M,)
        Corresponding fuzzy-enhanced row centers.
    v_lo_bounded : np.ndarray, shape (M,)
        EKM lower bounds for retained points (uncertainty propagation).
    v_hi_bounded : np.ndarray, shape (M,)
        EKM upper bounds for retained points (uncertainty propagation).
    in_gap_mask : np.ndarray of bool, shape (M,)
        True for points that fall inside the weld gap interval.
        These points are geometrically valid (gap floor / scatter) and
        must be preserved for V-groove feature extraction.
    u_start : float
        Left physical boundary (outer edge of Plate L).
    u_end : float
        Right physical boundary (outer edge of Plate R).
    gap_interval : tuple[float, float]
        (u_gap_left, u_gap_right) — inner edges of the weld gap.
    gap_detected : bool
        True when two distinct plate clusters were identified.
    plate_L_bounds : tuple[float, float]
        (u_start, u_gap_left) column range of the left plate.
    plate_R_bounds : tuple[float, float]
        (u_gap_right, u_end) column range of the right plate.
    ransac_line : tuple[float, float]
        Refined OLS line parameters (slope m*, intercept b*).
    n_inliers : int
        Number of RANSAC inlier points.
    n_discarded_clusters : int
        Clusters rejected as background / glare.
    """

    u_bounded: np.ndarray
    v_bounded: np.ndarray
    v_lo_bounded: np.ndarray
    v_hi_bounded: np.ndarray
    in_gap_mask: np.ndarray

    u_start: float
    u_end: float
    gap_interval: Tuple[float, float]
    gap_detected: bool

    ransac_line: Tuple[float, float]
    n_inliers: int
    n_valid_clusters: int


# ======================================================================
# Internal cluster descriptor (lightweight)
# ======================================================================

@dataclass
class _Cluster:
    """Describes a single contiguous cluster of inlier points."""

    indices: List[int]   # indices into the inlier arrays u_in / v_in
    u_min: float
    u_max: float
    span: float
    size: int


# ======================================================================
# Stage B.1 — Minimum-density pre-filter
# ======================================================================

def filter_minimum_density(
    u: np.ndarray,
    v: np.ndarray,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    L_min: int = 10,
    intra_gap_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove isolated column runs too short to be physical plate material.

    A *run* is a maximal set of consecutive indices in the sorted column
    array ``u`` where no two adjacent entries differ by more than
    ``intra_gap_px`` pixels.  Runs shorter than ``L_min`` are discarded
    as glare speckles or noise.

    Parameters
    ----------
    u, v : np.ndarray, shape (N,)
        Sorted sub-pixel column and row coordinates (u strictly ≥ previous).
    v_lo, v_hi : np.ndarray, shape (N,)
        EKM lower / upper bounds carried through the filter.
    L_min : int
        Minimum contiguous run length to retain.  Default 10 corresponds
        to approximately 1 mm of plate at nominal calibration scale.
    intra_gap_px : float
        Maximum consecutive column gap that is still considered *within*
        a single run.  Default 2.0 px.

    Returns
    -------
    u_f, v_f, v_lo_f, v_hi_f : np.ndarray
        Filtered arrays (may be shorter than input).

    Notes
    -----
    A gap ≤ ``intra_gap_px`` is tolerated within a run because
    ``interpolate_gaps`` in ``profile_smoother.py`` fills gaps up to
    5 px later in the pipeline; we do not want to split runs prematurely.
    """
    N = len(u)
    if N == 0:
        return u.copy(), v.copy(), v_lo.copy(), v_hi.copy()

    # Identify run boundaries
    keep = np.zeros(N, dtype=bool)
    run_start = 0

    for i in range(1, N):
        if (u[i] - u[i - 1]) > intra_gap_px:
            # Close the current run
            run_len = i - run_start
            if run_len >= L_min:
                keep[run_start:i] = True
            run_start = i

    # Close the final run
    run_len = N - run_start
    if run_len >= L_min:
        keep[run_start:N] = True

    n_removed = int(np.sum(~keep))
    if n_removed > 0:
        logger.debug(
            "Stage B.1: removed %d points in runs < L_min=%d.", n_removed, L_min
        )

    return u[keep], v[keep], v_lo[keep], v_hi[keep]


# ======================================================================
# Stage B.2 — RANSAC line fit
# ======================================================================

def ransac_line_fit(
    u: np.ndarray,
    v: np.ndarray,
    delta_plate: float = 4.0,
    n_ransac: Optional[int] = None,
    p_success: float = 0.99,
    outlier_fraction: float = 0.40,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, float, float]:
    """Fit a line v = m·u + b to (u, v) via RANSAC and refine with OLS.

    RANSAC [1] is used rather than direct OLS because the point cloud
    may contain 10–40 % outliers from glare highlights and background
    scatter.  The algorithm returns the maximal inlier set, which
    corresponds to physical plate material.

    Parameters
    ----------
    u, v : np.ndarray, shape (N,)
        Filtered sub-pixel coordinates (already sorted by u).
    delta_plate : float
        Inlier tolerance in pixels.  Points with residual
        |v_i − (m·u_i + b)| ≤ δ_plate are considered inliers.
        Derived from the depth-precision model:

            δZ   = Z² / (f · B) · δp
                 = 200² / (16 × 100) × 0.1  ≈  0.25 mm
            δv   = δZ · f_px / Z
                 ≈ 0.25 × 370 / 200         ≈  0.46 px

        Default 1.0 px ≈ 2σ of the plate measurement noise.
    n_ransac : int, optional
        Override the iteration count.  When *None* (default) it is
        computed from ``p_success`` and ``outlier_fraction`` via the
        standard RANSAC formula [1]:

            n = ⌈ log(1 - p) / log(1 - (1 - ε)²) ⌉

        where ε = ``outlier_fraction`` and the minimal sample size is 2.
    p_success : float
        Target probability of drawing at least one all-inlier sample.
    outlier_fraction : float
        Assumed worst-case fraction of outliers (ε in the RANSAC
        iteration formula).
    rng_seed : int
        Seed for reproducible random sampling.

    Returns
    -------
    inlier_mask : np.ndarray of bool, shape (N,)
        True for inlier points.
    m_star : float
        OLS-refined line slope.
    b_star : float
        OLS-refined line intercept.

    Raises
    ------
    ValueError
        If fewer than 2 points remain after filtering (no line possible).
    """
    N = len(u)
    if N < 2:
        raise ValueError(
            f"RANSAC requires at least 2 points, got {N}."
        )

    # Compute required iteration count from RANSAC theory [1]
    if n_ransac is None:
        # Minimal sample set s = 2 (line)
        inlier_prob = (1.0 - outlier_fraction) ** 2
        inlier_prob = max(inlier_prob, 1e-12)   # guard log(0)
        n_ransac = int(math.ceil(
            math.log(1.0 - p_success) / math.log(1.0 - inlier_prob)
        ))
        n_ransac = max(n_ransac, 100)            # floor at 100 for robustness
        logger.debug("Stage B.2: RANSAC n_iter=%d (ε=%.2f, p=%.2f).",
                     n_ransac, outlier_fraction, p_success)

    rng = np.random.RandomState(rng_seed)
    best_mask = np.zeros(N, dtype=bool)
    best_count = 0

    for _ in range(n_ransac):
        # Minimal sample: 2 distinct indices
        i_a, i_b = rng.choice(N, size=2, replace=False)
        u_a, v_a = float(u[i_a]), float(v[i_a])
        u_b, v_b = float(u[i_b]), float(v[i_b])

        du = u_b - u_a
        if abs(du) < 1e-12:
            continue   # degenerate vertical pair — skip

        m_cand = (v_b - v_a) / du
        b_cand = v_a - m_cand * u_a

        # Signed residuals (absolute value for inlier test)
        residuals = np.abs(v - (m_cand * u + b_cand))
        mask_cand = residuals <= delta_plate
        count_cand = int(np.sum(mask_cand))

        if count_cand > best_count:
            best_count = count_cand
            best_mask = mask_cand

        # Early exit: inlier fraction > 90 %
        if count_cand / N > 0.90:
            break

    if best_count < 2:
        raise ValueError(
            "RANSAC found fewer than 2 inliers — scene geometry is "
            "too cluttered or delta_plate is too tight."
        )

    # ── OLS refinement on full inlier set ──────────────────────────────
    # Minimize Σ (v_i − m·u_i − b)² over {i : best_mask[i]}
    u_in = u[best_mask].astype(np.float64)
    v_in = v[best_mask].astype(np.float64)
    N_in = len(u_in)

    S_u  = float(np.sum(u_in))
    S_v  = float(np.sum(v_in))
    S_uu = float(np.dot(u_in, u_in))
    S_uv = float(np.dot(u_in, v_in))

    denom_ls = N_in * S_uu - S_u ** 2
    if abs(denom_ls) < 1e-12:
        m_star = 0.0
        b_star = S_v / N_in
    else:
        m_star = (N_in * S_uv - S_u * S_v) / denom_ls
        b_star = (S_v - m_star * S_u) / N_in

    # Re-evaluate inlier membership with the refined line
    residuals_final = np.abs(v - (m_star * u + b_star))
    final_mask = residuals_final <= delta_plate
    final_count = int(np.sum(final_mask))

    logger.debug(
        "Stage B.2: RANSAC best=%d, OLS-refined inliers=%d / %d  "
        "line=(m=%.4f, b=%.4f).",
        best_count, final_count, N, m_star, b_star,
    )

    return final_mask, float(m_star), float(b_star)


# ======================================================================
# Stage B.3 — Jump-distance clustering
# ======================================================================

def jump_distance_clustering(
    u_in: np.ndarray,
    gap_min_px: float = 3.0,
    gross_gap_limit: float = 50.0,
) -> List[_Cluster]:
    """Partition an inlier column array into contiguous spatial clusters.

    Physical plate material appears as a dense run of columns; the weld
    gap creates a large spatial discontinuity.  Consecutive column gaps
    Δ_i = u_{i+1} − u_i form a bimodal distribution:

        * Intra-plate gaps:  ≈ 1 px  (dense Steger sampling)
        * Weld gap:          ≥ gap_min_px  (physical gap in metal)
        * Background jumps:  potentially very large (>> gap_min_px)

    The adaptive threshold

        Δ_split = max(gap_min_px,  5 × median(Δ_i < gross_gap_limit))

    separates these modes robustly, adapting to camera resolution and
    working-distance variations without manual re-tuning [2].

    Parameters
    ----------
    u_in : np.ndarray, shape (N_in,)
        Sorted inlier column coordinates (ascending).
    gap_min_px : float
        Minimum expected weld gap width in pixels.  Gaps smaller than
        this cannot be physical weld gaps.  Default 3.0 px.
    gross_gap_limit : float
        Upper limit for identifying "intra-plate" gaps when computing
        the adaptive threshold.  Gaps larger than this are definitely
        inter-cluster breaks and are excluded from the median.

    Returns
    -------
    list of _Cluster
        Each cluster describes a contiguous run of inlier points.
        Singleton clusters (size < 2) are discarded.
    """
    N_in = len(u_in)
    if N_in == 0:
        return []
    if N_in == 1:
        return [_Cluster(indices=[0], u_min=float(u_in[0]),
                         u_max=float(u_in[0]), span=0.0, size=1)]

    delta = u_in[1:] - u_in[:-1]   # shape (N_in - 1,)

    # Adaptive threshold: 5× median intra-plate gap, floored at gap_min_px
    small = delta[delta < gross_gap_limit]
    median_intra = float(np.median(small)) if len(small) > 0 else 1.0
    delta_split = max(gap_min_px, 5.0 * median_intra)

    logger.debug(
        "Stage B.3: median_intra=%.3f px, Δ_split=%.3f px.",
        median_intra, delta_split,
    )

    # Identify boundary positions (indices into u_in, not delta)
    split_positions = np.where(delta > delta_split)[0] + 1  # index of first point in new cluster

    # Build cluster ranges [start, end)
    boundaries = [0] + split_positions.tolist() + [N_in]
    clusters: List[_Cluster] = []

    for k in range(len(boundaries) - 1):
        a = boundaries[k]
        b = boundaries[k + 1]
        if (b - a) < 2:
            continue   # discard singletons
        clusters.append(_Cluster(
            indices=list(range(a, b)),
            u_min=float(u_in[a]),
            u_max=float(u_in[b - 1]),
            span=float(u_in[b - 1] - u_in[a]),
            size=b - a,
        ))

    logger.debug(
        "Stage B.3: %d clusters identified (Δ_split=%.2f px).",
        len(clusters), delta_split,
    )
    return clusters


# ======================================================================
# Stage B.4 — Plate-cluster selection and physical bound derivation
# ======================================================================

def find_global_extent_and_gap(
    clusters: List[_Cluster],
    span_min: float = 20.0,
    gap_min_px: float = 3.0,
) -> Tuple[float, float, float, float, bool, int]:
    """Identify the global bounding extent and the weld gap from valid clusters.

    Discard any cluster with span < span_min. The bounding extent covers
    all remaining clusters. The weld gap is identified as the largest jump
    between consecutive valid clusters.

    Parameters
    ----------
    clusters : list of _Cluster
        Output of ``jump_distance_clustering``.
    span_min : float
        Minimum cluster horizontal span to be considered valid plate material.
    gap_min_px : float
        Minimum gap width required for ``gap_detected`` to be True.

    Returns
    -------
    u_start  : float — left outer boundary (global minimum u)
    u_end    : float — right outer boundary (global maximum u)
    u_gap_left  : float — left edge of the weld gap
    u_gap_right : float — right edge of the weld gap
    gap_detected : bool
    n_valid_clusters : int
    """
    if len(clusters) == 0:
        raise ValueError("No clusters provided.")

    valid_clusters = [c for c in clusters if c.span >= span_min]
    
    if len(valid_clusters) == 0:
        # Fallback to the single largest cluster
        largest = max(clusters, key=lambda c: c.span)
        valid_clusters = [largest]
        logger.warning(
            "Stage B.4: no cluster met span_min=%.1f — using single largest cluster.",
            span_min
        )
        
    # Sort valid clusters left to right by their minimum column position
    valid_clusters = sorted(valid_clusters, key=lambda c: c.u_min)
    K = len(valid_clusters)
    
    u_start = min(c.u_min for c in valid_clusters)
    u_end = max(c.u_max for c in valid_clusters)
    
    gap_detected = False
    u_gap_left = (u_start + u_end) / 2.0
    u_gap_right = u_gap_left
    
    if K >= 2:
        gap_widths = []
        for k in range(K - 1):
            w_k = valid_clusters[k + 1].u_min - valid_clusters[k].u_max
            gap_widths.append(w_k)
            
        k_star = int(np.argmax(gap_widths))
        w_star = gap_widths[k_star]
        
        if w_star >= gap_min_px:
            u_gap_left = valid_clusters[k_star].u_max
            u_gap_right = valid_clusters[k_star + 1].u_min
            gap_detected = True

    logger.debug(
        "Stage B.4/B.5/B.6: u_start=%.1f, u_end=%.1f, "
        "gap=[%.1f, %.1f], gap_detected=%s, valid_clusters=%d.",
        u_start, u_end, u_gap_left, u_gap_right, gap_detected, K
    )

    return (u_start, u_end, u_gap_left, u_gap_right, gap_detected, K)


# ======================================================================
# Main entry point — Stage B orchestrator
# ======================================================================

def localize_seam_boundaries_geometric(
    u_raw: np.ndarray,
    v_raw: np.ndarray,
    v_lower: Optional[np.ndarray] = None,
    v_upper: Optional[np.ndarray] = None,
    *,
    L_min: int = 10,
    delta_plate: float = 4.0,
    n_ransac: Optional[int] = None,
    gap_min_px: float = 3.0,
    intra_gap_px: float = 2.0,
    span_min: float = 20.0,
) -> BoundaryResult:
    """Localize seam boundaries geometrically (RANSAC + jump-distance).

    Orchestrates Stages B.0 through B.7 as specified in the Stage B
    algorithm document.  The truncation is *topology-preserving*: tail
    points beyond the physical plate boundaries are removed, but all
    points within the seam extent — including the weld gap — are
    retained.

    Parameters
    ----------
    u_raw : np.ndarray, shape (N,)
        Sub-pixel column coordinates from Steger + EKM (``x_coords``
        field of ``FuzzyResult``).  Need not be pre-sorted; sorting is
        applied internally in Stage B.0.
    v_raw : np.ndarray, shape (N,)
        Corresponding fuzzy-enhanced row centers (``y_centers``).
    v_lower : np.ndarray, shape (N,), optional
        EKM left bounds (``y_lower`` field of ``FuzzyResult``).
        Propagated through truncation for downstream uncertainty use.
        When *None*, defaults to a copy of ``v_raw``.
    v_upper : np.ndarray, shape (N,), optional
        EKM right bounds (``y_upper`` field of ``FuzzyResult``).
        When *None*, defaults to a copy of ``v_raw``.
    L_min : int
        Minimum contiguous run length for Stage B.1 pre-filter.
    delta_plate : float
        RANSAC inlier tolerance in pixels (Stage B.2).
    n_ransac : int, optional
        Override RANSAC iteration count (Stage B.2).
    gap_min_px : float
        Minimum gap width in pixels to declare a physical weld gap
        (Stages B.3, B.4, B.5).
    intra_gap_px : float
        Column-spacing tolerance for treating consecutive points as
        within the same run (Stage B.1).

    Returns
    -------
    BoundaryResult
        Complete boundary localization result.  See the dataclass
        docstring for field descriptions.

    Raises
    ------
    ValueError
        If the input arrays are empty, inconsistently shaped, or if
        RANSAC fails to find a geometric consensus.
    """
    # ── Stage B.0 — Sort and validate ─────────────────────────────────
    u_raw = np.asarray(u_raw, dtype=np.float64).ravel()
    v_raw = np.asarray(v_raw, dtype=np.float64).ravel()

    N = len(u_raw)
    if N == 0:
        raise ValueError("Stage B: empty input arrays.")
    if len(v_raw) != N:
        raise ValueError(
            f"Stage B: u_raw length {N} ≠ v_raw length {len(v_raw)}."
        )

    # Default EKM bounds to the crisp center if not supplied
    if v_lower is None:
        v_lower = v_raw.copy()
    else:
        v_lower = np.asarray(v_lower, dtype=np.float64).ravel()

    if v_upper is None:
        v_upper = v_raw.copy()
    else:
        v_upper = np.asarray(v_upper, dtype=np.float64).ravel()

    if len(v_lower) != N or len(v_upper) != N:
        raise ValueError(
            "Stage B: v_lower / v_upper must match u_raw length."
        )

    # Sort all arrays by column coordinate
    order = np.argsort(u_raw, kind='stable')
    u  = u_raw[order]
    v  = v_raw[order]
    vl = v_lower[order]
    vh = v_upper[order]

    if N < 4:
        raise ValueError(
            f"Stage B: need at least 4 points, got {N}."
        )

    logger.info("Stage B.0: %d input points (sorted).", N)

    # ── Stage B.1 — Minimum-density pre-filter ────────────────────────
    u_f, v_f, vl_f, vh_f = filter_minimum_density(
        u, v, vl, vh, L_min=L_min, intra_gap_px=intra_gap_px,
    )
    N_f = len(u_f)
    logger.info("Stage B.1: %d points remain after density filter.", N_f)

    if N_f < 4:
        raise ValueError(
            "Stage B.1: too few points remain after density filter. "
            "Check L_min or the quality of Steger detection."
        )

    # ── Stage B.2 — RANSAC line fit ───────────────────────────────────
    inlier_mask, m_star, b_star = ransac_line_fit(
        u_f, v_f,
        delta_plate=delta_plate,
        n_ransac=n_ransac,
    )

    u_in  = u_f[inlier_mask]
    v_in  = v_f[inlier_mask]
    vl_in = vl_f[inlier_mask]
    vh_in = vh_f[inlier_mask]
    N_in  = len(u_in)

    logger.info(
        "Stage B.2: %d RANSAC inliers / %d filtered points  "
        "(line m=%.4f, b=%.4f).",
        N_in, N_f, m_star, b_star,
    )

    if N_in < 4:
        raise ValueError(
            f"Stage B.2: RANSAC inlier count ({N_in}) too small for "
            "reliable clustering."
        )

    # ── Stage B.3 — Jump-distance clustering ──────────────────────────
    clusters = jump_distance_clustering(u_in, gap_min_px=gap_min_px)
    logger.info("Stage B.3: %d clusters found.", len(clusters))

    if len(clusters) == 0:
        raise ValueError("Stage B.3: no clusters found in inlier set.")

    # ── Stage B.4/B.5/B.6 — Select valid clusters and derive global bounds ───────────
    (u_start, u_end,
     u_gap_left, u_gap_right,
     gap_detected, K) = find_global_extent_and_gap(clusters, span_min=span_min, gap_min_px=gap_min_px)

    logger.info(
        "Stage B.4/B.5/B.6: u_start=%.1f, u_end=%.1f, "
        "gap=[%.1f, %.1f], gap_detected=%s.",
        u_start, u_end, u_gap_left, u_gap_right, gap_detected,
    )

    # ── Stage B.6 — Topology-preserving truncation ────────────────────
    # Apply the physical bounds to the FULL sorted raw array (not just
    # the filtered/inlier subset) to preserve all points in the gap
    # region, even if they were marked as RANSAC outliers.  Weak
    # detections inside the gap are geometrically valid (gap-floor
    # scatter) and must not be discarded.

    bound_mask = (u >= u_start) & (u <= u_end)

    u_bounded  = u[bound_mask]
    v_bounded  = v[bound_mask]
    vl_bounded = vl[bound_mask]
    vh_bounded = vh[bound_mask]

    M = len(u_bounded)
    if M == 0:
        raise ValueError(
            "Stage B.6: no points remain in the truncation window "
            f"[{u_start:.1f}, {u_end:.1f}]. "
            "Check plate-cluster selection."
        )

    # ── Stage B.7 — Gap topology annotation ───────────────────────────
    in_gap_mask = (u_bounded >= u_gap_left) & (u_bounded <= u_gap_right)

    logger.info(
        "Stage B.7: %d / %d raw points retained (%.1f%%), "
        "%d in weld gap.",
        M, N, 100.0 * M / N, int(np.sum(in_gap_mask)),
    )

    return BoundaryResult(
        u_bounded    = u_bounded,
        v_bounded    = v_bounded,
        v_lo_bounded = vl_bounded,
        v_hi_bounded = vh_bounded,
        in_gap_mask  = in_gap_mask,
        u_start      = u_start,
        u_end        = u_end,
        gap_interval = (u_gap_left, u_gap_right),
        gap_detected = gap_detected,
        ransac_line  = (m_star, b_star),
        n_inliers    = N_in,
        n_valid_clusters = K,
    )
