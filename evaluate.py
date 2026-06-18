"""
Evaluation Script — Classical vs. Fuzzy-Enhanced Comparison.

Runs the pipeline on a test set (or synthetic profiles) and reports
quantitative comparison between classical and fuzzy-enhanced methods:

  - Classical: Steger's algorithm + intensity-weighted Center of Gravity
  - Fuzzy-Enhanced: IT2FLS (EKM) with uncertainty propagation

Metrics:
  - Center extraction MAE / RMSE (pixels)
  - Uncertainty interval width (fuzzy only)
  - Per-method breakdown
  - Processing time comparison

Usage
-----
With real images::

    python evaluate.py --image_dir data/test/images \\
                       --mask_dir  data/test/masks  \\
                       --output eval_results/

Synthetic benchmark (no images needed)::

    python evaluate.py --synthetic --output eval_results/
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import cv2
import numpy as np

from config import PipelineConfig
from preprocessing.roi_extractor import ROIExtractor
from fuzzy.fuzzy_pipeline import FuzzyPipeline, FuzzyResult
from extraction.profile_smoother import full_smoothing_pipeline
from utils import dice_score, iou_score, boundary_f1, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ------------------------------------------------------------------
# Synthetic profile generation
# ------------------------------------------------------------------

def generate_synthetic_profile(
    length: int = 200,
    center: float = 100.0,
    width: float = 10.0,
    amplitude: float = 200.0,
    noise_std: float = 5.0,
    saturation_level: int = 255,
    profile_type: str = 'gaussian',
    seed: int = None,
) -> Tuple[np.ndarray, float]:
    """Generate a synthetic laser stripe intensity profile.

    Parameters
    ----------
    length : int
        Profile length in pixels.
    center : float
        True sub-pixel center position.
    width : float
        Gaussian σ (or flat-top half-width).
    amplitude : float
        Peak intensity before saturation.
    noise_std : float
        Background Gaussian noise σ.
    saturation_level : int
        Pixel value at which saturation occurs.
    profile_type : str
        ``'gaussian'``, ``'saturated'``, ``'noisy'``, ``'asymmetric'``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    profile : np.ndarray
        1-D intensity profile (float64).
    true_center : float
        Ground-truth sub-pixel center position.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    x = np.arange(length, dtype=np.float64)

    if profile_type == 'gaussian':
        profile = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        profile += rng.normal(0, noise_std, length)
        true_center = center

    elif profile_type == 'saturated':
        # Gaussian that clips at saturation
        raw = amplitude * 1.5 * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        profile = np.minimum(raw, saturation_level)
        profile += rng.normal(0, noise_std, length)
        true_center = center

    elif profile_type == 'noisy':
        profile = (amplitude * 0.3) * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        profile += rng.normal(0, noise_std * 3, length)
        true_center = center

    elif profile_type == 'asymmetric':
        # Asymmetric Gaussian (different widths left/right)
        left = amplitude * np.exp(-((x - center) ** 2) / (2 * (width * 0.7) ** 2))
        right = amplitude * np.exp(-((x - center) ** 2) / (2 * (width * 1.3) ** 2))
        profile = np.where(x < center, left, right)
        profile += rng.normal(0, noise_std, length)
        # True center shifts slightly toward the broader side
        true_center = center + width * 0.05

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    profile = np.clip(profile, 0, 255).astype(np.float64)
    return profile, true_center


def generate_synthetic_image(
    height: int = 200,
    width: int = 400,
    center_y: float = 100.0,
    stripe_width: float = 8.0,
    amplitude: float = 200.0,
    noise_std: float = 5.0,
    saturation_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic laser stripe image with ground-truth mask.

    Parameters
    ----------
    height, width : int
        Image dimensions.
    center_y : float
        Vertical center of the stripe (can vary per column).
    stripe_width : float
        Gaussian σ of the stripe profile.
    amplitude : float
        Peak intensity.
    noise_std : float
        Background noise σ.
    saturation_fraction : float
        Fraction of columns that are saturated.
    seed : int
        Random seed.

    Returns
    -------
    image : np.ndarray
        Grayscale image (uint8).
    gt_centers : np.ndarray
        Per-column ground-truth center Y coordinates.
    """
    rng = np.random.RandomState(seed)

    # Slightly curved center line
    x = np.arange(width, dtype=np.float64)
    gt_centers = center_y + 5.0 * np.sin(2 * np.pi * x / width)

    y = np.arange(height, dtype=np.float64)
    yy, _ = np.meshgrid(y, x, indexing='ij')

    image = np.zeros((height, width), dtype=np.float64)
    for col in range(width):
        cy = gt_centers[col]
        amp = amplitude
        # Make some columns saturated
        if col / width < saturation_fraction:
            amp = amplitude * 1.8  # will clip to 255
        profile = amp * np.exp(-((y - cy) ** 2) / (2 * stripe_width ** 2))
        image[:, col] = profile

    # Add noise
    image += rng.normal(0, noise_std, (height, width))
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, gt_centers


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def compute_center_error(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE and RMSE between predicted and GT center lines.

    Matches predicted columns to the nearest GT column and computes
    the vertical (Y) error.
    """
    errors = []
    for px, py in zip(pred_x, pred_y):
        dists = np.abs(gt_x - px)
        nearest = np.argmin(dists)
        if dists[nearest] <= 2:  # Within 2-pixel column tolerance
            errors.append(abs(py - gt_y[nearest]))

    if len(errors) == 0:
        return {'mae_px': float('inf'), 'rmse_px': float('inf'), 'n_matched': 0}

    errors = np.array(errors)
    return {
        'mae_px': float(np.mean(errors)),
        'rmse_px': float(np.sqrt(np.mean(errors ** 2))),
        'max_error_px': float(np.max(errors)),
        'n_matched': len(errors),
    }


def extract_gt_centers(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground-truth center line from a binary mask.

    Uses column-wise center of gravity on the mask.
    """
    xs, ys = [], []
    for col in range(mask.shape[1]):
        col_data = mask[:, col].astype(np.float64)
        if col_data.sum() < 1:
            continue
        indices = np.arange(len(col_data))
        center = np.sum(indices * col_data) / np.sum(col_data)
        xs.append(col)
        ys.append(center)
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------------
# Synthetic benchmark
# ------------------------------------------------------------------

def run_synthetic_benchmark(output_dir: str) -> Dict:
    """Run comparison on synthetic profiles with known ground truth."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("═══ Running Synthetic Benchmark ═══")

    # Create pipeline
    pipeline = FuzzyPipeline()

    profile_types = ['gaussian', 'saturated', 'noisy', 'asymmetric']
    results_by_type: Dict[str, Dict] = {}

    for ptype in profile_types:
        logger.info("Testing profile type: %s", ptype)

        classical_errors = []
        fuzzy_errors = []
        uncertainties = []
        n_trials = 50

        for trial in range(n_trials):
            # Generate synthetic profile
            center = 100.0 + np.random.uniform(-2, 2)
            profile, true_center = generate_synthetic_profile(
                length=200, center=center, width=8.0,
                amplitude=200.0, noise_std=5.0,
                profile_type=ptype, seed=trial * 100 + hash(ptype) % 1000,
            )

            # Classical: intensity-weighted Center of Gravity
            y_idx = np.arange(len(profile), dtype=np.float64)
            weight_sum = profile.sum()
            if weight_sum > 1e-6:
                cog_center = float(np.dot(y_idx, profile) / weight_sum)
            else:
                cog_center = float(len(profile)) / 2.0
            classical_errors.append(abs(cog_center - true_center))

            # Fuzzy-enhanced: IT2FLS
            from fuzzy.type2_barycentric import IT2FLSExtractor
            it2fls = IT2FLSExtractor()
            s_coords = np.arange(len(profile), dtype=np.float64)
            y_c, y_l, y_r = it2fls.extract_center(profile, s_coords)

            if not np.isnan(y_c):
                fuzzy_errors.append(abs(y_c - true_center))
                uncertainties.append(abs(y_r - y_l))
            else:
                fuzzy_errors.append(float('inf'))
                uncertainties.append(float('inf'))

        classical_errors = np.array(classical_errors)
        fuzzy_errors = np.array(fuzzy_errors)
        uncertainties = np.array(uncertainties)

        # Filter out inf values
        valid_classical = classical_errors[np.isfinite(classical_errors)]
        valid_fuzzy = fuzzy_errors[np.isfinite(fuzzy_errors)]
        valid_unc = uncertainties[np.isfinite(uncertainties)]

        result = {
            'profile_type': ptype,
            'n_trials': n_trials,
            'classical_mae': float(np.mean(valid_classical)) if len(valid_classical) > 0 else float('inf'),
            'classical_rmse': float(np.sqrt(np.mean(valid_classical**2))) if len(valid_classical) > 0 else float('inf'),
            'fuzzy_mae': float(np.mean(valid_fuzzy)) if len(valid_fuzzy) > 0 else float('inf'),
            'fuzzy_rmse': float(np.sqrt(np.mean(valid_fuzzy**2))) if len(valid_fuzzy) > 0 else float('inf'),
            'mean_uncertainty': float(np.mean(valid_unc)) if len(valid_unc) > 0 else float('inf'),
            'fuzzy_valid_fraction': len(valid_fuzzy) / n_trials,
        }

        if result['classical_mae'] > 0:
            result['improvement_pct'] = (
                (result['classical_mae'] - result['fuzzy_mae'])
                / result['classical_mae'] * 100
            )
        else:
            result['improvement_pct'] = 0.0

        results_by_type[ptype] = result

        logger.info(
            "  %s: Classical MAE=%.4f px, Fuzzy MAE=%.4f px, "
            "Improvement=%.1f%%, Uncertainty=%.4f px",
            ptype, result['classical_mae'], result['fuzzy_mae'],
            result['improvement_pct'], result['mean_uncertainty'],
        )

    # Full-image synthetic test
    logger.info("Running full-image synthetic test...")
    image, gt_centers = generate_synthetic_image(seed=42)

    # Create mask via thresholding
    roi_extractor = ROIExtractor()
    mask = roi_extractor.predict(image)

    # Run fuzzy pipeline
    fuzzy_result = pipeline.process(image, mask)

    if len(fuzzy_result.x_coords) > 0:
        # Classical comparison
        gt_x = np.arange(len(gt_centers), dtype=np.float64)
        classical_err = compute_center_error(
            fuzzy_result.x_coords.astype(np.float64),
            fuzzy_result.y_centers_classical,
            gt_x, gt_centers,
        )
        fuzzy_err = compute_center_error(
            fuzzy_result.x_coords.astype(np.float64),
            fuzzy_result.y_centers,
            gt_x, gt_centers,
        )

        image_result = {
            'classical_mae': classical_err['mae_px'],
            'classical_rmse': classical_err['rmse_px'],
            'fuzzy_mae': fuzzy_err['mae_px'],
            'fuzzy_rmse': fuzzy_err['rmse_px'],
            'n_points': len(fuzzy_result.x_coords),
            'methods': dict(Counter(fuzzy_result.methods_used)),
        }
        results_by_type['full_image'] = image_result

        logger.info(
            "Full image: Classical MAE=%.4f px, Fuzzy MAE=%.4f px, "
            "Points=%d",
            classical_err['mae_px'], fuzzy_err['mae_px'],
            len(fuzzy_result.x_coords),
        )

    # Save results
    all_results = {'synthetic_profiles': results_by_type}
    save_metrics(all_results, str(out_path / 'synthetic_benchmark.json'))

    logger.info("═══ Synthetic Benchmark Complete ═══")
    return all_results


# ------------------------------------------------------------------
# Real image evaluation
# ------------------------------------------------------------------

def evaluate(
    image_dir: str,
    mask_dir: str,
    output_dir: str = 'eval_results',
) -> Dict:
    """Run evaluation on a test set with ground-truth masks.

    Parameters
    ----------
    image_dir : str
        Directory with test images.
    mask_dir : str
        Directory with ground-truth binary masks (same filenames).
    output_dir : str
        Output directory for results.

    Returns
    -------
    dict
        Aggregate and per-image metrics.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline components
    roi_extractor = ROIExtractor()
    fuzzy_pipeline = FuzzyPipeline()

    # Collect test images
    img_dir = Path(image_dir)
    msk_dir = Path(mask_dir)
    image_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )

    # Accumulators
    all_classical_mae: List[float] = []
    all_fuzzy_mae: List[float] = []
    all_classical_rmse: List[float] = []
    all_fuzzy_rmse: List[float] = []
    method_counter: Counter = Counter()
    per_image: List[Dict] = []

    for img_path in image_paths:
        mask_path = msk_dir / img_path.name
        if not mask_path.exists():
            logger.warning("No GT mask for %s — skipping.", img_path.name)
            continue

        # Load
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_mask is None:
            continue

        gt_mask_bin = (gt_mask > 127).astype(np.uint8)

        # ── ROI extraction ──
        pred_mask = roi_extractor.predict(image)
        pred_mask_bin = (pred_mask > 127).astype(np.uint8)

        # ── Fuzzy processing ──
        fuzzy_result: FuzzyResult = fuzzy_pipeline.process(image, pred_mask)

        if len(fuzzy_result.x_coords) == 0:
            continue

        # GT center line
        gt_x, gt_y = extract_gt_centers(gt_mask_bin)

        if len(gt_x) == 0:
            continue

        # Classical error
        classical_err = compute_center_error(
            fuzzy_result.x_coords.astype(np.float64),
            fuzzy_result.y_centers_classical,
            gt_x, gt_y,
        )

        # Fuzzy error
        fuzzy_err = compute_center_error(
            fuzzy_result.x_coords.astype(np.float64),
            fuzzy_result.y_centers,
            gt_x, gt_y,
        )

        all_classical_mae.append(classical_err['mae_px'])
        all_fuzzy_mae.append(fuzzy_err['mae_px'])
        all_classical_rmse.append(classical_err['rmse_px'])
        all_fuzzy_rmse.append(fuzzy_err['rmse_px'])

        for m in fuzzy_result.methods_used:
            method_counter[m] += 1

        record = {
            'image': img_path.name,
            'classical_mae': classical_err['mae_px'],
            'classical_rmse': classical_err['rmse_px'],
            'fuzzy_mae': fuzzy_err['mae_px'],
            'fuzzy_rmse': fuzzy_err['rmse_px'],
            'n_points': len(fuzzy_result.x_coords),
            'methods': dict(Counter(fuzzy_result.methods_used)),
        }
        per_image.append(record)

        logger.info(
            "%s — Classical MAE=%.4f, Fuzzy MAE=%.4f px",
            img_path.name, classical_err['mae_px'], fuzzy_err['mae_px'],
        )

    # ── Aggregate ──
    aggregate = {
        'n_images': len(per_image),
        'classical_mae_mean': float(np.mean(all_classical_mae)) if all_classical_mae else 0.0,
        'fuzzy_mae_mean': float(np.mean(all_fuzzy_mae)) if all_fuzzy_mae else 0.0,
        'classical_rmse_mean': float(np.mean(all_classical_rmse)) if all_classical_rmse else 0.0,
        'fuzzy_rmse_mean': float(np.mean(all_fuzzy_rmse)) if all_fuzzy_rmse else 0.0,
        'method_distribution': dict(method_counter),
    }

    if aggregate['classical_mae_mean'] > 0:
        aggregate['improvement_pct'] = (
            (aggregate['classical_mae_mean'] - aggregate['fuzzy_mae_mean'])
            / aggregate['classical_mae_mean'] * 100
        )

    results = {'aggregate': aggregate, 'per_image': per_image}

    save_metrics(results, str(out_path / 'evaluation_results.json'))

    logger.info("═══ Evaluation Summary ═══")
    logger.info("  Images evaluated   : %d", aggregate['n_images'])
    logger.info("  Classical MAE      : %.4f px", aggregate['classical_mae_mean'])
    logger.info("  Fuzzy MAE          : %.4f px", aggregate['fuzzy_mae_mean'])
    logger.info("  Classical RMSE     : %.4f px", aggregate['classical_rmse_mean'])
    logger.info("  Fuzzy RMSE         : %.4f px", aggregate['fuzzy_rmse_mean'])
    if 'improvement_pct' in aggregate:
        logger.info("  Improvement        : %.1f%%", aggregate['improvement_pct'])
    logger.info("  Methods used       : %s", dict(method_counter))

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate Weld Seam Detection: Classical vs Fuzzy-Enhanced',
    )
    parser.add_argument('--image_dir', default=None,
                        help='Directory with test images')
    parser.add_argument('--mask_dir', default=None,
                        help='Directory with ground-truth masks')
    parser.add_argument('--output', default='eval_results',
                        help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                        help='Run synthetic benchmark (no images needed)')
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_benchmark(args.output)
    elif args.image_dir and args.mask_dir:
        evaluate(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output,
        )
    else:
        logger.error(
            "Specify either --synthetic or both --image_dir and --mask_dir"
        )
        raise SystemExit(1)


if __name__ == '__main__':
    main()
