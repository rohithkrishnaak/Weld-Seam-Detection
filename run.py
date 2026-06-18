"""
Main Entry Point — Weld Seam Detection Pipeline (No Neural Networks).

Usage examples
--------------
Single image (fuzzy-enhanced)::

    python run.py --input image.png --output results/

Classical mode::

    python run.py --input image.png --output results/ --mode classical

Comparison mode (side-by-side classical vs fuzzy)::

    python run.py --input image.png --output results/ --mode compare

Directory batch::

    python run.py --input data/test/ --output results/ --format csv

With external calibration::

    python run.py --input image.png --output results/ \\
                  --calibration calibration_params.npz
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.inference import WeldSeamDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_calibration(calib_path: str) -> dict:
    """Load calibration parameters from an ``.npz`` file."""
    data = np.load(calib_path, allow_pickle=True)
    result = {
        'camera_matrix': data.get('K'),
        'dist_coeffs': data.get('dist'),
    }
    if 'plane_coeffs' in data:
        result['plane_coeffs'] = data['plane_coeffs']
    if 'T_cam2robot' in data:
        result['T_cam2robot'] = data['T_cam2robot']
    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Weld Seam Detection Pipeline — Dataset-Free Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', required=True,
        help='Input image path or directory of images',
    )
    parser.add_argument(
        '--output', default='output',
        help='Output directory for results (default: output/)',
    )
    parser.add_argument(
        '--calibration', default=None,
        help='Path to calibration .npz file (camera + plane)',
    )
    parser.add_argument(
        '--format', default='json', choices=['json', 'csv'],
        help='Coordinate export format',
    )
    parser.add_argument(
        '--mode', default='fuzzy',
        choices=['classical', 'fuzzy', 'compare'],
        help=(
            'Processing mode: '
            'classical (Steger + CoG), '
            'fuzzy (IT2FLS-enhanced), '
            'compare (side-by-side)'
        ),
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display results in an OpenCV window',
    )
    args = parser.parse_args()

    # ── Load calibration ──
    calib: dict = {}
    if args.calibration:
        calib = load_calibration(args.calibration)
        logger.info("Loaded calibration from %s", args.calibration)

    # ── Create detector ──
    detector = WeldSeamDetector(
        camera_matrix=calib.get('camera_matrix'),
        dist_coeffs=calib.get('dist_coeffs'),
        plane_coeffs=calib.get('plane_coeffs'),
        T_cam2robot=calib.get('T_cam2robot'),
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Run ──
    if input_path.is_dir():
        results = detector.process_directory(
            str(input_path), str(output_path), args.format,
            mode=args.mode,
        )
        logger.info("Batch complete — %d images processed.", len(results))
    else:
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error("Cannot read image: %s", input_path)
            sys.exit(1)

        result = detector.detect(image, mode=args.mode)

        if result.get('pixel_coords') is not None:
            # Save visualization
            vis = result.get('visualization')
            if vis is not None:
                vis_path = str(output_path / 'visualization.png')
                cv2.imwrite(vis_path, vis)
                logger.info("Visualization saved to %s", vis_path)

            # Export coordinates
            from pipeline.exporter import PathExporter
            exporter = PathExporter()
            coords = result.get('robot_coords')
            if coords is None:
                coords = result['pixel_coords']
            exporter.export(
                coords,
                str(output_path / 'seam_coords'),
                format=args.format,
            )

            # Print timing breakdown
            logger.info("Timing breakdown:")
            for k, v in result['timing'].items():
                logger.info("  %-20s %7.2f ms", k, v)

            # Print comparison if in compare mode
            if args.mode == 'compare' and result.get('pixel_coords_classical') is not None:
                fuzzy_coords = result['pixel_coords']
                classical_coords = result['pixel_coords_classical']
                n = min(len(fuzzy_coords), len(classical_coords))
                if n > 0:
                    diff = np.abs(fuzzy_coords[:n, 1] - classical_coords[:n, 1])
                    logger.info("Classical vs Fuzzy comparison:")
                    logger.info("  Mean Y-difference:  %.4f px", np.mean(diff))
                    logger.info("  Max Y-difference:   %.4f px", np.max(diff))

                    fuzzy_result = result.get('fuzzy_result')
                    if fuzzy_result is not None and fuzzy_result.uncertainties is not None:
                        unc = fuzzy_result.uncertainties
                        valid_unc = unc[unc > 0]
                        if len(valid_unc) > 0:
                            logger.info("  Mean uncertainty:   %.4f px", np.mean(valid_unc))

            # Optional display
            if args.show and vis is not None:
                cv2.imshow('Weld Seam Detection', vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            logger.warning("No seam detected in %s.", input_path.name)


if __name__ == '__main__':
    main()
