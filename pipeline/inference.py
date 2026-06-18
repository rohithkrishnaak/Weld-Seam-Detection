"""
Weld Seam Detection Inference Pipeline (No Neural Networks).

Orchestrates the full detection pipeline:
  Image → Classical ROI Extraction → Steger Ridge Detection →
  Dual-Path Processing (Classical vs. Fuzzy-Enhanced) →
  3-D Triangulation → Seam Feature Extraction → Coordinate Export

The pipeline provides two parallel extraction paths:
  - **Classical**: Steger's algorithm + intensity-weighted Center of Gravity
  - **Fuzzy-Enhanced**: IT2FLS (EKM) operating on cross-ridge profiles
    with uncertainty propagation through the 3D triangulation chain.

References
----------
[1] Steger, C. (1998). An unbiased detector of curvilinear structures.
    IEEE TPAMI, 20(2), 113-125.
[2] Wu, D. & Mendel, J.M. (2009). Enhanced Karnik-Mendel algorithms.
    IEEE Trans. Fuzzy Syst., 17(4), 923-934.
[3] Chaira, T. & Ray, A.K. (2010). Fuzzy Image Processing and
    Applications with MATLAB. CRC Press.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import PipelineConfig, DEFAULT_CONFIG
from preprocessing.roi_extractor import ROIExtractor
from fuzzy.fuzzy_pipeline import FuzzyPipeline, FuzzyResult
from extraction.profile_smoother import full_smoothing_pipeline
from geometry.triangulation import LaserTriangulator
from geometry.seam_features import SeamFeatureExtractor
from geometry.coordinate_chain import CoordinateChain
from pipeline.visualizer import Visualizer
from pipeline.exporter import PathExporter

logger = logging.getLogger(__name__)


class WeldSeamDetector:
    """End-to-end weld seam detection and coordinate extraction.

    No neural network dependencies.  All processing is CPU-based using
    classical image processing and fuzzy logic.

    Parameters
    ----------
    config : PipelineConfig, optional
        Master configuration.  Defaults to ``DEFAULT_CONFIG``.
    camera_matrix : np.ndarray, optional
        3×3 camera intrinsic matrix.  Overrides config calibration.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients.  Overrides config calibration.
    plane_coeffs : np.ndarray, optional
        Laser plane equation ``[A, B, C, D]``.  Overrides config calibration.
    T_cam2robot : np.ndarray, optional
        4×4 camera-to-robot homogeneous transform.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        plane_coeffs: Optional[np.ndarray] = None,
        T_cam2robot: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG

        # Classical ROI extraction (replaces DL-based ROI)
        self.roi_extractor = ROIExtractor(config=self.config.roi)

        # Fuzzy pipeline (dual-path: classical + IT2FLS-enhanced)
        self.fuzzy_pipeline = FuzzyPipeline(
            fuzzy_config=self.config.fuzzy,
            it2fls_config=self.config.it2fls,
            hardware_config=self.config.hardware,
            classical_config=self.config.classical,
        )

        # Geometry (only if calibration data is provided)
        self.triangulator: Optional[LaserTriangulator] = None
        self.coord_chain: Optional[CoordinateChain] = None
        self.seam_extractor = SeamFeatureExtractor()

        # Use provided calibration or fall back to config defaults
        K = camera_matrix if camera_matrix is not None else self.config.calibration.camera_matrix
        dist = dist_coeffs if dist_coeffs is not None else self.config.calibration.dist_coeffs
        plane = plane_coeffs if plane_coeffs is not None else self.config.calibration.laser_plane

        if K is not None and plane is not None:
            _dist = dist if dist is not None else np.zeros(5)
            self.triangulator = LaserTriangulator(K, _dist, plane)
            self.coord_chain = CoordinateChain(K, _dist, plane, T_cam2robot)

        # Visualization & export helpers
        self.visualizer = Visualizer()
        self.exporter = PathExporter()

    # ------------------------------------------------------------------
    # Single-image detection
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, mode: str = 'fuzzy') -> Dict:
        """Run the full detection pipeline on a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).
        mode : str
            Processing mode:
            - ``'classical'``: Steger + CoG only
            - ``'fuzzy'``: IT2FLS-enhanced (default)
            - ``'compare'``: Both paths side-by-side

        Returns
        -------
        dict
            Keys: ``pixel_coords``, ``pixel_coords_classical``,
            ``coords_3d``, ``coords_3d_interval``,
            ``robot_coords``, ``seam_features``, ``fuzzy_result``,
            ``timing``, ``visualization``.
        """
        timing: Dict[str, float] = {}

        # Prepare images for each stage:
        #  - ROI extractor needs BGR for chromaticity-based laser isolation
        #  - Steger/fuzzy pipeline needs single-channel with best laser contrast
        #    → use the RED channel (strongest signal for a 650nm laser)
        if image.ndim == 3:
            bgr = image
            # Red channel has strongest signal for 650nm laser
            gray = image[:, :, 2].copy()  # BGR → R channel
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = image.copy()

        # ── Stage 1: Classical ROI Extraction ──
        # Always pass BGR for chromaticity-based extraction
        t0 = time.perf_counter()
        roi_mask = self.roi_extractor.predict(bgr)
        timing['roi_ms'] = (time.perf_counter() - t0) * 1000

        if roi_mask.sum() == 0:
            logger.warning("No laser stripe detected in ROI extraction.")
            return {'pixel_coords': None, 'timing': timing}

        # ── Stage 2: Fuzzy Processing (dual-path) ──
        # Pass red-channel grayscale so Steger detects laser ridges, not texture
        t0 = time.perf_counter()
        fuzzy_result: FuzzyResult = self.fuzzy_pipeline.process(gray, roi_mask)
        timing['fuzzy_ms'] = (time.perf_counter() - t0) * 1000

        if len(fuzzy_result.x_coords) == 0:
            logger.warning("No seam detected in image.")
            return {'pixel_coords': None, 'timing': timing}

        # Select which centers to use based on mode
        if mode == 'classical':
            x_raw = fuzzy_result.x_coords.astype(np.float64)
            y_raw = fuzzy_result.y_centers_classical
        else:
            x_raw = fuzzy_result.x_coords.astype(np.float64)
            y_raw = fuzzy_result.y_centers

        # ── Stage 3: Smoothing ──
        t0 = time.perf_counter()
        if len(x_raw) > 3:
            x_smooth, y_smooth = full_smoothing_pipeline(x_raw, y_raw)
        else:
            x_smooth, y_smooth = x_raw.copy(), y_raw.copy()
        timing['smooth_ms'] = (time.perf_counter() - t0) * 1000

        pixel_coords = np.column_stack([x_smooth, y_smooth])

        # Also smooth classical path for comparison
        pixel_coords_classical = None
        if mode in ('compare', 'fuzzy'):
            x_cl = fuzzy_result.x_coords.astype(np.float64)
            y_cl = fuzzy_result.y_centers_classical
            if len(x_cl) > 3:
                x_cl_s, y_cl_s = full_smoothing_pipeline(x_cl, y_cl)
            else:
                x_cl_s, y_cl_s = x_cl.copy(), y_cl.copy()
            pixel_coords_classical = np.column_stack([x_cl_s, y_cl_s])

        # ── Stage 4: 3-D Triangulation ──
        coords_3d = None
        coords_3d_interval = None
        robot_coords = None
        seam_features = None

        if self.triangulator is not None and len(pixel_coords) > 0:
            t0 = time.perf_counter()

            # Crisp 3D points
            try:
                coords_3d = self.triangulator.pixels_to_3d_batch(pixel_coords)
            except ValueError as e:
                logger.warning("Triangulation failed: %s", e)
                coords_3d = None

            # 3D confidence interval from IT2FLS bounds
            if (mode != 'classical'
                    and fuzzy_result.y_lower is not None
                    and len(fuzzy_result.y_lower) > 0):
                try:
                    interval_results = []
                    # Use original (unsmoothed) ridge + normals for interval
                    n_pts = min(
                        len(fuzzy_result.x_coords),
                        len(fuzzy_result.normals_x),
                    )
                    for i in range(n_pts):
                        p_ridge = np.array([
                            float(fuzzy_result.x_coords[i]),
                            float(fuzzy_result.y_centers[i]),
                        ])
                        n_hat = np.array([
                            float(fuzzy_result.normals_x[i]),
                            float(fuzzy_result.normals_y[i]),
                        ])
                        y_c = 0.0  # Ridge is already at center
                        y_l = float(fuzzy_result.y_lower[i] - fuzzy_result.y_centers[i])
                        y_r = float(fuzzy_result.y_upper[i] - fuzzy_result.y_centers[i])
                        try:
                            P_c, P_l, P_r = self.triangulator.pixels_to_3d_interval(
                                p_ridge, y_c, y_l, y_r, n_hat,
                            )
                            interval_results.append({
                                'center': P_c, 'lower': P_l, 'upper': P_r,
                            })
                        except ValueError:
                            continue
                    if interval_results:
                        coords_3d_interval = interval_results
                except Exception as exc:
                    logger.warning("3D interval computation failed: %s", exc)

            timing['triangulation_ms'] = (time.perf_counter() - t0) * 1000

            # Robot-frame coordinates
            if self.coord_chain is not None and coords_3d is not None:
                robot_coords = self.coord_chain.pixels_to_robot_batch(
                    pixel_coords,
                )

            # Seam feature extraction from 3-D profile
            if coords_3d is not None and len(coords_3d) > 10:
                try:
                    seam_features = self.seam_extractor.extract_features(
                        coords_3d[:, 0], coords_3d[:, 2], joint_type='auto',
                    )
                except Exception as exc:
                    logger.warning("Seam feature extraction failed: %s", exc)

        timing['total_ms'] = sum(timing.values())

        # ── Stage 5: Visualization ──
        conf = fuzzy_result.confidences
        if len(conf) > len(x_smooth):
            conf = conf[:len(x_smooth)]

        vis_image = self.visualizer.draw_seam_overlay(
            image, x_smooth, y_smooth, conf,
        )

        return {
            'pixel_coords': pixel_coords,
            'pixel_coords_classical': pixel_coords_classical,
            'pixel_coords_raw': np.column_stack([
                fuzzy_result.x_coords, fuzzy_result.y_centers,
            ]),
            'coords_3d': coords_3d,
            'coords_3d_interval': coords_3d_interval,
            'robot_coords': robot_coords,
            'seam_features': seam_features,
            'fuzzy_result': fuzzy_result,
            'timing': timing,
            'visualization': vis_image,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    _IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        export_format: str = 'json',
        mode: str = 'fuzzy',
    ) -> List[Dict]:
        """Process every image in a directory.

        Parameters
        ----------
        input_dir : str
            Directory of input images.
        output_dir : str
            Directory for output files (visualizations + coordinates).
        export_format : str
            ``'json'`` or ``'csv'``.
        mode : str
            Processing mode (``'classical'``, ``'fuzzy'``, ``'compare'``).

        Returns
        -------
        list of dict
            Per-image detection results.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in self._IMAGE_EXTS
        )

        results: List[Dict] = []
        for img_path in image_paths:
            logger.info("Processing %s", img_path.name)
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Cannot read %s — skipping.", img_path.name)
                continue

            result = self.detect(image, mode=mode)
            results.append(result)

            # Save visualization
            vis = result.get('visualization')
            if vis is not None:
                cv2.imwrite(
                    str(out_path / f"vis_{img_path.stem}.png"), vis,
                )

            # Export coordinates
            if result.get('pixel_coords') is not None:
                coords = result.get('robot_coords')
                if coords is None:
                    coords = result['pixel_coords']
                self.exporter.export(
                    coords,
                    str(out_path / f"coords_{img_path.stem}"),
                    format=export_format,
                )

        logger.info("Processed %d / %d images", len(results), len(image_paths))
        return results
