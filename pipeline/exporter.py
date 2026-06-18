"""
Coordinate Export for Robot Path Planning.

Exports extracted weld seam coordinates in JSON and CSV formats
for consumption by robot controllers (ABB, FANUC, KUKA, UR).

Output includes:
- Ordered waypoints with optional orientation vectors
- Metadata (joint type, confidence, extraction method)
- Timing information for quality control
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PathExporter:
    """Export weld path coordinates to various formats."""

    def export(
        self,
        coordinates: np.ndarray,
        filepath: str,
        format: str = 'json',
        metadata: Optional[Dict] = None,
    ) -> str:
        """Export coordinates to file.

        Parameters
        ----------
        coordinates : np.ndarray
            Nx2 (pixel) or Nx3 (3-D) coordinate array.
        filepath : str
            Output file path (extension added automatically).
        format : str
            ``'json'`` or ``'csv'``.
        metadata : dict, optional
            Additional metadata to include in the export.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        if format == 'json':
            path = self._export_json(coordinates, filepath + '.json', metadata)
        elif format == 'csv':
            path = self._export_csv(coordinates, filepath + '.csv', metadata)
        else:
            raise ValueError(f"Unknown export format: {format}")
        return path

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def _export_json(
        self,
        coords: np.ndarray,
        path: str,
        meta: Optional[Dict],
    ) -> str:
        ndim = coords.shape[1] if coords.ndim > 1 else 1
        dim_labels = ['x', 'y', 'z'][:ndim]

        data: Dict = {
            'format_version': '1.0',
            'num_points': int(len(coords)),
            'dimensions': ndim,
            'dimension_labels': dim_labels,
            'unit': 'mm' if ndim == 3 else 'pixels',
            'path': [],
        }

        for i, row in enumerate(coords):
            point = {}
            for j, label in enumerate(dim_labels):
                val = float(row[j]) if coords.ndim > 1 else float(row)
                point[label] = round(val, 6)
            point['index'] = i
            data['path'].append(point)

        if meta:
            data['metadata'] = _make_json_serializable(meta)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Exported %d points to %s", len(coords), out)
        return str(out.resolve())

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(
        self,
        coords: np.ndarray,
        path: str,
        meta: Optional[Dict],
    ) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        ncols = coords.shape[1] if coords.ndim > 1 else 1
        headers = ['x', 'y', 'z'][:ncols]

        with open(out, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write metadata as comments
            if meta:
                for k, v in meta.items():
                    writer.writerow([f"# {k}={v}"])
            writer.writerow(headers)
            for row in coords:
                vals = row if coords.ndim > 1 else [row]
                writer.writerow([f"{v:.6f}" for v in vals])

        logger.info("Exported %d points to %s", len(coords), out)
        return str(out.resolve())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
