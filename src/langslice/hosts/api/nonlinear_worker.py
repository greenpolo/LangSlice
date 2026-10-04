"""The ABBA plugin's worker: refine ABBA's current atlas placement.

``nonlinear.abba`` on the engine service. It drives the ABBA registration
plugin's code (:mod:`langslice.hosts.integrations.abba`), so it is a host
module; the JVM-free linear snapshot worker is
:mod:`langslice.doors.api.abba_worker`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langslice.doors.api.abba_worker import _SUPPORTED_ATLASES, Emit, _finite_positive


def run_nonlinear(params: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Refine host placement; return atlas -> histology pairs in fixed-grid pixels."""
    import numpy as np
    import tifffile

    from langslice.hosts.integrations.abba import (
        LangSliceAbbaConfig,
        compute_registration_landmarks,
    )

    coords_path = Path(params["coords_path"]).expanduser().resolve(strict=True)
    histology_path = Path(params["histology_path"]).expanduser().resolve(strict=True)
    if coords_path.suffix.lower() == ".npy":
        coords = np.load(coords_path, allow_pickle=False)
    else:
        coords = np.moveaxis(tifffile.imread(coords_path), 0, -1)
    histology = tifffile.imread(histology_path)
    if histology.ndim == 3 and histology.shape[0] == 1:
        histology = histology[0]
    if (coords.ndim != 3 or coords.shape[-1] != 3 or histology.ndim != 2
            or coords.shape[:2] != histology.shape or not np.isfinite(coords).all()
            or not np.isfinite(histology).all()):
        raise ValueError("Finite AP/DV/ML coordinates and grayscale histology must share one grid")
    config = LangSliceAbbaConfig(**(params.get("config") or {}))
    if config.atlas_name not in _SUPPORTED_ATLASES:
        raise ValueError("The Fiji connector currently requires the Allen Mouse atlas")
    _finite_positive(config.voxel_size_um, "Registration voxel size")
    if not isinstance(config.landmark_grid, int) or config.landmark_grid < 2:
        raise ValueError("Landmark grid must contain at least two points per axis")
    emit({"kind": "log", "message": "Refining ABBA's current atlas placement"})
    source, target = compute_registration_landmarks(coords, histology, config)
    if (source.ndim != 2 or source.shape[1] != 2 or len(source) < 3
            or target.shape != source.shape or not np.isfinite(source).all()
            or not np.isfinite(target).all()):
        raise ValueError("Registration returned invalid paired landmarks")
    return {"source_points": source.tolist(), "target_points": target.tolist(),
            "coordinate_frame": "fixed_grid_pixels"}
