"""Silhouette-based affine preview registration for the GUI 3D viewer.

When the user locks an AP position, we want a brain-silhouette warped slice to
appear in the 3D viewer in well under a second — long before the full
nano-banana / Elastix pipeline finishes. An earlier intensity-based Elastix
attempt was both slow (~15s cold-start) and fragile across modalities (the NCC
metric chases noise when comparing histology staining patterns to atlas Nissl).
Both problems vanish if we register *shapes* instead of intensities.

The fit itself lives in :mod:`langslice.affine` (shared with the whole-brain
``transforms`` step). All this module adds is the viewer's output format: warp
the section RGB through the fitted affine and use the atlas root silhouette as
the alpha channel, so the 3D viewer renders a brain-shaped sheet rather than a
rectangular slab.

Cost: ~150ms in a warm Python process, no itk, no Elastix.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.affine import silhouette_affine
from langslice.atlas import load_atlas
from langslice.space import Plane

logger = logging.getLogger(__name__)


def quick_affine_register(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    out_path: Path,
) -> dict[str, Any]:
    """Silhouette-based affine alignment of slice → atlas root.

    Returns ``{warped_slice_path, elapsed_s, silhouette_iou}``. The warped
    PNG is RGBA at ~512px long-edge, alpha = atlas root silhouette, so the
    3D viewer can drop it in unchanged."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    fit = silhouette_affine(
        image,
        atlas=load_atlas(atlas_name),
        position_mm=position_mm,
        plane=plane,
    )

    # Linear interpolation for smoothness; clip to the atlas silhouette via
    # alpha so the viewer gets a brain-shaped sheet.
    warped_rgb = cv2.warpAffine(
        fit.slice_rgb, fit.matrix, fit.size,
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_rgba = np.dstack([warped_rgb, fit.atlas_mask])
    Image.fromarray(warped_rgba, mode="RGBA").save(out_path)

    elapsed = time.perf_counter() - started
    logger.info(
        "quick_affine_register: %s @ %.2fmm (%s) IoU=%.3f sign=%s -> %s in %.2fs",
        atlas_name, position_mm, plane, fit.iou, fit.sign_pattern, out_path, elapsed,
    )
    return {
        "warped_slice_path": str(out_path.resolve()),
        "elapsed_s": round(elapsed, 2),
        "silhouette_iou": round(fit.iou, 3),
    }
