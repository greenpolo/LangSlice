"""Exact conversion of normalized snapshot affines to ABBA's centred XY frame."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized_to_abba_affine(
    params: Any,
    *,
    size: tuple[int, int],
    pixel_size_um: float,
    rotation_deg: int = 0,
) -> np.ndarray:
    """Return the 3×4 world-mm transform of a centred, calibrated snapshot.

    ``size`` is the original exported image's full width and height; the
    quarter-turn swaps its axes before the normalized affine is applied.
    ``pixel_size_um`` is the export calibration, not the resized preview's.

    Stored six-number transforms describe the oriented SECTION frame. They
    already contain pivot displacement and shear, including when parameters
    were chosen on a larger padded display canvas. Recovering them from the
    rounded physical knobs instead would lose both.

    ABBA's XY axes and the snapshot pixel axes increase in the same direction.
    Conjugating the normalized transform by ``world = extent * (fraction-.5)``
    therefore carries the full affine over directly, with no extra sign flip.
    The slicing axis remains unchanged.
    """
    normalized = np.asarray(params, dtype=np.float64)
    if normalized.size != 6 or not np.all(np.isfinite(normalized)):
        raise ValueError("The snapshot affine must contain six finite parameters")
    normalized = normalized.reshape(2, 3)
    dimensions = np.asarray(size, dtype=np.float64)
    if dimensions.shape != (2,) or not np.all(np.isfinite(dimensions) & (dimensions > 0)):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be positive and finite")
    if rotation_deg % 90:
        raise ValueError("Snapshot orientation must be a multiple of 90 degrees")
    if rotation_deg % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * float(pixel_size_um) / 1000.0
    linear = normalized[:, :2] * extent[:, None] / extent[None, :]
    translation = extent * normalized[:, 2] + linear @ (extent / 2.0) - extent / 2.0
    result = np.eye(3, 4, dtype=np.float64)
    result[:2, :2] = linear
    result[:2, 3] = translation
    return result
