"""Native BigWarp registrations from calibrated landmark pairs.

The stored spline is the complete mapping, including its affine component.
ABBA resamples using a fixed-to-moving TPS, exactly as BigWarp does.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def spline_world_landmarks(
    spline: dict[str, Any], *, size: tuple[int, int], pixel_size_um: float,
    rotation_deg: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized oriented-section landmarks into centred ABBA mm."""
    dimensions = np.asarray(size, dtype=float)
    if dimensions.shape != (2,) or not np.all(np.isfinite(dimensions) & (dimensions > 0)):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0 or rotation_deg % 90:
        raise ValueError("Spline requires calibrated pixels and quarter-turn orientation")
    if rotation_deg % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * pixel_size_um / 1000.0
    recorded = np.asarray(spline.get("extent_mm"), dtype=float)
    if recorded.shape != (2,) or not np.allclose(recorded, extent, rtol=1e-5, atol=1e-7):
        raise ValueError("Spline physical extent does not match the calibrated ABBA snapshot")
    source = np.asarray(spline.get("source"), dtype=float)
    target = np.asarray(spline.get("target"), dtype=float)
    if (source.ndim != 2 or source.shape[1] != 2 or len(source) < 3
            or target.shape != source.shape or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))):
        raise ValueError("Spline needs at least three finite paired XY landmarks")
    for points in (source, target):
        if (len(np.unique(points, axis=0)) != len(points)
                or np.linalg.matrix_rank(np.column_stack([points, np.ones(len(points))])) < 3):
            raise ValueError("Spline landmarks must be distinct and non-collinear")
    return (source - .5) * extent, (target - .5) * extent


def prepare_spline_registration(abba: Any, source_mm: np.ndarray, target_mm: np.ndarray) -> Any:
    """Prepare a serializable, completed native BigWarp step without opening UI.

    Building and round-tripping it happens before removing any previous step.
    ``setTransform`` marks the native plugin complete; RegisterSliceAction then
    appends it directly instead of calling its interactive ``register`` method.
    """
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    from langslice.integrations.abba import _build_java_tps

    BigWarp = jimport(
        "ch.epfl.biop.registration.sourceandconverter.bigwarp.SacBigWarp2DRegistration"
    )
    ctx = abba.ij.context()
    service = ctx.getService(jimport("org.scijava.plugin.PluginService").class_)
    registration = service.getPlugin(BigWarp.class_).createInstance()
    registration.setScijavaContext(ctx)
    # This pullback is the canonical spline, not a reversed approximation of
    # some separately fitted forward spline. Python rendering uses the same fit.
    registration.setRealTransform(_build_java_tps(target_mm, source_mm))
    registration.setTransform(registration.getTransform())
    registration.setRegistrationParameters(jimport("java.util.HashMap")())
    registration.setRegistrationName("LangSlice landmarks")
    if not registration.isRegistrationDone():
        raise RuntimeError("ABBA could not prepare the landmark spline registration")
    return registration
