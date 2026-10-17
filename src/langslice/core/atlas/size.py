"""How big an atlas's brain is next to the adult mouse's: one factor per atlas.

The deformable fit's distances (its working pixels, margins, smoothing and
neighbourhoods) and the positioning ruler's margins were tuned on adult
mouse sections. A rat brain or a developing mouse brain is a different size,
so those distances are multiplied by :func:`brain_scale`: the cube root of
the atlas brain's volume over the adult Allen mouse's. The volume is the
annotated (non-zero) part of the annotation volume, counted on a grid of
about :data:`SAMPLE_UM` (:func:`annotated_volume_mm3`); the cube root of a
volume is a length that ignores how the volume is padded, oriented or
stretched along one axis.

The factor is rounded to :data:`SCALE_STEP`, so every atlas of the adult
CCFv3 brain, at any voxel size (``allen_mouse_10um`` to ``_100um``,
``ccfv3augmented_mouse_25um``), is exactly 1 and its fits are the ones
tuned on it (unrounded, 0.98 to 1.01 across them). Measured factors,
unrounded in brackets: ``whs_sd_rat_39um`` 1.70 (1.69; its annotation
includes some spinal cord and cranial nerve), ``swc_female_rat_50um`` 1.40
(1.42), DeMBA P4 0.70 (0.72), P10 0.90 (0.91), P14 0.95 (0.97).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from langslice.core.atlas.disk_cache import cached_volume

#: The adult Allen mouse brain's annotated volume (``allen_mouse_25um``, as
#: :func:`annotated_volume_mm3` counts it): the brain every distance was
#: tuned on, factor 1.
REFERENCE_VOLUME_MM3 = 505.4
#: The factor is rounded to a multiple of this.
SCALE_STEP = 0.05
#: The annotation is counted every this many micrometres (at least every voxel).
SAMPLE_UM = 100.0

_scales: dict[tuple[Any, ...], float] = {}


def annotated_volume_mm3(annotation: Any, resolution_um: Any) -> float:
    """The volume of the non-zero voxels of *annotation*, in cubic millimetres,
    counted every ``SAMPLE_UM`` (at least every voxel) along each axis."""
    resolution = np.asarray(resolution_um, dtype=np.float64)
    steps = [max(1, int(SAMPLE_UM // float(r))) for r in resolution]
    sampled = np.asarray(annotation)[tuple(slice(None, None, k) for k in steps)]
    voxel_mm3 = float(np.prod(resolution * np.asarray(steps) / 1000.0))
    return float(np.count_nonzero(sampled)) * voxel_mm3


def brain_volume_mm3(atlas: Any) -> float:
    """*atlas*'s annotated volume (:func:`annotated_volume_mm3`), kept in the
    atlas disk cache (``core.atlas.disk_cache``) once measured."""
    measured = cached_volume(atlas, "brain_volume_mm3", lambda: np.array(
        [annotated_volume_mm3(atlas.annotation, atlas.resolution)], dtype=np.float64))
    return float(measured[0])


def scale_for_volume(volume_mm3: float) -> float:
    """The factor of a brain of *volume_mm3*: the cube root of its ratio to
    :data:`REFERENCE_VOLUME_MM3`, rounded to :data:`SCALE_STEP` (never 0)."""
    if not math.isfinite(volume_mm3) or volume_mm3 <= 0:
        raise ValueError(f"An atlas brain volume must be positive, got {volume_mm3!r}")
    ratio = (volume_mm3 / REFERENCE_VOLUME_MM3) ** (1.0 / 3.0)
    return max(SCALE_STEP, round(round(ratio / SCALE_STEP) * SCALE_STEP, 2))


def brain_scale(atlas: Any) -> float:
    """How many times the adult mouse brain's linear size *atlas*'s brain is
    (module text): 1.0 for the adult CCFv3 brain. Measured once per atlas."""
    metadata = getattr(atlas, "metadata", None)
    key = (getattr(atlas, "atlas_name", None),
           metadata.get("version") if isinstance(metadata, dict) else None,
           tuple(float(r) for r in atlas.resolution), tuple(getattr(atlas, "shape", ()) or ()))
    if key not in _scales:
        _scales[key] = scale_for_volume(brain_volume_mm3(atlas))
    return _scales[key]


__all__ = ["REFERENCE_VOLUME_MM3", "SAMPLE_UM", "SCALE_STEP", "annotated_volume_mm3",
           "brain_scale", "brain_volume_mm3", "scale_for_volume"]
