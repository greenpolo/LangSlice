"""Read an atlas image as STalign's and emlddmm's 3-D fits take it: at 50 um.

Both engines fit against the whole atlas volume, averaged down to 50 um (the
atlas voxel of their published examples). The volume is the core's
(:func:`langslice.core.atlas.images.atlas_volume`, the borrowed Nissl of an
Allen CCFv3 atlas included) averaged in blocks of whole voxels, and kept on
disk by the core's atlas cache for the next call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from langslice.core.atlas.disk_cache import cached_volume
from langslice.core.atlas.images import atlas_volume
from langslice.core.atlas.load import load_atlas

ATLAS_UM = 50.0
SLAB_BLOCKS = 8


@dataclass(frozen=True)
class AveragedImage:
    """Hold an atlas image averaged down to :data:`ATLAS_UM`.

    Parameters
    ----------
    image
        The block-averaged image, float32, in the atlas's axis order.
    axes
        The centre of each averaged voxel along each axis, in atlas um
        (voxel ``i`` of the atlas has its centre at ``i * resolution``).
    centre
        The centre of the full-resolution volume, in atlas um.
    resolution
        The atlas's voxel size along each axis, in um.
    """

    image: np.ndarray
    axes: list[np.ndarray]
    centre: np.ndarray
    resolution: np.ndarray


def averaged_image(atlas: str, image: str) -> AveragedImage:
    """Read one image of an atlas at :data:`ATLAS_UM`.

    Parameters
    ----------
    atlas
        The BrainGlobe atlas's name, as the job names it.
    image
        ``template``, one of the atlas's images (the borrowed Nissl
        included) or ``nissl``.

    Returns
    -------
    AveragedImage
        The image averaged in blocks of whole voxels (a partial block at an
        axis's far edge is dropped), its axes, the volume's centre and the
        atlas's resolution.
    """
    source = load_atlas(atlas)
    steps = [float(step) for step in source.resolution]
    factors = [max(1, round(ATLAS_UM / step)) for step in steps]
    shape = np.asarray(source.annotation.shape, np.float64)
    kept = cached_volume(
        source,
        f"{image}_mean{'x'.join(map(str, factors))}",
        lambda: _block_mean(atlas_volume(source, image), factors),
    )
    axes = [
        (np.arange(n) * k + (k - 1) / 2.0) * step
        for n, k, step in zip(kept.shape, factors, steps, strict=True)
    ]
    resolution = np.asarray(steps, np.float64)
    return AveragedImage(kept, axes, (shape - 1) / 2.0 * resolution, resolution)


def _block_mean(data: np.ndarray, factors: list[int]) -> np.ndarray:
    """Average each whole block of ``factors`` voxels, a slab of blocks at a time
    (a 10 um volume is gigabytes)."""
    sizes = [int(size) // k * k for size, k in zip(data.shape, factors, strict=True)]
    blocks = [size // k for size, k in zip(sizes, factors, strict=True)]
    out = np.empty(blocks, np.float32)
    for first in range(0, blocks[0], SLAB_BLOCKS):
        last = min(first + SLAB_BLOCKS, blocks[0])
        slab = np.asarray(
            data[first * factors[0] : last * factors[0], : sizes[1], : sizes[2]], np.float32
        )
        shaped = slab.reshape(
            last - first, factors[0], blocks[1], factors[1], blocks[2], factors[2]
        )
        out[first:last] = shaped.mean((1, 3, 5))
    return out
