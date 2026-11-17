"""Read an atlas image as STalign's and emlddmm's 3-D fits take it: at 50 um.

Both engines fit against the whole atlas volume, averaged down to 50 um (the
atlas voxel of their published examples), so it is read here from BrainGlobe
in LangSlice's process, as the job reads its atlas, and kept on disk under
:data:`CACHE` for the next call. The borrowed Nissl of an Allen CCFv3 atlas
is built as LangSlice builds it (``core/atlas/images.py``).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile

CACHE = Path.home() / ".cache" / "langslice-engines"
ATLAS_UM = 50.0
TEMPLATE = "template"
BORROWED_NISSL = "ccfv3augmented_nissl"
AUGMENTED = "ccfv3augmented_mouse_25um"
AUGMENTED_VOXEL_UM, AUGMENTED_AP_OFFSET_UM = 25.0, 350.0


class _Atlas(Protocol):
    """Name the parts of a BrainGlobe atlas read here."""

    template: np.ndarray
    annotation: np.ndarray
    additional_references: Mapping[str, np.ndarray]
    resolution: tuple[float, ...]
    metadata: Mapping[str, str | float | list[float] | None]


class _AtlasApi(Protocol):
    """Name BrainGlobe's atlas class (its names are typed as a closed set)."""

    def BrainGlobeAtlas(  # noqa: N802
        self, atlas_name: str, /, *, check_latest: bool
    ) -> _Atlas: ...


class _Ndimage(Protocol):
    """Name the resampling of SciPy's ``ndimage`` used here (no stubs)."""

    def affine_transform(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
        /,
        *,
        offset: np.ndarray,
        output_shape: tuple[int, ...],
        order: int,
    ) -> np.ndarray: ...


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
        ``template``, one of the atlas's additional references, or
        :data:`BORROWED_NISSL`.

    Returns
    -------
    AveragedImage
        The image averaged in blocks of whole voxels, its axes, the
        volume's centre and the atlas's resolution.
    """
    source = _atlas(atlas)
    version = str(source.metadata.get("version", ""))
    CACHE.mkdir(parents=True, exist_ok=True)
    kept = CACHE / f"{atlas}_v{version}__{image}__{int(ATLAS_UM)}um.npz"
    if kept.exists():
        return _kept(kept)
    steps = [float(step) for step in source.resolution]
    factors = [max(1, round(ATLAS_UM / step)) for step in steps]
    resolution = np.asarray(steps, np.float64)
    shape = np.asarray(source.annotation.shape, np.float64)
    if image == BORROWED_NISSL:
        block = _borrowed_nissl(source, steps, factors)
    elif image == TEMPLATE:
        block = _block_mean(np.asarray(source.template, np.float32), factors)
    else:
        data = source.additional_references[image]
        block = _block_mean(np.asarray(data, np.float32), factors)
    axes = [
        (np.arange(n) * k + (k - 1) / 2.0) * step
        for n, k, step in zip(_dims(block), factors, steps, strict=True)
    ]
    centre = (shape - 1) / 2.0 * resolution
    averaged = AveragedImage(block, axes, centre, resolution)
    _keep(kept, averaged)
    return averaged


def _atlas(name: str) -> _Atlas:
    """Open a BrainGlobe atlas as the job does, without its version check."""
    api = cast("_AtlasApi", importlib.import_module("brainglobe_atlasapi"))
    return api.BrainGlobeAtlas(name, check_latest=False)


def _kept(path: Path) -> AveragedImage:
    """Read a volume kept by an earlier call."""
    stored = cast("NpzFile[np.float64]", np.load(path))
    with stored:
        return AveragedImage(
            np.asarray(stored["image"]),
            [np.asarray(stored[f"x{axis}"]) for axis in range(3)],
            np.asarray(stored["centre"]),
            np.asarray(stored["resolution"]),
        )


def _keep(path: Path, averaged: AveragedImage) -> None:
    """Save a volume under a temporary name, then rename it into place.

    A reader in another thread or process never opens half a file.
    """
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npz")
    np.savez(
        partial,
        image=averaged.image,
        x0=averaged.axes[0],
        x1=averaged.axes[1],
        x2=averaged.axes[2],
        centre=averaged.centre,
        resolution=averaged.resolution,
    )
    partial.replace(path)


def _dims(array: np.ndarray) -> tuple[int, ...]:
    """Read an array's shape as integers."""
    return cast("tuple[int, ...]", array.shape)


def _block_mean(data: np.ndarray, factors: list[int]) -> np.ndarray:
    """Average each whole block of ``factors`` voxels.

    A partial block at the far edge of an axis is dropped.
    """
    sizes = _dims(data)
    shape = [size // k * k for size, k in zip(sizes, factors, strict=True)]
    trimmed = data[: shape[0], : shape[1], : shape[2]]
    blocks = (
        shape[0] // factors[0],
        factors[0],
        shape[1] // factors[1],
        factors[1],
        shape[2] // factors[2],
        factors[2],
    )
    means = cast("np.ndarray", trimmed.reshape(blocks).mean((1, 3, 5)))
    return means


def _borrowed_nissl(
    source: _Atlas, steps: list[float], factors: list[int]
) -> np.ndarray:
    """Build the borrowed Nissl as LangSlice does, block-averaged.

    The augmented template is resampled (linear, voxel centres, its AP
    offset) onto the atlas grid and zeroed outside the brain, one slab of
    blocks at a time (a 10 um volume is gigabytes).
    """
    ndimage = cast("_Ndimage", importlib.import_module("scipy.ndimage"))
    template = np.asarray(_atlas(AUGMENTED).template, np.float32)
    annotation = np.asarray(source.annotation)
    step = np.asarray(steps, np.float64) / AUGMENTED_VOXEL_UM
    ap_offset = AUGMENTED_AP_OFFSET_UM / AUGMENTED_VOXEL_UM
    offset = 0.5 * step - 0.5 + np.array([ap_offset, 0.0, 0.0])
    rows, *plane = _dims(annotation)
    slabs: list[np.ndarray] = []
    for slab_index in range(rows // factors[0]):
        first = slab_index * factors[0]
        along = steps[0] / AUGMENTED_VOXEL_UM * slab_index * factors[0]
        shift = np.array([along, 0.0, 0.0])
        slab = ndimage.affine_transform(
            template,
            np.diag(step),
            offset=offset + shift,
            output_shape=(factors[0], *plane),
            order=1,
        )
        slab[annotation[first : first + factors[0]] == 0] = 0
        slabs.append(_block_mean(slab, factors))
    return np.concatenate(slabs, 0)
