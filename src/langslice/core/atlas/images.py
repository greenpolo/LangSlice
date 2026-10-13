"""The atlas's own images: its template and every additional reference it ships.

An atlas image is a grayscale volume on the atlas's own grid. BrainGlobe
atlases carry one ``template`` and may carry any number of additional
references (another modality or stain on the same grid, e.g.
``ccfv3augmented_mouse``'s single-animal Nissl, DeMBA's MRI, light-sheet and
STPT templates). LangSlice offers exactly these, under BrainGlobe's own
names, to the agent's pictures and fits. One temporary exception: an Allen
CCFv3 atlas also gets :data:`BORROWED_NISSL` (below).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.core.space import atlas_space_context

#: The name of the atlas's main image (BrainGlobe's ``template``).
TEMPLATE = "template"

#: TEMPORARY, on Harry Carey's advice (BrainGlobe): the Allen CCFv3 atlases
#: ship no Nissl, but ``ccfv3augmented_mouse_25um`` is the CCFv3 grid extended
#: 0.35 mm at the front, and its template is the Blue Brain averaged Nissl
#: aligned to the CCFv3 annotation (Piluso et al. 2025). An atlas covering
#: exactly the CCFv3 extent is offered that template as one more image,
#: resampled onto its own grid. Delete when BrainGlobe ships a Nissl with them.
BORROWED_NISSL = "ccfv3augmented_nissl"
_AUGMENTED = "ccfv3augmented_mouse_25um"
_AUGMENTED_VOXEL_MM = 0.025
_AUGMENTED_AP_OFFSET_MM = 0.35
_CCFV3_EXTENT_MM = (13.2, 8.0, 11.4)
_borrowed: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}


def _takes_borrowed_nissl(atlas: Any) -> bool:
    try:
        context = atlas_space_context(atlas)
    except Exception:  # noqa: BLE001 - not a BrainGlobe-style atlas
        return False
    extent = np.asarray(context.shape) * np.asarray(context.resolution_um) / 1000.0
    return (context.orientation == "asr" and getattr(atlas, "atlas_name", "") != _AUGMENTED
            and bool(np.allclose(extent, _CCFV3_EXTENT_MM, atol=0.026)))


def _borrowed_nissl(atlas: Any) -> np.ndarray:
    """The augmented atlas's template on *atlas*'s grid, zero outside its brain."""
    from scipy.ndimage import affine_transform

    from langslice.core.atlas.core import load_atlas

    annotation = np.asarray(atlas.annotation)
    key = (str(getattr(atlas, "atlas_name", "?")), annotation.shape)
    if key not in _borrowed:
        source = np.asarray(load_atlas(_AUGMENTED).template)
        step = np.asarray(atlas_space_context(atlas).resolution_um) / 1000.0 / _AUGMENTED_VOXEL_MM
        # Voxel centres: augmented index = step * index + 0.5 * step - 0.5 (+ the AP offset).
        offset = 0.5 * step - 0.5 + np.array([_AUGMENTED_AP_OFFSET_MM / _AUGMENTED_VOXEL_MM, 0, 0])
        volume = affine_transform(source, step, offset=offset, output_shape=annotation.shape,
                                  order=1, output=source.dtype)
        volume[annotation == 0] = 0  # the averaged template glows faintly beyond the tissue
        _borrowed.clear()  # one atlas at a time: a 10 um volume is gigabytes
        _borrowed[key] = volume
    return _borrowed[key]


def atlas_images(atlas: Any) -> tuple[str, ...]:
    """The atlas's image names: ``template``, then its additional references
    in BrainGlobe's order."""
    extra = getattr(atlas, "additional_references", None)
    names = (TEMPLATE, *(str(name) for name in (extra.keys() if extra is not None else ())))
    return (*names, BORROWED_NISSL) if _takes_borrowed_nissl(atlas) else names


def atlas_volume(atlas: Any, name: str) -> np.ndarray:
    """The atlas image *name* (one of :func:`atlas_images`) as a volume on the
    atlas grid; ``ValueError`` names the images the atlas has."""
    if name == TEMPLATE:
        return np.asarray(atlas.template)
    if name == BORROWED_NISSL and _takes_borrowed_nissl(atlas):
        return _borrowed_nissl(atlas)
    extra = getattr(atlas, "additional_references", None)
    if extra is None or name not in extra.keys():
        raise ValueError(f"{getattr(atlas, 'atlas_name', 'This atlas')} has no image {name!r}; "
                         f"its images: {', '.join(atlas_images(atlas))}")
    return np.asarray(extra[name])
