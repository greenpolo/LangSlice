"""The atlas's own images: its template and every additional reference it ships.

An atlas image is a grayscale volume on the atlas's own grid. BrainGlobe
atlases carry one ``template`` and may carry any number of additional
references (another modality or stain on the same grid, e.g.
``ccfv3augmented_mouse``'s single-animal Nissl, DeMBA's MRI, light-sheet and
STPT templates). LangSlice offers exactly these, under BrainGlobe's own
names, to the agent's pictures and fits. One addition: an Allen
CCFv3 atlas also gets :data:`BORROWED_NISSL` (below).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.core.space import atlas_space_context

#: The name of the atlas's main image (BrainGlobe's ``template``).
TEMPLATE = "template"
#: The alias every engine's ``atlas_image`` defaults to: the atlas's Nissl
#: where it has or borrows one, else its template (:func:`default_image`).
NISSL = "nissl"

#: On Harry Carey's advice (BrainGlobe), kept as the standing design: the Allen CCFv3 atlases
#: ship no Nissl, but ``ccfv3augmented_mouse_25um`` is the CCFv3 grid extended
#: 0.35 mm at the front, and its template is the Blue Brain averaged Nissl
#: aligned to the CCFv3 annotation (Piluso et al. 2025). An atlas covering
#: exactly the CCFv3 extent, in the CCFv3 coordinate space, is offered that
#: template as one more image, resampled onto its own grid.
BORROWED_NISSL = "ccfv3augmented_nissl"
AUGMENTED_ATLAS = "ccfv3augmented_mouse_25um"
AUGMENTED_VOXEL_MM = 0.025
#: The augmented atlas's front extension beyond the CCFv3 extent: a CCFv3 plane
#: sits this far (14 planes of 25 um) behind the same plane's index in it.
#: Checked by annotation agreement (best match at 14 planes, 91 % of a mid plane).
AUGMENTED_AP_OFFSET_MM = 0.35
_CCFV3_EXTENT_MM = (13.2, 8.0, 11.4)
#: BrainGlobe's name for the Allen CCFv3 coordinate space. The Allen CCFv2
#: atlases share CCFv3's grid but not its anatomy (their borders sit 100-300 um
#: from CCFv3's), so a grid match alone does not qualify an atlas.
CCFV3_SPACE = "allen-adult-mouse-ccf-space"
_borrowed: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}


def _coordinate_space(atlas: Any) -> str | None:
    """The coordinate space the atlas's BrainGlobe metadata names, if any."""
    metadata = getattr(atlas, "metadata", None)
    space = metadata.get("coordinate_space") if isinstance(metadata, dict) else None
    return str(space.get("name")) if isinstance(space, dict) and space.get("name") else None


def _takes_borrowed_nissl(atlas: Any) -> bool:
    try:
        context = atlas_space_context(atlas)
    except Exception:  # noqa: BLE001 - not a BrainGlobe-style atlas
        return False
    space = _coordinate_space(atlas)
    if space is not None and space != CCFV3_SPACE:
        return False
    extent = np.asarray(context.shape) * np.asarray(context.resolution_um) / 1000.0
    return (context.orientation == "asr" and getattr(atlas, "atlas_name", "") != AUGMENTED_ATLAS
            and bool(np.allclose(extent, _CCFV3_EXTENT_MM, atol=0.026)))


def _borrowed_nissl(atlas: Any) -> np.ndarray:
    """The augmented atlas's template on *atlas*'s grid, zero outside its brain."""
    from scipy.ndimage import affine_transform

    from langslice.core.atlas.core import load_atlas

    annotation = np.asarray(atlas.annotation)
    key = (str(getattr(atlas, "atlas_name", "?")), annotation.shape)
    if key not in _borrowed:
        source = np.asarray(load_atlas(AUGMENTED_ATLAS).template)
        step = np.asarray(atlas_space_context(atlas).resolution_um) / 1000.0 / AUGMENTED_VOXEL_MM
        # Voxel centres: augmented index = step * index + 0.5 * step - 0.5 (+ the AP offset).
        offset = 0.5 * step - 0.5 + np.array([AUGMENTED_AP_OFFSET_MM / AUGMENTED_VOXEL_MM, 0, 0])
        # The scale as a diagonal matrix: a 1-D one makes SciPy warn on every call.
        volume = affine_transform(source, np.diag(step), offset=offset,  # pyright: ignore[reportArgumentType]  # scipy takes a per-axis offset; its stub says float
                                  output_shape=annotation.shape, order=1, output=source.dtype)
        volume[annotation == 0] = 0  # the averaged template glows faintly beyond the tissue
        _borrowed.clear()  # one atlas at a time: a 10 um volume is gigabytes
        _borrowed[key] = volume
    return _borrowed[key]


def _own_images(atlas: Any) -> tuple[str, ...]:
    """The atlas's real image names, without the :data:`NISSL` alias."""
    extra = getattr(atlas, "additional_references", None)
    names = (TEMPLATE, *(str(name) for name in (extra.keys() if extra is not None else ())))
    return (*names, BORROWED_NISSL) if _takes_borrowed_nissl(atlas) else names


def atlas_images(atlas: Any) -> tuple[str, ...]:
    """The atlas's image names: ``template``, then its additional references
    in BrainGlobe's order, then the alias :data:`NISSL` (always accepted;
    :func:`default_image` says what it stands for)."""
    return (*_own_images(atlas), NISSL)


def default_image(atlas: Any) -> str:
    """The real image name :data:`NISSL` stands for on *atlas*: the borrowed
    Nissl (:data:`BORROWED_NISSL`) where the atlas takes it; the template of
    ``ccfv3augmented_mouse_25um`` (itself a Nissl); an additional reference
    whose name contains ``nissl``; else ``template``."""
    if _takes_borrowed_nissl(atlas):
        return BORROWED_NISSL
    if getattr(atlas, "atlas_name", "") == AUGMENTED_ATLAS:
        return TEMPLATE
    for name in _own_images(atlas):
        if "nissl" in name.lower():
            return name
    return TEMPLATE


def nissl_note(atlas: Any, asked: str) -> str | None:
    """What the reply says when *asked* is the default :data:`NISSL` and the atlas's template IS
    a Nissl (``ccfv3augmented_mouse_25um``): the request was honoured, not replaced. None for
    any other name or atlas."""
    if (asked == NISSL and default_image(atlas) == TEMPLATE
            and getattr(atlas, "atlas_name", "") == AUGMENTED_ATLAS):
        return ("your default \"nissl\" request was honoured: this atlas's template is a "
                "Nissl (the Blue Brain averaged Nissl)")
    return None


def resolve_image(atlas: Any, name: str) -> str:
    """*name* as a real image name of *atlas*: :data:`NISSL` becomes
    :func:`default_image`; any other name must be one of the atlas's images.
    ``ValueError`` names the images the atlas has."""
    if name == NISSL:
        return default_image(atlas)
    if name not in _own_images(atlas):
        raise ValueError(f"{getattr(atlas, 'atlas_name', 'This atlas')} has no image {name!r}; "
                         f"its images: {', '.join(atlas_images(atlas))}")
    return name


def atlas_volume(atlas: Any, name: str) -> np.ndarray:
    """The atlas image *name* (one of :func:`atlas_images`) as a volume on the
    atlas grid; ``ValueError`` names the images the atlas has."""
    name = resolve_image(atlas, name)
    if name == TEMPLATE:
        return np.asarray(atlas.template)
    if name == BORROWED_NISSL:
        return _borrowed_nissl(atlas)
    extra = getattr(atlas, "additional_references", None)
    return np.asarray(extra[name])  # pyright: ignore[reportOptionalSubscript]  # resolve_image checked it
