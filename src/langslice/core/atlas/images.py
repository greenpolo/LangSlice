"""The atlas's own images: its template and every additional reference it ships.

An atlas image is a grayscale volume on the atlas's own grid. BrainGlobe
atlases carry one ``template`` and may carry any number of additional
references (another modality or stain on the same grid, e.g.
``ccfv3augmented_mouse``'s single-animal Nissl, DeMBA's MRI, light-sheet and
STPT templates). LangSlice offers exactly these, under BrainGlobe's own
names, to the agent's pictures and fits; it adds none of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The name of the atlas's main image (BrainGlobe's ``template``).
TEMPLATE = "template"


def atlas_images(atlas: Any) -> tuple[str, ...]:
    """The atlas's image names: ``template``, then its additional references
    in BrainGlobe's order."""
    extra = getattr(atlas, "additional_references", None)
    return (TEMPLATE, *(str(name) for name in (extra.keys() if extra is not None else ())))


def atlas_volume(atlas: Any, name: str) -> np.ndarray:
    """The atlas image *name* (one of :func:`atlas_images`) as a volume on the
    atlas grid; ``ValueError`` names the images the atlas has."""
    if name == TEMPLATE:
        return np.asarray(atlas.template)
    extra = getattr(atlas, "additional_references", None)
    if extra is None or name not in extra.keys():
        raise ValueError(f"{getattr(atlas, 'atlas_name', 'This atlas')} has no image {name!r}; "
                         f"its images: {', '.join(atlas_images(atlas))}")
    return np.asarray(extra[name])
