"""The ``nissl`` atlas image: a Nissl template aligned to the Allen CCFv3.

The Allen Institute's own Nissl volume (the one ABBA ships) is misaligned
with the CCFv3 annotation (Piluso et al., Imaging Neuroscience 2025,
doi:10.1162/imag_a_00565). The same paper's Blue Brain population-averaged
Nissl template (734 brains) is aligned to it, and BrainGlobe packages it as
the reference image of ``ccfv3augmented_mouse_25um`` (atlas version 1.0).

That atlas is the CCFv3 grid extended at the front and back: its AP axis is
566 voxels of 25 um against the CCFv3's 528, and the CCFv3 sits at AP index
14 inside it, 0.35 mm from its anterior edge (tissue outlines agree with
``allen_mouse_25um`` at Dice 0.997 there; DV and ML are identical). So a
point at CCFv3 millimetres (AP, DV, ML) is that point plus 0.35 mm AP in the
augmented atlas, and the template is sampled there for any Allen mouse
(CCFv3) atlas resolution. It is downloaded by BrainGlobe on first use.

Outside the run atlas's brain the template is set to zero (the averaged
template carries a faint glow beyond the tissue).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from langslice.core.oblique import plane_index_coordinates
from langslice.core.space import Plane, atlas_space_context

#: The BrainGlobe atlas whose reference image is the aligned Nissl template.
NISSL_ATLAS = "ccfv3augmented_mouse_25um"
#: Its (AP, DV, ML) shape and voxel size; anything else is refused.
NISSL_SHAPE = (566, 320, 456)
NISSL_VOXEL_MM = 0.025
#: Where the CCFv3 begins along the augmented atlas's AP axis.
AP_OFFSET_MM = 0.35
#: The Allen CCFv3 extent in mm (AP, DV, ML) an atlas must cover.
CCFV3_EXTENT_MM = (13.2, 8.0, 11.4)

_volumes: dict[str, np.ndarray] = {}
_lock = threading.Lock()


def _default_loader(name: str) -> Any:
    from langslice.core.atlas import load_atlas

    return load_atlas(name)


@dataclass(frozen=True)
class NisslAtlas:
    """The aligned Nissl template, read through *loader* (a BrainGlobe atlas
    by name; downloaded on first use)."""

    loader: Callable[[str], Any] = _default_loader

    @staticmethod
    def compatible(atlas: Any) -> bool:
        """True for an asr atlas covering exactly the Allen CCFv3 extent."""
        try:
            context = atlas_space_context(atlas)
        except Exception:  # noqa: BLE001 - a non-BrainGlobe atlas has no Nissl
            return False
        if context.orientation != "asr":
            return False
        extent = np.asarray(context.shape) * np.asarray(context.resolution_um) / 1000.0
        return bool(np.allclose(extent, CCFV3_EXTENT_MM, atol=0.026))

    def volume(self) -> np.ndarray:
        """The template, (AP, DV, ML) uint16, loaded once per process."""
        with _lock:
            held = _volumes.get(NISSL_ATLAS)
            if held is None:
                held = np.asarray(self.loader(NISSL_ATLAS).reference)
                if held.shape != NISSL_SHAPE:
                    raise ValueError(f"{NISSL_ATLAS}: expected shape {NISSL_SHAPE}, got "
                                     f"{held.shape}; the CCFv3 offset was measured for v1.0")
                _volumes[NISSL_ATLAS] = held
            return held

    def sample_plane(
        self,
        atlas: Any,
        position_mm: float,
        plane: Plane = "coronal",
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> np.ndarray:
        """The template on *atlas*'s display-oriented plane grid, float32.

        The array has exactly ``annotation_slice``'s shape at the same
        arguments, so a linear placement for that grid places it too.
        """
        if not self.compatible(atlas):
            raise ValueError("The nissl atlas image needs an Allen mouse (CCFv3) atlas")
        context = atlas_space_context(atlas)
        resolution_mm = np.asarray(context.resolution_um, dtype=np.float64) / 1000.0
        coords = plane_index_coordinates(atlas, position_mm, plane, pitch_deg, yaw_deg)
        millimetres = (coords + 0.5) * resolution_mm[:, None, None]
        millimetres[0] += AP_OFFSET_MM
        index = millimetres / NISSL_VOXEL_MM - 0.5
        volume = self.volume()
        lo = np.maximum(np.floor(index.reshape(3, -1).min(axis=1)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(index.reshape(3, -1).max(axis=1)).astype(int) + 2,
                        volume.shape)
        if np.any(hi <= lo):
            return np.zeros(coords.shape[1:], dtype=np.float32)
        block = np.asarray(volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
        sampled = map_coordinates(block, index - lo[:, None, None], order=1,
                                  mode="constant", cval=0.0)
        inside = np.rint(coords).astype(np.intp)
        annotation = np.asarray(atlas.annotation)
        for axis, size in enumerate(annotation.shape):
            np.clip(inside[axis], 0, size - 1, out=inside[axis])
        brain = annotation[inside[0], inside[1], inside[2]] > 0
        return np.where(brain, sampled, 0.0).astype(np.float32)
