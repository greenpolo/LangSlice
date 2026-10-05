"""ABBA's cached Allen CCFv3 volumes (Nissl, ARA, label borders), read directly.

ABBA ships the Allen mouse atlas as a BigDataViewer HDF5 file plus an XML
(``~/cached_atlas/ccf2017-mod65000-border-centered-mm-bc.h5`` and
``mouse_brain_ccfv3p1.xml``). Setups: s00 NISSL, s01 LABELS_BORDER, s02 ARA,
s03 LABELS_MOD_65000 (int16; ids wrap, so it is never used as labels here).
Each setup is a 10 um pyramid (factors 1, 2, 4, 8) stored as ``cells`` in
(z, y, x) order, where BDV x is AP (1320), y is DV (800) and z is ML (1140).

Index correspondence with BrainGlobe ``allen_mouse_*``: transposing a level-0
``cells`` array to (x, y, z) gives (AP, DV, ML) and reproduces BrainGlobe's
``allen_mouse_10um`` reference bit for bit — but that reference (and the
annotation) are exactly left-right symmetric, so that match cannot tell an
identity ML order from a mirrored one. The ML order was settled instead by
(1) the XML: every setup shares one affine, x -> 0.01*x, y -> 0.01*y,
z -> 11.4 - 0.01*z, so ABBA expresses its ML direction in the transform and
leaves the voxel data in Allen's native order; (2) an asymmetric check:
ABBA's Nissl correlates with the Allen Institute's own ``ara_nissl_50.nrrd``
(LPS-labelled, native index order) with a positive antisymmetric part
(r = 0.26 to 0.59 over seven coronal levels), i.e. the
same ML order, not mirrored; (3) BrainGlobe builds ``allen_mouse`` from the
same Allen arrays with an identity transform. So h5 index k on the ML axis is
BrainGlobe ML index k. What remains unverified is only (3), BrainGlobe's own
packaging, which no image test on a symmetric atlas can check. Hemisphere
NAMES differ between conventions (Allen documents PIR, BrainGlobe labels
the same array ``asr``, ABBA calls ML world < 5.7 mm "Left"); co-location is
all this reader needs and all it claims.

Voxel centres: Allen grids put voxel i's centre at (i + 0.5) * resolution,
which is how a BrainGlobe index at any resolution is mapped onto the 10 um
grid. BDV pyramid level voxel j of factor f is centred on level-0 index
j * f + (f - 1) / 2. ``h5py`` is imported lazily; it ships in the
``registration`` extra.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.ndimage import map_coordinates

from langslice.core.oblique import plane_index_coordinates
from langslice.core.space import Plane, atlas_space_context

#: Where ABBA caches its Allen atlas.
DEFAULT_CACHE_DIR = Path.home() / "cached_atlas"
H5_NAME = "ccf2017-mod65000-border-centered-mm-bc.h5"
XML_NAME = "mouse_brain_ccfv3p1.xml"
#: The channels this reader serves, by ABBA setup name.
CHANNELS = ("NISSL", "ARA", "LABELS_BORDER")
#: ABBA's grid: 10 um voxels, (AP, DV, ML) shape.
VOXEL_MM = 0.01
SHAPE_AP_DV_ML = (1320, 800, 1140)
#: The XML affine this reader's ML reasoning depends on (row-major 3x4).
EXPECTED_AFFINE = (0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, -0.01, 11.4)


@dataclass(frozen=True)
class AbbaAtlas:
    """Paths to ABBA's cached Allen atlas and the setup id of each channel."""

    h5_path: Path
    setups: dict[str, int]

    @classmethod
    def find(cls, directory: str | Path | None = None) -> AbbaAtlas | None:
        """The cached atlas in *directory* (default ``~/cached_atlas``), or None."""
        root = Path(directory) if directory is not None else DEFAULT_CACHE_DIR
        h5_path, xml_path = root / H5_NAME, root / XML_NAME
        if not h5_path.is_file() or not xml_path.is_file():
            return None
        return cls(h5_path=h5_path, setups=_read_setups(xml_path))

    def compatible(self, atlas: Any) -> bool:
        """True for an Allen mouse atlas covering ABBA's exact physical extent."""
        context = atlas_space_context(atlas)
        if context.orientation != "asr":
            return False
        extent = np.asarray(context.shape) * np.asarray(context.resolution_um) / 1000.0
        expected = np.asarray(SHAPE_AP_DV_ML) * VOXEL_MM
        return bool(np.allclose(extent, expected, atol=0.026))

    def sample_plane(
        self,
        channel: str,
        atlas: Any,
        position_mm: float,
        plane: Plane = "coronal",
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> np.ndarray:
        """*channel* on the BrainGlobe atlas's display-oriented plane grid.

        The returned array has exactly ``annotation_slice``'s shape at the same
        arguments, so a linear placement for that grid places it too. The
        coarsest pyramid level no coarser than the atlas's own voxels is read,
        and only the slab the plane passes through.
        """
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
        if not self.compatible(atlas):
            raise ValueError("ABBA's cached volume only matches an asr Allen mouse atlas")
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "Reading ABBA's atlas needs h5py: install LangSlice's 'registration' extra"
            ) from exc
        context = atlas_space_context(atlas)
        resolution_mm = np.asarray(context.resolution_um, dtype=np.float64) / 1000.0
        coords = plane_index_coordinates(atlas, position_mm, plane, pitch_deg, yaw_deg)
        # BrainGlobe index -> physical mm (voxel centres) -> ABBA level-0 index.
        level0 = (coords + 0.5) * resolution_mm[:, None, None] / VOXEL_MM - 0.5
        setup = self.setups[channel]
        with h5py.File(self.h5_path, "r") as handle:
            resolutions = cast(Any, handle[f"s{setup:02d}/resolutions"])
            factors = np.asarray(resolutions[()], dtype=np.float64)
            level = _choose_level(factors[:, 0], float(resolution_mm.min()))
            factor = float(factors[level, 0])
            cells = cast(Any, handle[f"t00000/s{setup:02d}/{level}/cells"])
            index = (level0 - (factor - 1.0) / 2.0) / factor  # (AP, DV, ML) at this level
            shape_xyz = cells.shape[::-1]
            lo = np.maximum(np.floor(index.reshape(3, -1).min(axis=1)).astype(int) - 1, 0)
            hi = np.minimum(np.ceil(index.reshape(3, -1).max(axis=1)).astype(int) + 2, shape_xyz)
            if np.any(hi <= lo):
                return np.zeros(coords.shape[1:], dtype=np.float32)
            block = np.asarray(cells[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]], dtype=np.float32)
        volume = np.ascontiguousarray(block.transpose(2, 1, 0))  # (AP, DV, ML)
        local = index - lo[:, None, None]
        sampled = map_coordinates(volume, local, order=1, mode="constant", cval=0.0)
        return np.asarray(sampled, dtype=np.float32)


def _choose_level(factors: np.ndarray, atlas_voxel_mm: float) -> int:
    """The coarsest pyramid level whose voxels are no larger than the atlas's."""
    fine_enough = [i for i, f in enumerate(factors) if f * VOXEL_MM <= atlas_voxel_mm + 1e-9]
    return max(fine_enough, key=lambda i: factors[i]) if fine_enough else 0


def _read_setups(xml_path: Path) -> dict[str, int]:
    """Setup id per channel name, refusing an XML whose affine differs from ABBA's.

    The ML correspondence documented above was derived for that affine; any
    other file would need its own derivation, not a silent assumption.
    """
    root = ET.parse(xml_path).getroot()
    setups: dict[str, int] = {}
    for setup in root.iter("ViewSetup"):
        name = (setup.findtext("name") or "").strip()
        identifier = setup.findtext("id")
        if name and identifier is not None:
            setups[name] = int(identifier)
    for registration in root.iter("ViewRegistration"):
        affine = registration.findtext("ViewTransform/affine")
        values = tuple(float(v) for v in (affine or "").split())
        if len(values) != 12 or not np.allclose(values, EXPECTED_AFFINE):
            raise ValueError(f"{xml_path.name}: unexpected ABBA atlas affine {values}")
    missing = [name for name in CHANNELS if name not in setups]
    if missing:
        raise ValueError(f"{xml_path.name}: missing ABBA setups {missing}")
    return setups
