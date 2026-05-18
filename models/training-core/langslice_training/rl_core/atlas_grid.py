"""Pre-rendered atlas slice cache for RLVR rollouts.

Slicing the atlas on every fetch_atlas tool call would dominate rollout time.
We pay the cost once at training startup: for each (atlas, plane) pair in the
training set, render reference slices on a fixed 0.05 mm grid across the valid
range and stash the resulting PIL images in memory. fetch_atlas then turns into
a dictionary lookup with a small position-quantization step.

Reuses ``langslice_harness.atlas`` for slice extraction; does not reimplement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from PIL import Image

Plane = Literal["coronal", "sagittal", "horizontal"]


def canonicalize_atlas_name(atlas_name: str) -> str:
    try:
        from langslice_harness.atlas.core import canonicalize_atlas_name as _canonicalize
    except ModuleNotFoundError:
        return atlas_name

    return _canonicalize(atlas_name)


def load_atlas(atlas_name: str):
    from langslice_harness.atlas.core import load_atlas as _load_atlas

    return _load_atlas(atlas_name)


def get_position_range_mm(atlas, *, plane: Plane):  # noqa: ANN001
    from langslice_harness.atlas.core import get_position_range_mm as _range_mm

    return _range_mm(atlas, plane=plane)


def get_reference_slice(atlas, position_mm: float, *, plane: Plane):  # noqa: ANN001
    from langslice_harness.atlas.core import get_reference_slice as _reference_slice

    return _reference_slice(atlas, position_mm, plane=plane)

# 0.05 mm step matches the spec's atlas-grid granularity. Position keys are
# stored as int(round(position_mm / GRID_STEP_MM)) so float jitter from agent
# arguments cannot produce two cache entries for what is functionally one slice.
GRID_STEP_MM: float = 0.05


def _quantize(position_mm: float) -> int:
    return int(round(position_mm / GRID_STEP_MM))


def _grid_position_mm(key: int) -> float:
    return key * GRID_STEP_MM


@dataclass(frozen=True)
class _GridKey:
    atlas_name: str
    plane: Plane


@dataclass(frozen=True)
class GridRange:
    """Valid sliceable range for a single (atlas, plane) pair, in mm."""

    pos_lo: float
    pos_hi: float


class AtlasGrid:
    """In-memory cache of pre-rendered atlas reference slices.

    Each (atlas, plane) pair is rendered on a uniform ``GRID_STEP_MM`` grid
    spanning the valid range. ``get_slice`` snaps a requested position to the
    nearest grid point and returns the cached PIL image.
    """

    def __init__(
        self,
        slices: dict[_GridKey, dict[int, Image.Image]],
        ranges: dict[_GridKey, GridRange],
    ) -> None:
        self._slices = slices
        self._ranges = ranges

    @property
    def pairs(self) -> list[tuple[str, Plane]]:
        return [(k.atlas_name, k.plane) for k in self._slices]

    def range_mm(self, atlas_name: str, plane: Plane) -> GridRange:
        key = _GridKey(canonicalize_atlas_name(atlas_name), plane)
        if key not in self._ranges:
            raise KeyError(
                f"AtlasGrid has no entry for ({atlas_name!r}, {plane!r}); "
                f"known pairs: {self.pairs}"
            )
        return self._ranges[key]

    def get_slice(
        self, atlas_name: str, plane: Plane, position_mm: float
    ) -> tuple[Image.Image, float]:
        """Snap ``position_mm`` to the nearest grid point and return (image, snapped_mm)."""
        key = _GridKey(canonicalize_atlas_name(atlas_name), plane)
        try:
            grid = self._slices[key]
        except KeyError as exc:
            raise KeyError(
                f"AtlasGrid has no entry for ({atlas_name!r}, {plane!r}); "
                f"known pairs: {self.pairs}"
            ) from exc
        rng = self._ranges[key]
        clipped = max(rng.pos_lo, min(rng.pos_hi, float(position_mm)))
        qkey = _quantize(clipped)
        # The grid was built across [pos_lo, pos_hi]; clipping above plus the
        # round here guarantees qkey is in the dict.
        if qkey not in grid:
            qkey = min(grid.keys(), key=lambda k: abs(k - qkey))
        return grid[qkey], _grid_position_mm(qkey)


def build_atlas_grid(
    pairs: Iterable[tuple[str, Plane]],
    *,
    step_mm: float = GRID_STEP_MM,
) -> AtlasGrid:
    """Pre-render reference slices for every (atlas, plane) pair on a uniform grid.

    Pairs are deduplicated. Each unique atlas is loaded at most once. Slices
    are rendered grayscale via ``get_reference_slice`` — colored / boundary
    variants are not part of the agent's tool surface and intentionally omitted.
    """
    if step_mm <= 0:
        raise ValueError(f"step_mm must be positive, got {step_mm}")

    unique_pairs = sorted({(canonicalize_atlas_name(name), plane) for name, plane in pairs})
    slices: dict[_GridKey, dict[int, Image.Image]] = {}
    ranges: dict[_GridKey, GridRange] = {}

    for atlas_name, plane in unique_pairs:
        atlas = load_atlas(atlas_name)
        pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
        key = _GridKey(atlas_name, plane)
        ranges[key] = GridRange(pos_lo=pos_lo, pos_hi=pos_hi)

        # `_quantize` rounds to nearest 0.05 mm. If pos_hi is e.g. 13.225 then
        # ceil-rounding it gives 13.25, which is past the atlas's last valid
        # index — `get_reference_slice` rejects that with a ValueError. Clamp
        # both ends INWARD so every grid point we ask the atlas to render is
        # provably in-range. The same logic applies symmetrically at pos_lo
        # (rare in practice since pos_lo is 0, but cheap to handle).
        lo_q = _quantize(pos_lo)
        if _grid_position_mm(lo_q) < pos_lo:
            lo_q += 1
        hi_q = _quantize(pos_hi)
        if _grid_position_mm(hi_q) > pos_hi:
            hi_q -= 1
        if hi_q < lo_q:
            raise ValueError(
                f"Empty grid for ({atlas_name!r}, {plane!r}): "
                f"valid range [{pos_lo:.4f}, {pos_hi:.4f}] mm is narrower than "
                f"step {step_mm:.3f} mm"
            )

        per_pair: dict[int, Image.Image] = {}
        for qkey in range(lo_q, hi_q + 1):
            position_mm = _grid_position_mm(qkey)
            # Atlas resolution is ~25 um, much finer than 0.05 mm grid, so
            # snapping to grid does not drop slice fidelity.
            per_pair[qkey] = get_reference_slice(atlas, position_mm, plane=plane)
        slices[key] = per_pair

    return AtlasGrid(slices=slices, ranges=ranges)
