"""Placement contract and the section/working pixel grids a fit moves between.

Coordinates: a section pixel's centre is its integer index (OpenCV's
convention); physical millimetres on the section are ``index * mm_per_px``,
so pixel (0, 0)'s centre is at (0, 0) mm. The fit itself runs on a coarser
*working* grid over the same physical area, and its displacement fields are
in millimetres, which do not change between the two grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

from langslice.space import Plane

if TYPE_CHECKING:
    from langslice.core.handoff import LinearRegistrationInput


@dataclass(frozen=True)
class Placement:
    """A linear placement of an atlas plane on one section image.

    ``atlas_to_section`` maps pixel centres of the display-oriented native
    atlas plane (``langslice.atlas.render.annotation_slice`` at this
    position, plane and cutting angles; native image axes, no mirror) to
    pixel centres of the section image the fit is given. This is
    ``core.handoff``'s ``atlas_to_slice`` and the image tool's
    ``atlas_to_canvas``, unchanged.
    """

    atlas_name: str
    position_mm: float
    plane: Plane
    pitch_deg: float
    yaw_deg: float
    atlas_to_section: np.ndarray
    section_mm_per_px: float
    source: str = "supplied"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.atlas_to_section, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0.0, 0.0, 1.0])
                or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
            raise ValueError("atlas_to_section must be a finite invertible 3x3 affine")
        if not (np.isfinite(self.section_mm_per_px) and self.section_mm_per_px > 0):
            raise ValueError("section_mm_per_px must be positive and finite")
        if self.plane not in ("coronal", "sagittal", "horizontal"):
            raise ValueError(f"Unknown plane: {self.plane!r}")
        object.__setattr__(self, "atlas_to_section", matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_name": self.atlas_name, "position_mm": float(self.position_mm),
            "plane": self.plane, "pitch_deg": float(self.pitch_deg),
            "yaw_deg": float(self.yaw_deg),
            "atlas_to_section": self.atlas_to_section.tolist(),
            "section_mm_per_px": float(self.section_mm_per_px), "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Placement:
        return cls(
            atlas_name=str(value["atlas_name"]), position_mm=float(value["position_mm"]),
            plane=cast(Plane, value["plane"]), pitch_deg=float(value["pitch_deg"]),
            yaw_deg=float(value["yaw_deg"]),
            atlas_to_section=np.asarray(value["atlas_to_section"], dtype=np.float64),
            section_mm_per_px=float(value["section_mm_per_px"]),
            source=str(value.get("source", "supplied")),
        )


def placement_from_handoff(prepared: LinearRegistrationInput) -> Placement:
    """The placement a ``core.handoff.prepare_linear_registration`` result carries.

    The section image to fit is ``prepared.image`` (the oriented rendered
    section, not the acquisition TIFF).
    """
    return Placement(
        atlas_name=prepared.atlas_name, position_mm=prepared.position_mm,
        plane=prepared.plane, pitch_deg=prepared.pitch_deg, yaw_deg=prepared.yaw_deg,
        atlas_to_section=prepared.atlas_to_slice,
        section_mm_per_px=float(prepared.metadata["section_um_per_px"]) / 1000.0,
        source=str(prepared.metadata.get("source", "linear_state")),
    )


def pixel_center_map(source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    """3x3 map of pixel centres from one grid onto a resized grid over the same area."""
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return np.array([[sx, 0.0, (sx - 1.0) / 2.0],
                     [0.0, sy, (sy - 1.0) / 2.0],
                     [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class WorkingGrid:
    """The section's physical area resampled to the fit's working resolution.

    Per-axis spacing absorbs rounding of the working size, so the working grid
    covers exactly the section's extent. ``origin_mm`` is the physical position
    of working pixel (0, 0)'s centre in the section's millimetre frame.
    """

    section_size: tuple[int, int]
    section_mm_per_px: float
    size: tuple[int, int]

    @classmethod
    def for_section(
        cls, section_size: tuple[int, int], section_mm_per_px: float, working_mm: float,
    ) -> WorkingGrid:
        factor = min(1.0, section_mm_per_px / working_mm)
        width = max(8, int(round(section_size[0] * factor)))
        height = max(8, int(round(section_size[1] * factor)))
        return cls(section_size, float(section_mm_per_px), (width, height))

    @property
    def spacing_mm(self) -> tuple[float, float]:
        return (self.section_mm_per_px * self.section_size[0] / self.size[0],
                self.section_mm_per_px * self.section_size[1] / self.size[1])

    @property
    def origin_mm(self) -> tuple[float, float]:
        sx, sy = self.spacing_mm
        return (0.5 * sx - 0.5 * self.section_mm_per_px, 0.5 * sy - 0.5 * self.section_mm_per_px)

    @property
    def section_to_working(self) -> np.ndarray:
        return pixel_center_map(self.section_size, self.size)

    def to_working(self, image: np.ndarray, *, nearest: bool = False) -> np.ndarray:
        """Downsample a section-grid image (area average, or nearest for labels)."""
        if tuple(image.shape[1::-1]) == self.size:
            return image
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(image, self.size, interpolation=flags)

    def field_to_section(self, field_mm: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate a working-grid millimetre field onto the section grid."""
        if tuple(field_mm.shape[1::-1]) == self.section_size:
            return field_mm.astype(np.float32)
        return np.stack([
            cv2.resize(np.ascontiguousarray(field_mm[..., k], dtype=np.float32),
                       self.section_size, interpolation=cv2.INTER_LINEAR)
            for k in range(2)
        ], axis=-1)


def warp_affine(
    image: np.ndarray, matrix: np.ndarray, size: tuple[int, int], *, nearest: bool = False,
) -> np.ndarray:
    """Place a native atlas image with a 3x3 pixel-centre map (outside = 0).

    Nearest-neighbour warps run in float64 so large atlas ids survive exactly.
    """
    if nearest:
        return cv2.warpAffine(
            image.astype(np.float64), np.asarray(matrix)[:2], size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(image.dtype)
    return cv2.warpAffine(
        image.astype(np.float32), np.asarray(matrix)[:2], size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def sample_native(
    native: np.ndarray, coordinates: np.ndarray, *, nearest: bool = True,
) -> np.ndarray:
    """Sample a native atlas image at (H, W, 2) [x, y] native pixel coordinates.

    Outside the native grid gives 0 (background).
    """
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    height, width = native.shape
    if nearest:
        ix, iy = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        result = np.zeros(x.shape, dtype=native.dtype)
        result[inside] = native[iy[inside], ix[inside]]
        return result
    return cv2.remap(
        native.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
