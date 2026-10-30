"""A record's fields from the step that made them: onto the section, composed.

An engine fits on a working grid coarser than the section; a record's
fields are on the section grid and, for a step fitted on an earlier
record, composed with that record's. This module does both, once: a fit
makes its record's fields with it, and a saved record is rebuilt with it
on loading, so the two are the same arrays bit for bit. What a record
keeps of its step (its ``stored_step``) is one of:

==============  ========================================================
``EngineStep``  the engine's forward and inverse fields on its working
                grid, float32
``MeshStep``    a landmark application's mesh vertices and triangles
None            nothing: the record's section-grid fields are stored
                whole (a map made elsewhere, a fit on the section's own
                grid, and a restricted fit, whose tapered step's inverse
                is a fixed point that takes seconds to compute)
==============  ========================================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

# scipy ships no type stubs
from scipy import ndimage as ndi  # pyright: ignore[reportMissingTypeStubs]

from langslice.core.fits.deformable.geometry import WorkingGrid
from langslice.core.geometry.triangulation import PiecewiseAffine

#: scipy's bilinear sampler, typed here: scipy ships no type stubs.
_map_coordinates = cast(
    "Callable[..., np.ndarray]",
    ndi.map_coordinates,  # pyright: ignore[reportUnknownMemberType]
)


@dataclass(frozen=True)
class EngineStep:
    """An engine's step as it produced it, on its working grid.

    Parameters
    ----------
    grid
        The working grid the engine fitted on, over the section grid.
    field_mm
        Section to placed atlas, ``(rows, cols, 2)`` float32 ``(dx, dy)``
        in millimetres on ``grid``.
    inverse_field_mm
        Placed atlas to section, the same shape, units and grid.
    """

    grid: WorkingGrid
    field_mm: np.ndarray
    inverse_field_mm: np.ndarray


@dataclass(frozen=True)
class MeshStep:
    """A landmark application's step: its triangle mesh, in section pixels.

    Parameters
    ----------
    source_xy
        The mesh's vertices on the section, ``(n, 2)`` pixel ``(x, y)``,
        the four fixed corners first.
    target_xy
        Where each vertex goes on the placed atlas, the same shape.
    simplices
        The mesh's triangles, ``(m, 3)`` vertex indices, as its Delaunay
        triangulation made them.
    """

    source_xy: np.ndarray
    target_xy: np.ndarray
    simplices: np.ndarray


#: What a record's fields are rebuilt from; None: they are stored whole.
StoredStep = EngineStep | MeshStep


def carry_to_section(step: EngineStep) -> tuple[np.ndarray, np.ndarray]:
    """Carry an engine's step onto the section grid, bilinearly.

    Parameters
    ----------
    step
        The engine's fields on its working grid.

    Returns
    -------
    field_mm : numpy.ndarray
        The step's forward field on the section grid, ``(height, width,
        2)`` float32.
    inverse_field_mm : numpy.ndarray
        The step's inverse field on the section grid.
    """
    field_mm = step.grid.field_to_section(step.field_mm)
    inverse_mm = step.grid.field_to_section(step.inverse_field_mm)
    return field_mm, inverse_mm


def mesh_fields(
    step: MeshStep, section_size: tuple[int, int], mm_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a landmark mesh's fields at every section pixel centre.

    Parameters
    ----------
    step
        The mesh's vertices on the section and on the placed atlas.
    section_size
        The section grid's ``(width, height)`` in pixels.
    mm_per_px
        The section grid's millimetres per pixel.

    Returns
    -------
    field_mm : numpy.ndarray
        Section to placed atlas, ``(height, width, 2)`` float64 ``(dx,
        dy)`` in millimetres; NaN outside the mesh.
    inverse_field_mm : numpy.ndarray
        Placed atlas to section, the same shape and units.

    """
    # The mesh's own Delaunay maps forward by a vectorised lookup; given
    # triangles, by a loop over them, ten times slower. Its own triangles
    # are the stored ones wherever scipy triangulates as it did then.
    mesh = PiecewiseAffine(step.source_xy, step.target_xy)
    if not np.array_equal(mesh.simplices, step.simplices):
        mesh = PiecewiseAffine(step.source_xy, step.target_xy, step.simplices)
    xx, yy = _pixel_centres(section_size)
    points = np.stack([xx, yy], -1)
    forward = mesh.map(points)
    inverse = mesh.inverse(points)
    return (forward - points) * mm_per_px, (inverse - points) * mm_per_px


def sample_field(
    field_mm: np.ndarray, coordinates_px: np.ndarray
) -> np.ndarray:
    """Sample a field bilinearly at any pixel positions of its own grid.

    Parameters
    ----------
    field_mm
        A ``(rows, cols, 2)`` field.
    coordinates_px
        Where to sample it, ``(..., 2)`` pixel ``(x, y)``; beyond the edge
        the nearest edge value.

    Returns
    -------
    numpy.ndarray
        The field there, ``(..., 2)`` float32.
    """
    rows, cols = coordinates_px[..., 1], coordinates_px[..., 0]
    channels = [
        _map_coordinates(
            field_mm[..., k], [rows, cols], order=1, mode="nearest"
        )
        for k in range(2)
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def compose_forward(
    step_mm: np.ndarray, previous_mm: np.ndarray, mm_per_px: float
) -> np.ndarray:
    """Compose a step's forward field onto the record it was fitted on.

    Section point p goes to p + w(p) in the previous frame, then on by the
    previous field there.

    Parameters
    ----------
    step_mm
        The step's forward field on the section grid.
    previous_mm
        The previous record's forward field, the same grid.
    mm_per_px
        The section grid's millimetres per pixel.

    Returns
    -------
    numpy.ndarray
        The total forward field, ``(height, width, 2)``.
    """
    # numpy types a shape loosely: a field is (height, width, 2)
    height, width, _ = cast("tuple[int, int, int]", step_mm.shape)
    xx, yy = _pixel_centres((width, height))
    moved_x = xx + step_mm[..., 0] / mm_per_px
    moved_y = yy + step_mm[..., 1] / mm_per_px
    moved = np.stack([moved_x, moved_y], axis=-1)
    return step_mm + sample_field(previous_mm, moved)


def compose_inverse(
    step_inverse_mm: np.ndarray,
    previous_inverse_mm: np.ndarray,
    mm_per_px: float,
) -> np.ndarray:
    """Compose a step's inverse field after the previous record's inverse.

    Placed point q goes to q + v_prev(q), then on by the step's inverse
    there.

    Parameters
    ----------
    step_inverse_mm
        The step's inverse field on the section grid.
    previous_inverse_mm
        The previous record's inverse field, the same grid.
    mm_per_px
        The section grid's millimetres per pixel.

    Returns
    -------
    numpy.ndarray
        The total inverse field, ``(height, width, 2)``.
    """
    # numpy types a shape loosely: a field is (height, width, 2)
    height, width, _ = cast("tuple[int, int, int]", step_inverse_mm.shape)
    xx, yy = _pixel_centres((width, height))
    back_x = xx + previous_inverse_mm[..., 0] / mm_per_px
    back_y = yy + previous_inverse_mm[..., 1] / mm_per_px
    back = np.stack([back_x, back_y], axis=-1)
    return previous_inverse_mm + sample_field(step_inverse_mm, back)


def rebuilt_fields(
    step: StoredStep,
    section_size: tuple[int, int],
    mm_per_px: float,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a saved record's fields from its step, as its fit made them.

    Parameters
    ----------
    step
        What the record keeps of its step.
    section_size
        The section grid's ``(width, height)`` in pixels.
    mm_per_px
        The section grid's millimetres per pixel.
    previous
        The forward and inverse fields of the record the step was fitted
        on, as stored; None for a step on the linear placement.

    Returns
    -------
    field_mm : numpy.ndarray
        The record's forward field on the section grid, before rounding
        to storage precision.
    inverse_field_mm : numpy.ndarray
        Its inverse field, likewise.
    """
    if isinstance(step, MeshStep):
        forward_mm, inverse_mm = mesh_fields(step, section_size, mm_per_px)
        step_mm = forward_mm.astype(np.float32)
        step_inverse_mm = inverse_mm.astype(np.float32)
    else:
        step_mm, step_inverse_mm = carry_to_section(step)
    if previous is None:
        return step_mm, step_inverse_mm
    field_mm = compose_forward(step_mm, previous[0], mm_per_px)
    inverse_mm = compose_inverse(step_inverse_mm, previous[1], mm_per_px)
    return field_mm, inverse_mm


def _pixel_centres(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``x`` and ``y`` of every pixel centre of a grid, float64."""
    width, height = size
    pixels = np.indices((height, width), dtype=np.float64)
    return pixels[1, ...], pixels[0, ...]
