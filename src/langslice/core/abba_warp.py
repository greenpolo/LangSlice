"""An applied deformation as ABBA landmark pairs, on top of LangSlice's affine step.

The Fiji connector lands a section's deformation as a SECOND ABBA
registration step (a BigWarp thin-plate spline, ``SacBigWarp2DRegistration``)
on top of the affine step :func:`langslice.core.abba_affine.normalized_to_abba_affine`
gives it. ABBA applies the affine first, so the warp step maps points
ALREADY PLACED by the affine. This module computes that step's landmark pairs.

The frame (derived from ``normalized_to_abba_affine``). The six stored
numbers ``M`` (``core.affine.denormalized_affine`` on the oriented working
render of size ``(w, h)``) take an oriented render pixel ``x`` of the
section to the render pixel ``M x`` where the linear placement draws the
atlas point that belongs there (``core.handoff.linear_placement_matrix``:
``atlas_to_render = M^-1 A``, ``A`` the atlas drawn on the render's frame).
The ABBA affine row is ``S M S^-1`` with ``S(x) = extent * (x / (w, h) - 1/2)``,
``extent`` the oriented snapshot's size in millimetres (the snapshot is
centred on ABBA's world origin; the quarter turn and flip are the
connector's orientation, ``R`` below, already inside the affine step). So
after the affine step, the section's file pixel ``f`` sits at

    before(f) = S M R f = S A n_lin(f),     n_lin = file_to_native (linear)

(``R`` = ``SectionFrame.file_to_render``), and the atlas point the
deformation gives it, ``n_w(f)`` (:func:`langslice.core.maps.native_points`:
the record's field on its fit grid, the working-grid resize and the
orientation handled there), sits at

    after(f) = S A n_w(f) = S M atlas_to_render n_w(f).

The warp step moves ``before(f)`` to ``after(f)``. Its landmark pairs are
returned as :func:`langslice.core.abba_spline.spline_world_landmarks`
returns a spline's: ``source`` = the point before (the section side of
the step), ``target`` = where it goes (the atlas side), centred ABBA world
millimetres. ABBA resamples with the pull-back ``target -> source``, a TPS
interpolating these pairs; :func:`warp_world_landmarks` checks that TPS,
not the pairs.

Accuracy and folds, as ``abba_spline._sample_elastix_landmarks`` checks an
Elastix map: the exact map (the record's bilinear field) must not fold
(sampled Jacobian of the residual > 0 on a 65x65 screen of the snapshot);
then a regular grid over the snapshot of 9x9, 17x17, 25x25 and at most
33x33 file points is tried, the first whose TPS pull-back reproduces the
exact one within *tolerance_mm* (5 um) at three independent off-grid probe
sets (a denser grid with the edges, and two offset grids) with no
non-positive sampled Jacobian is returned. Otherwise ``ValueError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.affine import denormalized_affine

if TYPE_CHECKING:
    from langslice.core.deformable.record import DeformableRecord
    from langslice.core.maps import SectionFrame

logger = logging.getLogger(__name__)

#: Grids tried, in file points per side (the last is ABBA's 1089-point cap).
GRID_SIDES = (9, 17, 25, 33)
#: Largest off-grid error accepted, in millimetres.
TOLERANCE_MM = 0.005
#: Points per side of the fold screen of the exact map.
SCREEN_SIDE = 65
#: Smallest Jacobian determinant accepted (a fold or collapse below it).
MIN_JACOBIAN = 1e-6


def world_frame(frame: SectionFrame, params: Any, *, size: tuple[int, int],
                pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """``(S @ M, S @ M @ atlas_to_render)``: 3x3 maps on ``[x, y, 1]`` from
    an oriented render pixel, and from a native atlas-plane pixel, to ABBA's
    centred world millimetres after LangSlice's affine step (see the module
    text). *size* and *pixel_size_um* are the snapshot's, as the affine row
    is built from them."""
    dimensions = np.asarray(size, dtype=np.float64)
    if dimensions.shape != (2,) or not np.all(np.isfinite(dimensions) & (dimensions > 0)):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if tuple(int(v) for v in size) != tuple(int(v) for v in frame.file_size):
        raise ValueError("The section frame was not built from this snapshot's pixels")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be positive and finite")
    if int(frame.rotation_deg) % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * float(pixel_size_um) / 1000.0
    width, height = (float(v) for v in frame.render_size)
    scale = np.array([[extent[0] / width, 0.0, -extent[0] / 2.0],
                      [0.0, extent[1] / height, -extent[1] / 2.0],
                      [0.0, 0.0, 1.0]])
    placed = scale @ np.vstack([denormalized_affine(params, frame.render_size),
                                [0.0, 0.0, 1.0]])
    return placed, placed @ np.asarray(frame.atlas_to_render, dtype=np.float64)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def warp_world_landmarks(
    frame: SectionFrame, warp: DeformableRecord, params: Any, *,
    size: tuple[int, int], pixel_size_um: float, tolerance_mm: float = TOLERANCE_MM,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(source, target)``, N-by-2 centred ABBA world millimetres: the
    warp step's landmark pairs on top of the affine step (module text).
    *params* are the section's six stored numbers (the ones *frame* was
    built from: :func:`langslice.core.maps.stored_params`). ``ValueError``
    when the deformation folds, or no grid up to 33x33 meets *tolerance_mm*.
    *diagnostics* receives the grid, point count, measured error and
    smallest sampled Jacobian."""
    from langslice.core.landmark_warp import _ThinPlateKernel
    from langslice.core.maps import native_points

    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Export tolerance must be positive and finite")
    render_to_world, native_to_world = world_frame(frame, params, size=size,
                                                   pixel_size_um=pixel_size_um)
    file_to_world = render_to_world @ np.asarray(frame.file_to_render, dtype=np.float64)
    width, height = (float(v) for v in frame.file_size)
    low, high = np.array([-0.5, -0.5]), np.array([width - 0.5, height - 0.5])

    def grid(n: int, offset: float = 0.0) -> np.ndarray:
        fractions = (np.linspace(0.0, 1.0, n) if offset == 0
                     else (np.arange(n - 1) + offset) / (n - 1))
        xs = low[0] + fractions * (high[0] - low[0])
        ys = low[1] + fractions * (high[1] - low[1])
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack((gx.ravel(), gy.ravel()))

    def warped_native(points: np.ndarray) -> np.ndarray:
        nx, ny = native_points(frame, warp, points[None, :, 0], points[None, :, 1])
        return np.column_stack((nx[0], ny[0]))

    def pairs(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _apply(file_to_world, points), _apply(native_to_world, warped_native(points))

    # The exact map must not fold: the residual's Jacobian in file pixels.
    fit_px = float(frame.file_size[0]) / float(max(1, int(warp.section_size[0])))
    epsilon = 0.25 * max(fit_px, 1e-3)
    screen = grid(SCREEN_SIDE)
    to_linear = np.linalg.inv(np.asarray(frame.file_to_native(), dtype=np.float64))
    columns = []
    for axis in np.eye(2) * epsilon:
        ahead = _apply(to_linear, warped_native(screen + axis))
        behind = _apply(to_linear, warped_native(screen - axis))
        columns.append((ahead - behind) / (2.0 * epsilon))
    exact = np.linalg.det(np.stack(columns, axis=2))
    if not np.isfinite(exact).all() or np.any(exact <= MIN_JACOBIAN):
        raise ValueError("The deformation folds or collapses on the sampled snapshot")

    last_error = float("inf")
    for n in GRID_SIDES:
        source, target = pairs(grid(n))
        try:
            pullback = _ThinPlateKernel(target, source, max_points=GRID_SIDES[-1] ** 2)
        except ValueError:
            continue
        probes = np.vstack((grid(2 * n - 1), grid(n, 0.37), grid(n, 0.71)))
        probe_source, probe_target = pairs(probes)
        error = np.linalg.norm(pullback.forward(probe_target) - probe_source, axis=1)
        last_error = float(np.max(error))
        determinant = np.linalg.det(pullback.jacobian(probe_target))
        if not np.isfinite(determinant).all() or np.any(determinant <= MIN_JACOBIAN):
            continue
        if last_error <= tolerance_mm:
            report = {"grid_size": n, "points": len(source), "max_error_mm": last_error,
                      "tolerance_mm": float(tolerance_mm),
                      "min_sampled_jacobian": float(determinant.min()),
                      "min_exact_jacobian": float(exact.min()),
                      "validation_points": len(probes)}
            if diagnostics is not None:
                diagnostics.update(report)
            logger.info("ABBA warp approximation of %s: %s", frame.section_id, report)
            return source, target
    raise ValueError(
        "The deformation could not be expressed as ABBA's thin-plate spline within "
        f"{tolerance_mm * 1000:g} um using at most {GRID_SIDES[-1] ** 2} points (last "
        f"maximum error {last_error * 1000:.3g} um)")
