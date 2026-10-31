"""Exact conversion of normalized snapshot affines to ABBA's centred XY frame.

An applied deformation as ABBA landmark pairs, on top of LangSlice's affine
step.

The Fiji connector lands a section's deformation as a SECOND ABBA registration
step (a BigWarp thin-plate spline, ``BigWarpSource2DRegistration``) on top of
the affine step :func:`langslice.core.maps.abba.normalized_to_abba_affine`
gives it. ABBA applies the affine first, so the warp step maps points ALREADY
PLACED by the affine. This module computes that step's landmark pairs.

The frame (derived from ``normalized_to_abba_affine``). The six stored numbers
``M`` (``core.geometry.affine.denormalized_affine`` on the oriented working
render of size ``(w, h)``) take an oriented render pixel ``x`` of the section
to the render pixel ``M x`` where the linear placement draws the atlas point
that belongs there (``core.fits.frame.linear_placement_matrix``:
``atlas_to_render = M^-1 A``, ``A`` the atlas drawn on the render's frame). The
ABBA affine row is ``S M S^-1`` with ``S(x) = extent * (x / (w, h) - 1/2)``,
``extent`` the oriented snapshot's size in millimetres (the snapshot is centred
on ABBA's world origin; the quarter turn and flip are the connector's
orientation, ``R`` below, already inside the affine step). So after the affine
step, the section's file pixel ``f`` sits at

    before(f) = S M R f = S A n_lin(f),     n_lin = file_to_native (linear)

(``R`` = ``SectionFrame.file_to_render``), and the atlas point the deformation
gives it, ``n_w(f)`` (:func:`langslice.core.maps.section_maps.native_points`:
the record's field on its fit grid, the working-grid resize and the orientation
handled there), sits at

    after(f) = S A n_w(f) = S M atlas_to_render n_w(f).

The warp step moves ``before(f)`` to ``after(f)``. Its landmark pairs are
returned as ``source`` = the point before (the section side of
the step) and ``target`` = where it goes (the atlas side), centred ABBA
world millimetres. ABBA resamples with the pull-back ``target -> source``,
a TPS interpolating these pairs; :func:`warp_world_landmarks` checks that
TPS, not the pairs.

Where the pairs go. The region that matters is the atlas overlap: the
section points inside the record's footprint (the placed atlas brain the
fit worked on, widened by its margin) or that the deformation carries into
the atlas brain, where ABBA draws atlas borders over the section. A job
holds no tissue mask of its own, so the overlap stands for the tissue; a
record with no overlap (none on the snapshot) uses the whole snapshot. At
most :data:`MAX_POINTS` pairs (1089: ABBA's cost grows fast with the point
count; in ABBA 0.24.1 building the step takes ~3 s at 1089 points against
~5 min at 4225) are placed by :func:`langslice.core.maps.tps_landmarks.refine`:
an even grid of about :data:`START_POINTS` over the overlap's bounding box and
a coarse :data:`FRAME_SIDE` x :data:`FRAME_SIDE` grid over the rest of the
snapshot, then pairs added where the spline's estimated atlas-side error
(its section-side error times the deformation's local stretch) is largest
at a search grid of :data:`SEARCH_SIDE` points along the snapshot's long
side, until the budget
or the tolerance (5 um at the search grid) is reached. A round whose new
pairs fold the spline anywhere on the snapshot (tested on the search grid
and on a finer lattice around each new pair) drops them near the fold.
When that start already folds a smaller start is tried, then the even
snapshot grids of 33x33 down to 9x9 points; only when every one folds is
``ValueError`` raised.

Accuracy. The sent pairs' spline is measured against the exact map at
held-out probes inside the overlap (a lattice of twice the search grid's
step, halfway between its points, never used to choose pairs), on both
sides: on the atlas side, how far ABBA draws a section point from where
LangSlice draws it (ABBA's inverse of the pull-back, by Newton's method),
and on the section side. The connector's warp row carries the atlas side's
median, 95th and 99th percentiles and maximum. The exact map's own
smallest sampled Jacobian (a 65x65 screen) is reported, not enforced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from langslice.core.fits.deformable.geometry import sample_native
from langslice.core.geometry.affine import denormalized_affine
from langslice.core.maps.section_maps import (
    file_to_fit,
    native_points,
    point_rows,
)
from langslice.core.maps.tps_landmarks import (
    Accuracy,
    BoolArray,
    FloatArray,
    Pairs,
    Placed,
    Samples,
    Spacing,
    accuracy,
    determinants,
    folded_at,
    pairs_at,
    pullback,
    refine,
)

if TYPE_CHECKING:
    from langslice.core.fits.deformable.record import DeformableRecord
    from langslice.core.maps.section_maps import SectionFrame

logger = logging.getLogger(__name__)

#: ABBA's point budget: its cost grows fast with the point count.
MAX_POINTS = 33 * 33
#: Pairs in the even start grid over the overlap's bounding box, about a
#: quarter of the budget so most pairs go where the spline errs.
START_POINTS = 17 * 17
#: The smaller start tried when that one folds.
SMALL_START_POINTS = 9 * 9
#: The start box's margin around the overlap, as a fraction of its size.
START_MARGIN = 0.05
#: Points per side of the coarse grid over the rest of the snapshot.
FRAME_SIDE = 9
#: Search points along the snapshot's long side: about 25-50 um apart on
#: a whole mouse section, finer than the warps' smoothing.
SEARCH_SIDE = 320
#: Smallest distance between a new pair and a placed one, in search steps.
MIN_GAP_STEPS = 1.5
#: Smallest distance between two pairs added in one round, in search steps.
BATCH_GAP_STEPS = 8.0
#: The even snapshot grids tried when every adaptive start folds, in points
#: per side.
GRID_SIDES = (33, 25, 17, 9)
#: Search error at which pairs stop being added, in millimetres.
TOLERANCE_MM = 0.005
#: Points per side of the screen of the exact map's Jacobian (reported).
SCREEN_SIDE = 65
#: Smallest fraction of the snapshot the overlap may cover before the whole
#: snapshot is used instead.
MIN_REGION_FRACTION = 0.01


def normalized_to_abba_affine(
    params: Any,
    *,
    size: tuple[int, int],
    pixel_size_um: float,
    rotation_deg: int = 0,
) -> np.ndarray:
    """Return the 3×4 world-mm transform of a centred, calibrated snapshot.

    ``size`` is the original exported image's full width and height; the
    quarter-turn swaps its axes before the normalized affine is applied.
    ``pixel_size_um`` is the export calibration, not the resized preview's.

    Stored six-number transforms describe the oriented SECTION frame. They
    already contain pivot displacement and shear, including when parameters
    were chosen on a larger padded display canvas. Recovering them from the
    rounded physical knobs instead would lose both.

    ABBA's XY axes and the snapshot pixel axes increase in the same direction.
    Conjugating the normalized transform by ``world = extent * (fraction-.5)``
    therefore carries the full affine over directly, with no extra sign flip.
    The slicing axis remains unchanged.
    """
    normalized = np.asarray(params, dtype=np.float64)
    if normalized.size != 6 or not np.all(np.isfinite(normalized)):
        raise ValueError(
            "The snapshot affine must contain six finite parameters"
        )
    normalized = normalized.reshape(2, 3)
    dimensions = np.asarray(size, dtype=np.float64)
    if dimensions.shape != (2,) or not np.all(
        np.isfinite(dimensions) & (dimensions > 0)
    ):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be positive and finite")
    if rotation_deg % 90:
        raise ValueError(
            "Snapshot orientation must be a multiple of 90 degrees"
        )
    if rotation_deg % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * float(pixel_size_um) / 1000.0
    linear = normalized[:, :2] * extent[:, None] / extent[None, :]
    translation = (
        extent * normalized[:, 2] + linear @ (extent / 2.0) - extent / 2.0
    )
    result = np.eye(3, 4, dtype=np.float64)
    result[:2, :2] = linear
    result[:2, 3] = translation
    return result


def world_frame(
    frame: SectionFrame,
    params: tuple[float, ...],
    *,
    size: tuple[int, int],
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(S @ M, S @ M @ atlas_to_render)``: 3x3 maps on ``[x, y, 1]`` from
    an oriented render pixel, and from a native atlas-plane pixel, to ABBA's
    centred world millimetres after LangSlice's affine step (see the module
    text). *size* and *pixel_size_um* are the snapshot's, as the affine row
    is built from them."""
    dimensions = np.asarray(size, dtype=np.float64)
    if dimensions.shape != (2,) or not np.all(
        np.isfinite(dimensions) & (dimensions > 0)
    ):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if tuple(int(v) for v in size) != tuple(int(v) for v in frame.file_size):
        raise ValueError(
            "The section frame was not built from this snapshot's pixels"
        )
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be positive and finite")
    if int(frame.rotation_deg) % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * float(pixel_size_um) / 1000.0
    width, height = (float(v) for v in frame.render_size)
    scale = np.array(
        [
            [extent[0] / width, 0.0, -extent[0] / 2.0],
            [0.0, extent[1] / height, -extent[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    placed = scale @ np.vstack(
        [denormalized_affine(params, frame.render_size), [0.0, 0.0, 1.0]]
    )
    return placed, placed @ np.asarray(frame.atlas_to_render, dtype=np.float64)


@dataclass(frozen=True)
class WarpLandmarks:
    """Hold a deformation's landmark pairs for ABBA and their accuracy.

    Parameters
    ----------
    source_mm
        N-by-2 centred ABBA world millimetres: section points as the affine
        step places them.
    target_mm
        Where the deformation takes them.
    placement
        ``adaptive`` (the greedy placement) or ``grid`` (an even snapshot
        grid, the fallback).
    region
        Where the accuracy is measured and pairs are sought: ``atlas
        overlap`` or ``whole snapshot``.
    accuracy
        The spline's measured difference from the deformation.
    tolerance_mm
        The tolerance the placement grew towards.
    min_exact_jacobian
        The deformation's own smallest sampled Jacobian (reported).
    rounds
        Greedy rounds run.
    dropped
        Pairs dropped because they folded the spline.
    """

    source_mm: FloatArray
    target_mm: FloatArray
    placement: str
    region: str
    accuracy: Accuracy
    tolerance_mm: float
    min_exact_jacobian: float
    rounds: int
    dropped: int

    @property
    def within_tolerance(self) -> bool:
        """Tell whether the atlas-side maximum is within the tolerance."""
        return self.accuracy.atlas.max_mm <= self.tolerance_mm


@dataclass(frozen=True)
class _WorldMap:
    """The deformation as world-millimetre pairs of snapshot file points."""

    frame: SectionFrame
    record: DeformableRecord
    file_to_world: FloatArray
    native_to_world: FloatArray

    def pairs(self, points_px: FloatArray) -> tuple[FloatArray, FloatArray]:
        native = self.warped_native(points_px)
        before = _apply(self.file_to_world, points_px)
        return before, _apply(self.native_to_world, native)

    def warped_native(self, points_px: FloatArray) -> FloatArray:
        count = len(points_px)
        nx, ny = native_points(
            self.frame,
            self.record,
            point_rows(points_px[:, 0]),
            point_rows(points_px[:, 1]),
        )
        return np.column_stack((nx.ravel()[:count], ny.ravel()[:count]))

    def stretch(self, points_px: FloatArray) -> FloatArray:
        """Differentiate the deformation's target by its source.

        Central differences a quarter of a fit-grid pixel apart, within
        one cell of the record's bilinear field.
        """
        epsilon = _half_cell_px(self.frame, self.record)
        source_columns: list[FloatArray] = []
        target_columns: list[FloatArray] = []
        for axis in _steps(epsilon):
            source_ahead, target_ahead = self.pairs(points_px + axis)
            source_behind, target_behind = self.pairs(points_px - axis)
            source_columns.append(source_ahead - source_behind)
            target_columns.append(target_ahead - target_behind)
        per_source = np.linalg.inv(np.stack(source_columns, axis=2))
        return np.stack(target_columns, axis=2) @ per_source

    def in_overlap(self, points_px: FloatArray) -> BoolArray:
        """Mark the points the atlas overlaps.

        A point is in the overlap when it lies in the record's footprint
        or the deformation carries it into the atlas brain.
        """
        footprint = np.asarray(self.record.footprint, dtype=bool)
        fit_size = (footprint.shape[1], footprint.shape[0])
        on_fit = _apply(file_to_fit(self.frame, fit_size), points_px)
        pixel = np.rint(on_fit).astype(np.int64)
        column, row = pixel[:, 0], pixel[:, 1]
        inside = cast(
            "BoolArray",
            (column >= 0) & (column < fit_size[0]) & (row >= 0),
        )
        inside &= cast("BoolArray", row < fit_size[1])
        on_footprint = np.zeros(len(points_px), dtype=bool)
        on_footprint[inside] = footprint[row[inside], column[inside]]

        native = self.warped_native(points_px)
        labels = cast(
            "NDArray[np.uint32]",
            sample_native(self.frame.annotation, native[None]),
        )
        in_brain = cast("BoolArray", labels.reshape(-1) != 0)
        return on_footprint | in_brain


def warp_world_landmarks(
    frame: SectionFrame,
    record: DeformableRecord,
    params: tuple[float, ...],
    *,
    size: tuple[int, int],
    pixel_size_um: float,
    tolerance_mm: float = TOLERANCE_MM,
) -> WarpLandmarks:
    """Place the warp step's landmark pairs on top of the affine step.

    Parameters
    ----------
    frame
        The section's frame, built from the snapshot's pixels.
    record
        The applied deformation.
    params
        The section's six stored numbers (the ones *frame* was built from).
    size
        The snapshot's width and height in pixels.
    pixel_size_um
        The snapshot's calibration.
    tolerance_mm
        The search error at which pairs stop being added.

    Returns
    -------
    WarpLandmarks
        The pairs (module text) and their measured accuracy.

    Raises
    ------
    ValueError
        When the tolerance is not positive, the snapshot is not the
        frame's, or the spline folds at every start and every grid.
    """
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Export tolerance must be positive and finite")
    render_to_world, native_to_world = world_frame(
        frame, params, size=size, pixel_size_um=pixel_size_um
    )
    file_to_world = render_to_world @ np.asarray(
        frame.file_to_render, dtype=np.float64
    )
    exact = _WorldMap(frame, record, file_to_world, native_to_world)

    step_px = max(frame.file_size) / SEARCH_SIDE
    search, region = _samples(exact, step_px, step_px)
    # held out: every other search step, halfway between search points
    probes, _ = _samples(exact, 2.0 * step_px, step_px, region=region)
    spacing = Spacing(
        min_gap_px=MIN_GAP_STEPS * step_px,
        batch_gap_px=BATCH_GAP_STEPS * step_px,
        step_px=step_px,
    )

    adaptive = _adaptive(exact, search, spacing, tolerance_mm)
    if adaptive is not None:
        pairs, placement = adaptive.pairs, "adaptive"
        rounds, dropped = adaptive.rounds, adaptive.dropped
    else:
        pairs, placement = _even_grid(exact, search), "grid"
        rounds = dropped = 0

    kernel = pullback(pairs.source_mm, pairs.target_mm)
    landmarks = WarpLandmarks(
        source_mm=pairs.source_mm,
        target_mm=pairs.target_mm,
        placement=placement,
        region=region,
        accuracy=accuracy(kernel, probes),
        tolerance_mm=float(tolerance_mm),
        min_exact_jacobian=_exact_jacobian(exact),
        rounds=rounds,
        dropped=dropped,
    )

    atlas = landmarks.accuracy.atlas
    logger.info(
        "ABBA warp of %s: %d pairs (%s, %d rounds, %d dropped for folds); "
        "over the %s the spline is %.1f um from the deformation at the "
        "median, %.1f um at 99%% and %.1f um at most (%d held-out points)",
        frame.section_id,
        len(pairs.source_mm),
        placement,
        rounds,
        dropped,
        region,
        atlas.median_mm * 1000,
        atlas.p99_mm * 1000,
        atlas.max_mm * 1000,
        landmarks.accuracy.probes,
    )
    return landmarks


def _apply(matrix: FloatArray, points: FloatArray) -> FloatArray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _lattice(
    size: tuple[int, int], step_px: float, first_px: float
) -> FloatArray:
    """Lay a square lattice of file points over the snapshot.

    The points are *step_px* apart, the first *first_px* inside the edge.
    """
    xs = np.arange(-0.5 + first_px, size[0] - 0.5, step_px)
    ys = np.arange(-0.5 + first_px, size[1] - 0.5, step_px)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)


def _samples(
    exact: _WorldMap,
    step_px: float,
    first_px: float,
    *,
    region: str | None = None,
) -> tuple[Samples, str]:
    """Sample the deformation on a lattice and mark the region.

    The region is *region*, or else the atlas overlap unless it is too
    small; its name is returned with the samples.
    """
    points = _lattice(exact.frame.file_size, step_px, first_px)
    source, target = exact.pairs(points)
    overlap = exact.in_overlap(points)
    if region is None:
        enough = float(np.mean(overlap)) >= MIN_REGION_FRACTION
        region = "atlas overlap" if enough else "whole snapshot"
    inside = overlap
    if region != "atlas overlap":
        inside = np.ones(len(points), dtype=bool)
    stretch = exact.stretch(points)
    return Samples(points, source, target, inside, stretch), region


def _adaptive(
    exact: _WorldMap,
    search: Samples,
    spacing: Spacing,
    tolerance_mm: float,
) -> Placed | None:
    """Place pairs greedily from the first start that does not fold.

    None when every start folds.
    """
    for count in (START_POINTS, SMALL_START_POINTS):
        start = _start(exact.frame.file_size, search, count)
        adaptive = refine(
            pairs_at(exact, start),
            exact,
            search,
            spacing,
            budget=MAX_POINTS,
            tolerance_mm=tolerance_mm,
        )
        if adaptive is not None:
            return adaptive
    return None


def _start(size: tuple[int, int], search: Samples, count: int) -> FloatArray:
    """Lay the start's file points.

    About *count* points evenly spaced over the region's bounding box, and
    a coarse snapshot grid outside it.
    """
    inside = search.points_px[search.in_region]
    x_low, x_high = _padded(inside[:, 0], size[0])
    y_low, y_high = _padded(inside[:, 1], size[1])

    width, height = x_high - x_low, y_high - y_low
    columns = max(2, round(math.sqrt(count * width / height)))
    rows = max(2, round(count / columns))
    box = _even((x_low, x_high), (y_low, y_high), columns, rows)

    spacing = max(width / (columns - 1), height / (rows - 1))
    frame = _even(*_snapshot(size), FRAME_SIDE, FRAME_SIDE)
    outside_x = np.maximum(x_low - frame[:, 0], frame[:, 0] - x_high)
    outside_y = np.maximum(y_low - frame[:, 1], frame[:, 1] - y_high)
    away = np.maximum(outside_x, outside_y) > spacing
    return np.vstack((box, frame[away]))


def _padded(values: FloatArray, length: int) -> tuple[float, float]:
    """Find the span of *values* widened by the start margin.

    The span is kept within the snapshot's *length* pixels.
    """
    low, high = float(np.min(values)), float(np.max(values))
    margin = START_MARGIN * (high - low)
    return max(low - margin, -0.5), min(high + margin, length - 0.5)


def _snapshot(
    size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Give the snapshot's span in x and in y, pixel edge to pixel edge."""
    return (-0.5, size[0] - 0.5), (-0.5, size[1] - 0.5)


def _even(
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    columns: int,
    rows: int,
) -> FloatArray:
    """Lay an even grid of *columns* by *rows* over two spans."""
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_span[0], x_span[1], columns),
        np.linspace(y_span[0], y_span[1], rows),
    )
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)


def _even_grid(
    exact: _WorldMap,
    search: Samples,
) -> Pairs:
    """Find the largest even snapshot grid whose spline does not fold."""
    snapshot = _snapshot(exact.frame.file_size)
    for side in GRID_SIDES:
        grid = pairs_at(exact, _even(*snapshot, side, side))
        try:
            kernel = pullback(grid.source_mm, grid.target_mm)
        except ValueError:
            continue
        if not np.any(folded_at(kernel, search.target_mm)):
            return grid
    raise ValueError(
        "the deformation's thin-plate spline folds at every start and "
        f"every grid up to {GRID_SIDES[0]}x{GRID_SIDES[0]} points"
    )


def _exact_jacobian(exact: _WorldMap) -> float:
    """Find the deformation's smallest Jacobian on a snapshot screen.

    The Jacobian is the residual's, in file pixels.
    """
    frame = exact.frame
    screen = _even(*_snapshot(frame.file_size), SCREEN_SIDE, SCREEN_SIDE)
    epsilon = _half_cell_px(frame, exact.record)
    file_to_native = np.asarray(frame.file_to_native(), dtype=np.float64)
    to_linear = np.linalg.inv(file_to_native).astype(np.float64)

    columns: list[FloatArray] = []
    for axis in _steps(epsilon):
        ahead = _apply(to_linear, exact.warped_native(screen + axis))
        behind = _apply(to_linear, exact.warped_native(screen - axis))
        columns.append((ahead - behind) / (2.0 * epsilon))
    determinant = determinants(np.stack(columns, axis=2))
    if not np.isfinite(determinant).all():
        return float("nan")
    return float(np.min(determinant))


def _half_cell_px(frame: SectionFrame, record: DeformableRecord) -> float:
    """Give a finite-difference step: a quarter of a fit-grid pixel.

    The step is in file pixels, so both points of a central difference lie
    in one cell of the record's bilinear field.
    """
    fit_px = frame.file_size[0] / max(1, record.section_size[0])
    return 0.25 * max(float(fit_px), 1e-3)


def _steps(epsilon: float) -> tuple[FloatArray, FloatArray]:
    """Give the steps of *epsilon* along x and along y."""
    return np.array([epsilon, 0.0]), np.array([0.0, epsilon])
