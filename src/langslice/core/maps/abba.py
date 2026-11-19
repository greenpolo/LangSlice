"""An applied deformation as ABBA landmark pairs in ABBA's world frame.

The file export (``tools/bookkeeping/abba_export.py``) writes a section's
deformation as a SECOND ABBA registration step (a BigWarp thin-plate spline,
``BigWarpSource2DRegistration``) on top of the affine step that
:func:`langslice.core.maps.abba_world.abba_placement` gives it. ABBA applies
the affine first, so the warp step maps points ALREADY PLACED by the affine.
This module computes that step's landmark pairs.

The caller passes two 3x3 maps on ``[x, y, 1]``: ``file_to_world`` takes a
section's file pixel to ABBA's centred world millimetres after the affine
step (the section side of a pair), and ``native_to_world`` takes a native
atlas-plane pixel to the same millimetres. The atlas point the deformation
gives a file pixel ``f`` is ``n_w(f)``
(:func:`langslice.core.maps.section_maps.native_points`: the record's field
on its fit grid, the working-grid resize and the orientation handled there).
The warp step moves ``file_to_world(f)`` to ``native_to_world(n_w(f))``. Its
landmark pairs are returned as ``source`` = the point before (the section
side of the step) and ``target`` = where it goes (the atlas side), centred
ABBA world millimetres. ABBA resamples with the pull-back ``target ->
source``, a TPS interpolating these pairs.

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
and on the section side. The exact map's own smallest sampled Jacobian (a
65x65 screen) is reported, not enforced.

``MIN_POINTS`` is the fewest pairs a warp takes and ``MIN_GRID_SIDE`` the
even grid of that smallest budget.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from langslice.core.fits.deformable.geometry import sample_native
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

MAX_POINTS = 33 * 33
START_POINTS = 17 * 17
SMALL_START_POINTS = 9 * 9
START_MARGIN = 0.05
FRAME_SIDE = 9
SEARCH_SIDE = 320
MIN_GAP_STEPS = 1.5
BATCH_GAP_STEPS = 8.0
GRID_SIDES = (33, 25, 17, 9)
MIN_POINTS = 16
MIN_GRID_SIDE = 4
TOLERANCE_MM = 0.005
SCREEN_SIDE = 65
MIN_REGION_FRACTION = 0.01


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


@dataclass(frozen=True)
class Budget:
    """How many pairs a placement may use, and its starts and fallbacks.

    The default is :data:`MAX_POINTS`; a smaller budget (a warp a person
    will edit in BigWarp) scales the start grids by the same fraction and
    keeps only the even fallback grids that fit.
    """

    max_points: int
    start_points: int
    small_start_points: int
    frame_side: int
    grid_sides: tuple[int, ...]


def budget(*, max_points: int = MAX_POINTS) -> Budget:
    """Size a placement's starts for at most *max_points* pairs.

    Parameters
    ----------
    max_points
        The most pairs, at least :data:`MIN_POINTS`.

    Returns
    -------
    Budget
        :data:`MAX_POINTS`'s starts scaled by ``max_points / MAX_POINTS``.

    Raises
    ------
    ValueError
        For fewer than :data:`MIN_POINTS` or more than :data:`MAX_POINTS`.
    """
    if not MIN_POINTS <= max_points <= MAX_POINTS:
        raise ValueError(f"A warp takes {MIN_POINTS} to {MAX_POINTS} landmark pairs")
    share = max_points / MAX_POINTS
    sides = tuple(side for side in GRID_SIDES if side * side <= max_points)
    return Budget(
        max_points=max_points,
        start_points=max(9, round(START_POINTS * share)),
        small_start_points=max(9, round(SMALL_START_POINTS * share)),
        frame_side=max(2, round(FRAME_SIDE * math.sqrt(share))),
        grid_sides=sides or (MIN_GRID_SIDE,),
    )


def world_landmarks(
    frame: SectionFrame,
    record: DeformableRecord,
    file_to_world: FloatArray,
    native_to_world: FloatArray,
    *,
    tolerance_mm: float = TOLERANCE_MM,
    max_points: int = MAX_POINTS,
) -> WarpLandmarks:
    """Place a warp step's landmark pairs in any ABBA world frame.

    Parameters
    ----------
    frame
        The section's frame.
    record
        The applied deformation.
    file_to_world
        3x3 on ``[x, y, 1]``: a file pixel to ABBA world millimetres after
        the affine step (the section side of the pairs).
    native_to_world
        3x3 on ``[x, y, 1]``: a native atlas-plane pixel to the same world
        millimetres (the atlas side).
    tolerance_mm
        The search error at which pairs stop being added.
    max_points
        The most pairs (:func:`budget`): fewer are easier to edit by hand
        and follow the deformation less closely.

    Returns
    -------
    WarpLandmarks
        The pairs (module text) and their measured accuracy.

    Raises
    ------
    ValueError
        When the tolerance is not positive, the budget out of range, or
        the spline folds at every start and every grid.
    """
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Export tolerance must be positive and finite")
    sizes = budget(max_points=max_points)
    exact = _WorldMap(
        frame,
        record,
        np.asarray(file_to_world, dtype=np.float64),
        np.asarray(native_to_world, dtype=np.float64),
    )

    step_px = max(frame.file_size) / SEARCH_SIDE
    search, region = _samples(exact, step_px, step_px)
    probes, _ = _samples(exact, 2.0 * step_px, step_px, region=region)
    spacing = Spacing(
        min_gap_px=MIN_GAP_STEPS * step_px,
        batch_gap_px=BATCH_GAP_STEPS * step_px,
        step_px=step_px,
    )

    adaptive = _adaptive(exact, search, spacing, tolerance_mm, sizes)
    if adaptive is not None:
        pairs, placement = adaptive.pairs, "adaptive"
        rounds, dropped = adaptive.rounds, adaptive.dropped
    else:
        pairs, placement = _even_grid(exact, search, sizes), "grid"
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


def _lattice(size: tuple[int, int], step_px: float, first_px: float) -> FloatArray:
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
    sizes: Budget,
) -> Placed | None:
    """Place pairs greedily from the first start that does not fold.

    None when every start folds.
    """
    for count in (sizes.start_points, sizes.small_start_points):
        start = _start(exact.frame.file_size, search, count, sizes.frame_side)
        adaptive = refine(
            pairs_at(exact, start),
            exact,
            search,
            spacing,
            budget=sizes.max_points,
            tolerance_mm=tolerance_mm,
        )
        if adaptive is not None:
            return adaptive
    return None


def _start(size: tuple[int, int], search: Samples, count: int, frame_side: int) -> FloatArray:
    """Lay the start's file points.

    About *count* points evenly spaced over the region's bounding box, and
    a coarse snapshot grid of *frame_side* per side outside it.
    """
    inside = search.points_px[search.in_region]
    x_low, x_high = _padded(inside[:, 0], size[0])
    y_low, y_high = _padded(inside[:, 1], size[1])

    width, height = x_high - x_low, y_high - y_low
    columns = max(2, round(math.sqrt(count * width / height)))
    rows = max(2, round(count / columns))
    box = _even((x_low, x_high), (y_low, y_high), columns, rows)

    spacing = max(width / (columns - 1), height / (rows - 1))
    frame = _even(*_snapshot(size), frame_side, frame_side)
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


def _even_grid(exact: _WorldMap, search: Samples, sizes: Budget) -> Pairs:
    """Find the largest even snapshot grid whose spline does not fold."""
    snapshot = _snapshot(exact.frame.file_size)
    sides = sizes.grid_sides
    for side in sides:
        grid = pairs_at(exact, _even(*snapshot, side, side))
        try:
            kernel = pullback(grid.source_mm, grid.target_mm)
        except ValueError:
            continue
        if not np.any(folded_at(kernel, search.target_mm)):
            return grid
    raise ValueError(
        "the deformation's thin-plate spline folds at every start and "
        f"every grid up to {sides[0]}x{sides[0]} points"
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
