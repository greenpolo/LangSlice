"""Landmark pairs whose thin-plate spline follows a dense map closely.

A dense map (here a deformation's) is known exactly at any point of a
section image. A program that only takes landmark pairs (ABBA's BigWarp
step) rebuilds it as the interpolating thin-plate spline (TPS) of those
pairs, so the pairs decide how far the rebuilt map strays from the exact
one between them. This module chooses the pairs under a point budget.

Placement. A start grid covers the region that matters (the atlas-overlap
region, chosen by the caller) at even spacing, with a coarse grid over the
rest of the image so the spline stays tame where nothing is drawn. Then
the points are added greedily: in each round the spline is fitted, its
atlas-side error is estimated at a dense search grid inside the region
(the section-side error times the exact map's derivative there), and up to
:data:`BATCH` new pairs are placed at the largest errors, at least
``batch_gap`` apart and ``min_gap`` from every pair already placed. It
stops at the budget, or when the largest error at the search grid is
within the tolerance. Every step is deterministic.

Folds. A spline whose pull-back has a Jacobian determinant at or below
:data:`MIN_JACOBIAN` anywhere on the search grid (the whole image), or on a
finer lattice around each new pair, folds. Both are refined where the map
stretches the section (its derivative from the caller), so that the
pull-back, which lives on the atlas side, is tested about as densely there.
A round whose new pairs make the spline fold drops the new pairs within
:data:`BLOCK_GAPS` smallest gaps of a folded point (all of them when none
is that near), blocks the search points around them, and fits again; the
pairs kept before the round are never dropped. A start that folds already
is reported as None, and the caller falls back to another.

Accuracy. The chosen spline is measured at held-out probe points (never
used to choose pairs) inside the region, on both sides of the map: on the
section side, how far the spline's pull-back of the exact target lands from
the exact source; on the atlas side, how far the spline's forward map (the
pull-back inverted by Newton's method, as ABBA inverts it) carries a source
from its exact target. The atlas side is where a user sees the section
drawn under the atlas borders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from langslice.core.geometry.thin_plate import ThinPlateKernel

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Pairs added per round of the greedy placement: few enough that each
#: round reacts to the last, many enough that a budget of about a thousand
#: points takes about a dozen rounds.
BATCH = 64
#: The fold check around each new pair: a lattice of LOCAL_SIDE points per
#: side spanning LOCAL_SPAN smallest gaps either side, so its step (a third
#: of a gap) resolves a fold between two pairs as close as allowed.
LOCAL_SIDE = 13
LOCAL_SPAN = 2.0
#: The most a fold-check lattice is refined where the map stretches the
#: section: the pull-back is tested at most this many times more finely
#: per side, so stretched tissue is tested about as densely on the atlas
#: side as unstretched tissue.
MAX_REFINE = 4
#: A new pair within this many smallest gaps of a fold is dropped, and no
#: pair is sought again that close to it: small enough that a steep part of
#: the map, where pairs often fold, still gets the pairs that do not.
BLOCK_GAPS = 2.0
#: Smallest Jacobian determinant accepted (a fold or collapse below it).
MIN_JACOBIAN = 1e-6
#: Newton steps when the pull-back is inverted at a probe point; a step
#: under :data:`NEWTON_STOP_MM` ends it early.
NEWTON_STEPS = 12
#: A Newton step this small (millimetres) has converged.
NEWTON_STOP_MM = 1e-9
#: Points whose nearest centre is sought per block, which bounds the memory
#: a distance query holds.
NEAREST_BLOCK = 2048


class ExactMap(Protocol):
    """A dense map known exactly at any image point."""

    def pairs(self, points_px: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the source and target millimetres of image points.

        Parameters
        ----------
        points_px
            N-by-2 image pixel ``[x, y]`` points.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            The points before the map and where it takes them, N-by-2 each.
        """
        ...


@dataclass(frozen=True)
class Samples:
    """Hold image points with their exact pairs and region membership.

    Parameters
    ----------
    points_px
        N-by-2 image pixel ``[x, y]`` points.
    source_mm
        Where each point sits before the map.
    target_mm
        Where the exact map takes it.
    in_region
        Whether the point lies in the region that matters.
    stretch
        N-by-2-by-2: the exact map's derivative there (target by source),
        which carries a section-side error to the atlas side.
    """

    points_px: FloatArray
    source_mm: FloatArray
    target_mm: FloatArray
    in_region: BoolArray
    stretch: FloatArray


@dataclass(frozen=True)
class Spacing:
    """Hold the distances the greedy placement keeps between pairs.

    Parameters
    ----------
    min_gap_px
        Smallest distance from a new pair to any pair already placed.
    batch_gap_px
        Smallest distance between two pairs added in one round.
    step_px
        The search grid's step.
    """

    min_gap_px: float
    batch_gap_px: float
    step_px: float


@dataclass(frozen=True)
class Pairs:
    """Hold landmark pairs with the image points they come from.

    Parameters
    ----------
    points_px
        The image points.
    source_mm
        Their sources: the spline's landmarks before the map.
    target_mm
        Their targets: where the map takes them.
    """

    points_px: FloatArray
    source_mm: FloatArray
    target_mm: FloatArray

    def joined(self, other: Pairs) -> Pairs:
        """Return these pairs followed by *other*.

        Parameters
        ----------
        other
            The pairs to append.

        Returns
        -------
        Pairs
            Both sets, these first.
        """
        return Pairs(
            points_px=np.vstack((self.points_px, other.points_px)),
            source_mm=np.vstack((self.source_mm, other.source_mm)),
            target_mm=np.vstack((self.target_mm, other.target_mm)),
        )

    def subset(self, kept: BoolArray) -> Pairs:
        """Return the pairs *kept* marks.

        Parameters
        ----------
        kept
            One flag per pair.

        Returns
        -------
        Pairs
            The marked pairs, in order.
        """
        return Pairs(
            points_px=self.points_px[kept],
            source_mm=self.source_mm[kept],
            target_mm=self.target_mm[kept],
        )


def pairs_at(exact: ExactMap, points_px: FloatArray) -> Pairs:
    """Pair image points with where the exact map puts them.

    Parameters
    ----------
    exact
        The map that gives each point its source and target.
    points_px
        N-by-2 image points.

    Returns
    -------
    Pairs
        The points, their sources and their targets.
    """
    source, target = exact.pairs(points_px)
    return Pairs(points_px=points_px, source_mm=source, target_mm=target)


@dataclass(frozen=True)
class Placed:
    """Hold the pairs a placement chose and how it went.

    Parameters
    ----------
    pairs
        The chosen pairs.
    rounds
        Greedy rounds run.
    dropped
        Pairs dropped because they made the spline fold.
    search_max_mm
        The largest atlas-side error estimate at the search grid's region
        points when the placement stopped.
    """

    pairs: Pairs
    rounds: int
    dropped: int
    search_max_mm: float


@dataclass(frozen=True)
class ErrorStats:
    """Hold an error's distribution in millimetres.

    Parameters
    ----------
    median_mm, p95_mm, p99_mm, max_mm
        The median, 95th and 99th percentiles and the maximum.
    """

    median_mm: float
    p95_mm: float
    p99_mm: float
    max_mm: float


@dataclass(frozen=True)
class Accuracy:
    """Hold a spline's measured difference from the exact map.

    Parameters
    ----------
    atlas
        On the atlas side: where the spline draws a section point against
        where the exact map draws it.
    section
        On the section side: where the spline samples the section for a
        target against where the exact map samples it.
    probes
        Held-out probe points measured (inside the region).
    unconverged
        Probes where Newton's method did not converge; their atlas-side
        error is taken as at least the section-side one.
    min_jacobian
        The smallest Jacobian determinant of the pull-back at the probes.
    """

    atlas: ErrorStats
    section: ErrorStats
    probes: int
    unconverged: int
    min_jacobian: float


def pullback(source_mm: FloatArray, target_mm: FloatArray) -> ThinPlateKernel:
    """Fit the spline a landmark program builds from pairs.

    Parameters
    ----------
    source_mm
        The pairs' points before the map.
    target_mm
        Where the map takes them.

    Returns
    -------
    ThinPlateKernel
        The interpolating spline from targets to sources (the pull-back a
        resampler uses).
    """
    return ThinPlateKernel(target_mm, source_mm, max_points=len(source_mm))


def folded_at(kernel: ThinPlateKernel, target_mm: FloatArray) -> BoolArray:
    """Mark the targets where a pull-back spline folds.

    Parameters
    ----------
    kernel
        The pull-back spline.
    target_mm
        Atlas-side points to test.

    Returns
    -------
    BoolArray
        True where the Jacobian determinant is at or below
        :data:`MIN_JACOBIAN`, or not finite.
    """
    determinant = determinants(kernel.jacobian(target_mm))
    return ~np.isfinite(determinant) | (determinant <= MIN_JACOBIAN)


def determinants(matrices: FloatArray) -> FloatArray:
    """Take the determinant of each of N 2-by-2 matrices.

    Parameters
    ----------
    matrices
        N-by-2-by-2 matrices.

    Returns
    -------
    FloatArray
        The N determinants.
    """
    product = matrices[:, 0, 0] * matrices[:, 1, 1]
    return product - matrices[:, 0, 1] * matrices[:, 1, 0]


def _lengths(vectors: FloatArray) -> FloatArray:
    """Measure each of N 2-vectors."""
    return np.hypot(vectors[:, 0], vectors[:, 1])


def refine(
    start: Pairs,
    exact: ExactMap,
    search: Samples,
    spacing: Spacing,
    *,
    budget: int,
    tolerance_mm: float,
) -> Placed | None:
    """Grow a start set of pairs greedily where the spline errs most.

    Parameters
    ----------
    start
        The start's pairs.
    exact
        The map the spline should follow.
    search
        The dense search grid: its region points are where errors are
        sought, all its points where folds are tested.
    spacing
        The distances kept between pairs.
    budget
        The most pairs placed.
    tolerance_mm
        The largest atlas-side search error at which growing stops.

    Returns
    -------
    Placed | None
        The pairs and how the placement went; None when the start's spline
        cannot be built or already folds.
    """
    try:
        kernel = pullback(start.source_mm, start.target_mm)
    except ValueError:
        return None
    checked = _refined(exact, search, spacing.step_px)
    if np.any(folded_at(kernel, checked.target_mm)):
        return None

    current = start
    candidates = search.points_px[search.in_region]
    blocked = np.zeros(len(candidates), dtype=bool)
    rounds = dropped = 0
    error = _atlas_error(kernel, search)
    while len(current.points_px) < budget and np.max(error) > tolerance_mm:
        count = min(BATCH, budget - len(current.points_px))
        order = np.argsort(-error, kind="stable")
        batch = _batch(
            candidates, order, blocked, current.points_px, spacing, count
        )
        if len(batch) == 0:
            break
        rounds += 1

        added = pairs_at(exact, candidates[batch])
        stretch = search.stretch[search.in_region][batch]
        grown = _grow(current, added, stretch, exact, checked, spacing)
        block_px = BLOCK_GAPS * spacing.min_gap_px
        blocked |= _near(candidates, grown.dropped_px, block_px)
        dropped += len(grown.dropped_px)
        if grown.kernel is not None:
            current, kernel = grown.pairs, grown.kernel
            error = _atlas_error(kernel, search)
    return Placed(
        pairs=current,
        rounds=rounds,
        dropped=dropped,
        search_max_mm=float(np.max(error)),
    )


@dataclass(frozen=True)
class _Grown:
    """Hold one round's pairs once its folding pairs are dropped."""

    pairs: Pairs
    kernel: ThinPlateKernel | None
    dropped_px: FloatArray


def _grow(
    current: Pairs,
    added: Pairs,
    stretch: FloatArray,
    exact: ExactMap,
    checked: Pairs,
    spacing: Spacing,
) -> _Grown:
    """Add pairs to a non-folding set, dropping those that make it fold.

    Folds are tested at the *checked* points over the whole image and on
    a finer lattice around each new pair (refined by its *stretch*), where
    two close pairs can fold the spline between search points.
    """
    around = _around(added.points_px, stretch, spacing.min_gap_px)
    local = pairs_at(exact, around)
    tested_px = np.vstack((checked.points_px, local.points_px))
    tested_mm = np.vstack((checked.target_mm, local.target_mm))
    kept = np.ones(len(added.points_px), dtype=bool)
    while kept.any():
        trial = current.joined(added.subset(kept))
        try:
            kernel = pullback(trial.source_mm, trial.target_mm)
        except ValueError:
            kernel = None
            folds = tested_px
        else:
            folds = tested_px[folded_at(kernel, tested_mm)]

        if kernel is not None and len(folds) == 0:
            dropped = added.points_px[~kept]
            return _Grown(pairs=trial, kernel=kernel, dropped_px=dropped)
        radius_px = BLOCK_GAPS * spacing.min_gap_px
        kept &= ~_culprits(added.points_px, kept, folds, radius_px)
    return _Grown(pairs=current, kernel=None, dropped_px=added.points_px)


def _around(
    points_px: FloatArray, stretch: FloatArray, gap_px: float
) -> FloatArray:
    """Lay a fine square lattice around each point.

    Each spans :data:`LOCAL_SPAN` gaps either side, with
    :data:`LOCAL_SIDE` points per side, refined where the map stretches.
    """
    factors = _refinement(stretch)
    lattices = [np.empty((0, 2))]
    for factor in cast("list[int]", np.unique(factors).tolist()):
        side = (LOCAL_SIDE - 1) * factor + 1
        span = LOCAL_SPAN * gap_px
        offsets = np.linspace(-span, span, side)
        square = _square(offsets)
        centres = points_px[factors == factor]
        lattices.append((centres[:, None, :] + square[None]).reshape(-1, 2))
    return np.vstack(lattices)


def _refined(exact: ExactMap, search: Samples, step_px: float) -> Pairs:
    """Spread fold-check points over each search point's cell.

    A cell gets k by k points where the map stretches the section k-fold
    (at most :data:`MAX_REFINE`), so their targets lie about a search step
    apart on the atlas side too.
    """
    factors = _refinement(search.stretch)
    lattices = [search.points_px[factors == 1]]
    for factor in cast("list[int]", np.unique(factors[factors > 1]).tolist()):
        offsets = ((np.arange(factor) + 0.5) / factor - 0.5) * step_px
        centres = search.points_px[factors == factor]
        square = _square(offsets)
        lattices.append((centres[:, None, :] + square[None]).reshape(-1, 2))
    return pairs_at(exact, np.vstack(lattices))


def _refinement(stretch: FloatArray) -> NDArray[np.int64]:
    """Count the refinement per side a stretch asks for (1 to MAX_REFINE).

    The count is the stretch's largest singular value, rounded up.
    """
    flat = stretch.reshape(-1, 4)
    squares = flat * flat
    frobenius = squares[:, 0] + squares[:, 1] + squares[:, 2] + squares[:, 3]
    determinant = determinants(stretch)
    spread = np.sqrt(np.maximum(frobenius**2 - 4.0 * determinant**2, 0.0))
    largest = np.sqrt(0.5 * (frobenius + spread))
    factors = np.ceil(np.nan_to_num(largest, nan=MAX_REFINE))
    return np.clip(factors, 1, MAX_REFINE).astype(np.int64)


def _square(offsets: FloatArray) -> FloatArray:
    """Lay the square lattice of all pairs of *offsets*."""
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _culprits(
    added_px: FloatArray,
    kept: BoolArray,
    folds_px: FloatArray,
    radius_px: float,
) -> BoolArray:
    """Mark the kept new pairs near a fold, or all when none is near.

    A pair is near within *radius_px*.
    """
    near = kept & _near(added_px, folds_px, radius_px)
    if near.any():
        return near
    return kept


def _near(
    candidates_px: FloatArray,
    centres_px: FloatArray,
    radius_px: float,
) -> BoolArray:
    """Mark the candidates within a radius of any centre."""
    if len(centres_px) == 0:
        return np.zeros(len(candidates_px), dtype=bool)
    return _nearest(candidates_px, centres_px) <= radius_px


def _nearest(points_px: FloatArray, centres_px: FloatArray) -> FloatArray:
    """Measure each point's distance to its nearest centre."""
    nearest = np.empty(len(points_px))
    for start in range(0, len(points_px), NEAREST_BLOCK):
        block = points_px[start : start + NEAREST_BLOCK]
        dx = block[:, None, 0] - centres_px[None, :, 0]
        dy = block[:, None, 1] - centres_px[None, :, 1]
        squared = cast("FloatArray", np.min(dx * dx + dy * dy, axis=1))
        nearest[start : start + NEAREST_BLOCK] = np.sqrt(squared)
    return nearest


def _batch(
    candidates_px: FloatArray,
    order: NDArray[np.intp],
    blocked: BoolArray,
    points_px: FloatArray,
    spacing: Spacing,
    count: int,
) -> NDArray[np.intp]:
    """Choose up to *count* candidates in *order*, kept apart."""
    gap = _nearest(candidates_px, points_px)
    eligible = order[~blocked[order] & (gap[order] > spacing.min_gap_px)]
    batch: list[int] = []
    for index in cast("list[int]", eligible.tolist()):
        if batch:
            chosen = candidates_px[batch]
            apart = _lengths(chosen - candidates_px[index : index + 1])
            if np.min(apart) <= spacing.batch_gap_px:
                continue
        batch.append(index)
        if len(batch) == count:
            break
    return np.asarray(batch, dtype=np.intp)


def _atlas_error(kernel: ThinPlateKernel, search: Samples) -> FloatArray:
    """Estimate the atlas-side error at the search grid's region points.

    The section-side error carried through the exact map's derivative:
    where the deformation stretches the section, a small section-side
    error is a large one under the atlas borders.
    """
    inside = search.in_region
    estimate = kernel.forward(search.target_mm[inside])
    section = estimate - search.source_mm[inside]
    stretch = search.stretch[inside]
    atlas = (
        stretch[:, :, 0] * section[:, :1] + stretch[:, :, 1] * section[:, 1:]
    )
    return _lengths(atlas)


def accuracy(kernel: ThinPlateKernel, probes: Samples) -> Accuracy:
    """Measure a pull-back spline against the exact map at held-out probes.

    Parameters
    ----------
    kernel
        The pull-back spline the pairs build.
    probes
        Points never used to choose the pairs; those in the region are
        measured.

    Returns
    -------
    Accuracy
        Both sides' error distributions and the smallest Jacobian.
    """
    inside = probes.in_region
    source, target = probes.source_mm[inside], probes.target_mm[inside]
    section = _lengths(kernel.forward(target) - source)

    landed, converged = _inverse(kernel, source, target)
    atlas = _lengths(landed - target)
    atlas[~converged] = np.maximum(atlas[~converged], section[~converged])

    determinant = determinants(kernel.jacobian(target))
    return Accuracy(
        atlas=_stats(atlas),
        section=_stats(section),
        probes=len(source),
        unconverged=int(np.sum(~converged)),
        min_jacobian=float(np.min(determinant)),
    )


def _inverse(
    kernel: ThinPlateKernel,
    source_mm: FloatArray,
    start_mm: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Invert the pull-back at *source_mm* by Newton's method.

    Returns the points it takes there, started from *start_mm*, and
    whether each converged.
    """
    landed = start_mm.copy()
    step_mm = np.full(len(landed), np.inf)
    for _ in range(NEWTON_STEPS):
        residual = kernel.forward(landed) - source_mm
        jacobian = kernel.jacobian(landed)
        solved = np.linalg.solve(jacobian, residual[..., None])
        step = cast("FloatArray", solved[..., 0])
        landed -= step
        step_mm = _lengths(step)
        if np.max(step_mm) < NEWTON_STOP_MM:
            break
    residual_mm = _lengths(kernel.forward(landed) - source_mm)
    return landed, (residual_mm < 1e-6) & np.isfinite(step_mm)


def _stats(error_mm: FloatArray) -> ErrorStats:
    """Summarize an error by its median, percentiles and maximum."""
    if len(error_mm) == 0:
        return ErrorStats(median_mm=0.0, p95_mm=0.0, p99_mm=0.0, max_mm=0.0)
    return ErrorStats(
        median_mm=float(np.percentile(error_mm, 50)),
        p95_mm=float(np.percentile(error_mm, 95)),
        p99_mm=float(np.percentile(error_mm, 99)),
        max_mm=float(np.max(error_mm)),
    )
