"""Anchor seeding: get a first, globally consistent set of positions on the stack.

Plain code, no stack-aware agent. A handful of sections spread center-out
across the corrected order are estimated with the single-slice worker, and
every other section is filled in by interpolating between them. The result is
a starting point for the positioning step, not an answer: anchors carry
``position_source="anchor"``, everything else ``"interpolated"``.

Center-out on purpose: the extremes of a stack (very anterior sections in
mouse especially) carry the least distinctive anatomy, so they are the worst
sections to spend an anchor estimate on.
"""

from __future__ import annotations

import logging
from collections.abc import Collection

from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.estimation_agents import run_slice_estimation
from langslice.linear.whole_brain.signals import interpolate_positions
from langslice.linear.whole_brain.state import SliceState, StackState

logger = logging.getLogger(__name__)

#: Anchor budget: small stacks get fewer estimates, larger stacks a few more.
ANCHORS_SMALL_STACK = 4
ANCHORS_LARGE_STACK = 6
SMALL_STACK_MAX = 20

DAMAGED_CAVEAT = "position interpolated; slice damaged"


def anchor_budget(n_slices: int) -> int:
    """How many anchors to spend on a stack of *n_slices* sections."""
    return ANCHORS_SMALL_STACK if n_slices <= SMALL_STACK_MAX else ANCHORS_LARGE_STACK


def select_anchor_indices(
    n_slices: int, n_anchors: int, *, damaged: Collection[int] = ()
) -> list[int]:
    """Pick *n_anchors* corrected indices, spread center-out, skipping damaged.

    Anchors are evenly spaced with a half-gap inset at each end, so index 0 and
    index n-1 are avoided unless the budget covers the whole stack. A candidate
    that lands on a damaged section walks outward to the nearest undamaged one.
    """
    if n_slices <= 0 or n_anchors <= 0:
        return []
    usable = [i for i in range(n_slices) if i not in set(damaged)]
    if not usable:
        return []
    n_anchors = min(n_anchors, len(usable))
    if n_anchors == len(usable):
        return usable

    usable_set = set(usable)
    chosen: list[int] = []
    gap = n_slices / (n_anchors + 1)
    for k in range(1, n_anchors + 1):
        ideal = min(n_slices - 1, max(0, int(round(k * gap)) - 1))
        pick = _nearest_usable(ideal, usable_set, chosen, n_slices)
        if pick is not None:
            chosen.append(pick)

    # Rounding can collapse two candidates onto one section; top up center-out.
    middle = n_slices // 2
    for offset in range(n_slices):
        if len(chosen) >= n_anchors:
            break
        for candidate in (middle + offset, middle - offset):
            if candidate in usable_set and candidate not in chosen:
                chosen.append(candidate)
                break
    return sorted(chosen)


def _nearest_usable(
    ideal: int, usable: set[int], taken: Collection[int], n_slices: int
) -> int | None:
    """The undamaged, not-yet-taken index closest to *ideal*."""
    for offset in range(n_slices):
        for candidate in (ideal + offset, ideal - offset):
            if candidate in usable and candidate not in taken:
                return candidate
    return None


async def seed_positions(
    state: StackState, ctx: EngineContext, *, pos_lo: float, pos_hi: float
) -> tuple[int, int]:
    """Seed every section's position; return ``(anchors_estimated, interpolated)``.

    Anchor estimates run one at a time — they are full single-slice agent
    sessions and running them concurrently is the fastest way to hit a
    provider rate limit. A failed anchor is retried once on its nearest
    undamaged neighbour; if every anchor fails there is nothing to interpolate
    from and the step raises.
    """
    ordered = state.in_order()
    n = len(ordered)
    if n == 0:
        return (0, 0)

    damaged = {i for i, record in enumerate(ordered) if record.damaged}
    if len(damaged) == n:
        # Nothing undamaged to anchor on: damaged sections are still better
        # than no positions at all.
        state.notes.append("seed: every section is marked damaged; anchoring anyway")
        damaged = set()

    planned = select_anchor_indices(n, anchor_budget(n), damaged=damaged)
    taken = set(planned)
    anchored = 0
    for index in planned:
        placed = await _estimate_anchor(index, ordered, state, ctx, pos_lo, pos_hi)
        if placed is None:
            replacement = _nearest_usable(index, set(range(n)) - damaged, taken, n)
            if replacement is None:
                continue
            taken.add(replacement)
            state.notes.append(
                f"seed: retrying anchor on {ordered[replacement].id} "
                f"(nearest undamaged neighbour)"
            )
            placed = await _estimate_anchor(
                replacement, ordered, state, ctx, pos_lo, pos_hi
            )
        if placed is not None:
            anchored += 1

    if anchored == 0:
        raise RuntimeError(
            f"seed: all {len(planned)} anchor estimates failed; no positions to "
            "interpolate from (see the run log for the individual errors)"
        )

    filled = _fill_between_anchors(state, ordered, pos_lo=pos_lo, pos_hi=pos_hi)
    return (anchored, filled)


async def _estimate_anchor(
    index: int,
    ordered: list[SliceState],
    state: StackState,
    ctx: EngineContext,
    pos_lo: float,
    pos_hi: float,
) -> float | None:
    """Estimate one anchor and write it, or return None on failure."""
    record = ordered[index]
    try:
        result = await run_slice_estimation(
            image_path=ctx.image_path(record.id),
            atlas_name=state.atlas,
            plane=state.plane,
            model_name=ctx.model,
        )
    except Exception as exc:
        logger.warning("seed: anchor estimate failed for %s: %s", record.id, exc)
        state.notes.append(f"seed: anchor {record.id} failed ({exc})")
        return None
    record.position_mm = min(pos_hi, max(pos_lo, float(result.position_mm)))
    record.position_source = "anchor"
    ctx.progress(f"[seed] anchor {record.id} -> {record.position_mm:.3f} mm")
    return record.position_mm


def _fill_between_anchors(
    state: StackState, ordered: list[SliceState], *, pos_lo: float, pos_hi: float
) -> int:
    """Interpolate every non-anchor section from the anchors. Returns the count."""
    known: list[float | None] = [
        record.position_mm if record.position_source == "anchor" else None
        for record in ordered
    ]
    anchors = [value for value in known if value is not None]
    # Beyond the outermost anchors we step by the nominal interval; the sign
    # says which way the stack runs. With one anchor there is nothing to
    # compare, so assume the stack runs with the atlas axis.
    # ponytail: direction from the anchors, not from axis_directions — the
    # anchors are measurements, the survey's reading is a claim.
    forward = len(anchors) < 2 or anchors[-1] >= anchors[0]
    step = state.interval_mm if forward else -state.interval_mm

    filled = interpolate_positions(known, interval_mm=step)
    count = 0
    for record, value in zip(ordered, filled, strict=True):
        if record.position_source == "anchor":
            continue
        record.position_mm = min(pos_hi, max(pos_lo, float(value)))
        record.position_source = "interpolated"
        count += 1
        if record.damaged and DAMAGED_CAVEAT not in record.caveats:
            record.caveats.append(DAMAGED_CAVEAT)
    return count
