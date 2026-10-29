"""The submit gates: what a stack needs before it can be submitted."""

from __future__ import annotations

from typing import Any

from langslice.core.fits import deformation
from langslice.core.stack.stack import StackState
from langslice.job.settings.spec import JobSpec

#: A reported interval break must exceed this multiple of the median spacing.
INTERVAL_BREAK_MIN_RATIO = 1.5
#: With ``strict_interval``, every spacing within this fraction of the interval.
STRICT_INTERVAL_TOLERANCE = 0.10


# --- the submit gates ------------------------------------------------------------


def missing_positions(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section has a position, else the rejection
    naming the sections without one."""
    missing = [
        record.id for record in state.in_order() if record.position_mm is None
    ]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_POSITIONS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) have no position; "
            "every section needs one, damaged ones included."
        ),
    }


def strict_interval_error(
    state: StackState, breaks: list[int]
) -> dict[str, Any] | None:
    """Every consecutive spacing within 10% of the interval, no breaks."""
    if breaks:
        return {
            "status": "error",
            "error": "STRICT_INTERVAL",
            "reported_breaks": [
                record.id
                for record in state.in_order()
                if record.index_corrected in set(breaks)
            ],
            "message": (
                "strict_interval is on: interval_breaks must be empty, "
                f"{len(breaks)} were reported."
            ),
        }
    interval = float(state.interval_mm or 0.0)
    if interval <= 0:
        return None
    placed = [s for s in state.in_order() if s.position_mm is not None]
    tolerance = interval * STRICT_INTERVAL_TOLERANCE
    failures: list[dict[str, Any]] = []
    for before, after in zip(placed, placed[1:], strict=False):
        spacing = abs(float(after.position_mm) - float(before.position_mm))  # type: ignore[arg-type]
        if abs(spacing - interval) > tolerance:
            failures.append(
                {
                    "between": [before.id, after.id],
                    "spacing_mm": round(spacing, 3),
                }
            )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "STRICT_INTERVAL",
        "interval_mm": round(interval, 3),
        "tolerance_mm": round(tolerance, 3),
        "failures": failures,
        "message": (
            f"strict_interval is on: every consecutive spacing must be within "
            f"{STRICT_INTERVAL_TOLERANCE:.0%} of {interval:.3f} mm; "
            f"{len(failures)} pair(s) are not."
        ),
    }


def unsupported_breaks(
    state: StackState, breaks: list[int]
) -> dict[str, Any] | None:
    """Check reported interval breaks against the spacing that was WRITTEN.

    A break at the section of corrected index *i* (``submit`` names it by
    filename) claims the gap between *i-1* and *i* is larger than the rest
    of the stack's; the positions on the state make that
    claim checkable without an image.
    """
    if not breaks:
        return None
    ordered = state.in_order()
    deltas = [
        abs(float(b.position_mm) - float(a.position_mm))  # type: ignore[arg-type]
        for a, b in zip(ordered, ordered[1:], strict=False)
        if a.position_mm is not None and b.position_mm is not None
    ]
    if not deltas:
        return None
    median = float(sorted(deltas)[len(deltas) // 2])
    threshold = median * INTERVAL_BREAK_MIN_RATIO

    failures: list[dict[str, Any]] = []
    for index in sorted(set(breaks)):
        position = next(
            (i for i, r in enumerate(ordered) if r.index_corrected == index),
            None,
        )
        if position is None or position == 0:
            failures.append(
                {
                    **(
                        {"id": ordered[position].id}
                        if position is not None
                        else {}
                    ),
                    "error": "NOT_A_GAP",
                    "reason": (
                        "no section comes before it in the stack's order; a break "
                        "names the section AFTER the gap."
                    ),
                }
            )
            continue
        before, after = ordered[position - 1], ordered[position]
        if before.position_mm is None or after.position_mm is None:
            continue
        written = abs(float(after.position_mm) - float(before.position_mm))
        if written > threshold:
            continue
        failures.append(
            {
                "id": after.id,
                "error": "NOT_A_GAP",
                "written_interval_mm": round(written, 3),
                "median_interval_mm": round(median, 3),
                "between": [before.id, after.id],
            }
        )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "INTERVAL_BREAKS_UNSUPPORTED",
        "failures": failures,
        "message": (
            f"{len(failures)} reported interval break(s) are not in the "
            "positions you wrote. A break is accepted only where the "
            f"written interval exceeds {INTERVAL_BREAK_MIN_RATIO:g}x the "
            "stack's median written spacing."
        ),
    }


def missing_transforms(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section carries a transform, else the rejection."""
    missing = [
        record.id for record in state.in_order() if record.transform is None
    ]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_TRANSFORMS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no "
            "transform; every section needs one, damaged ones included."
        ),
    }


def missing_deformations(
    state: StackState,
    exempt: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any] | None:
    """``None`` when every section carries a deformation at its current linear
    placement, or else the rejection. The sections in *exempt* (their
    deformation frozen) need none."""
    missing: list[dict[str, str]] = []
    for record in state.in_order():
        if record.id in exempt:
            continue
        held = record.deformation or {}
        if not held:
            missing.append({"id": record.id, "reason": "no deformation"})
        elif held.get("linear_key") != deformation.linear_key(state, record):
            missing.append(
                {
                    "id": record.id,
                    "reason": "made at a different linear placement",
                }
            )
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_DEFORMATIONS",
        "sections": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no deformation at their "
            "current placement. The user requires a deformation on every section: fit one on each."
        ),
    }


def submit_errors(
    state: StackState,
    spec: JobSpec,
    breaks: list[int],
) -> dict[str, Any] | None:
    """Every gate that applies to this run, in order. The deformation gate
    applies only when the host requires one (``NonlinearSpec.require_deformation``)."""
    if spec.has("position"):
        refusal = missing_positions(state) or (
            strict_interval_error(state, breaks)
            if spec.position.strict_interval
            else unsupported_breaks(state, breaks)
        )
        if refusal is not None:
            return refusal
    if spec.has("transform"):
        refusal = missing_transforms(state)
        if refusal is not None:
            return refusal
    if spec.has("nonlinear") and spec.nonlinear.require_deformation:
        frozen = {
            record.id
            for record in state.slices
            if "deformation" in spec.frozen_for(record.id)
        }
        return missing_deformations(state, frozen)
    return None
