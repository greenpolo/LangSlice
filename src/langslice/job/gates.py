"""The submit gates: what a stack needs before it can be submitted."""

from __future__ import annotations

from typing import Any

from langslice.core.fits import deformation
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.core.stack.section import INITIAL
from langslice.core.stack.stack import Stack
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tools_on import ToolsOn

INTERVAL_BREAK_MIN_RATIO = 1.5
STRICT_INTERVAL_TOLERANCE = 0.10


def _gate(code: RefusalCode, message: str, **facts: object) -> Refused:
    """Return a submit gate's refusal: status ``refused``, then *message*.

    The call is valid; the stack is not ready for it.
    """
    return Refused(code, message, status=ReplyStatus.REFUSED, **facts)


def _tools_in_brackets(on: ToolsOn, tools: tuple[str, ...]) -> str:
    """The tools of *tools* that are on, as " (a or b)"; empty when none is."""
    named = on.listed([(name, name) for name in tools], last="or")
    return f" ({named})" if named else ""


def missing_positions(state: Stack, on: ToolsOn) -> Refused | None:
    """``None`` when every section's position was supplied or written, else
    the refusal naming the sections still at their initial position."""
    missing = [record.id for record in state.in_order() if record.position_source == INITIAL]
    if not missing:
        return None
    return _gate(
        RefusalCode.GATE_REFUSED,
        f"{len(missing)} of {len(state.slices)} section(s) are still at their "
        "initial position (the job's start and spacing); every section needs "
        "a position written, damaged ones included: write every section's own position "
        "first" + on.say("position_sections", " with position_sections") + ".",
        missing_ids=missing,
    )


def strict_interval_error(state: Stack, breaks: list[int]) -> Refused | None:
    """Every consecutive spacing within 10% of the interval, no breaks."""
    if breaks:
        return _gate(
            RefusalCode.GATE_REFUSED,
            f"strict_interval is on: interval_breaks must be empty, {len(breaks)} were reported. "
            f"Make every spacing within {STRICT_INTERVAL_TOLERANCE:.0%} of the nominal interval "
            "and report no break (the user asked for a strict interval).",
            reported_breaks=[
                record.id for record in state.in_order() if record.index_corrected in set(breaks)
            ],
        )
    interval = float(state.interval_mm or 0.0)
    if interval <= 0:
        return None
    ordered = state.in_order()
    tolerance = interval * STRICT_INTERVAL_TOLERANCE
    failures: list[dict[str, Any]] = []
    for before, after in zip(ordered, ordered[1:], strict=False):
        spacing = abs(after.position_mm - before.position_mm)
        if abs(spacing - interval) > tolerance:
            failures.append(
                {
                    "between": [before.id, after.id],
                    "spacing_mm": round(spacing, 3),
                }
            )
    if not failures:
        return None
    return _gate(
        RefusalCode.GATE_REFUSED,
        f"strict_interval is on: every consecutive spacing must be within "
        f"{STRICT_INTERVAL_TOLERANCE:.0%} of {interval:.3f} mm; "
        f"{len(failures)} pair(s) are not. Make every spacing within "
        f"{STRICT_INTERVAL_TOLERANCE:.0%} of the nominal interval and report no break "
        "(the user asked for a strict interval).",
        interval_mm=round(interval, 3),
        tolerance_mm=round(tolerance, 3),
        failures=failures,
    )


def unsupported_breaks(state: Stack, breaks: list[int]) -> Refused | None:
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
        abs(b.position_mm - a.position_mm) for a, b in zip(ordered, ordered[1:], strict=False)
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
                    **({"id": ordered[position].id} if position is not None else {}),
                    "error": RefusalCode.GATE_REFUSED,
                    "reason": (
                        "no section comes before it in the stack's order; a break "
                        "names the section AFTER the gap."
                    ),
                }
            )
            continue
        before, after = ordered[position - 1], ordered[position]
        written = abs(after.position_mm - before.position_mm)
        if written > threshold:
            continue
        failures.append(
            {
                "id": after.id,
                "error": RefusalCode.GATE_REFUSED,
                "written_interval_mm": round(written, 3),
                "median_interval_mm": round(median, 3),
                "between": [before.id, after.id],
            }
        )
    if not failures:
        return None
    return _gate(
        RefusalCode.GATE_REFUSED,
        f"{len(failures)} reported interval break(s) are not in the "
        "positions you wrote. A break is accepted only where the "
        f"written interval exceeds {INTERVAL_BREAK_MIN_RATIO:g}x the "
        "stack's median written spacing; check the positions on both sides of each reported "
        "break first.",
        failures=failures,
    )


def missing_transforms(state: Stack, on: ToolsOn) -> Refused | None:
    """``None`` when every section carries a transform, else the refusal."""
    missing = [record.id for record in state.in_order() if record.transform is None]
    if not missing:
        return None
    return _gate(
        RefusalCode.GATE_REFUSED,
        f"{len(missing)} of {len(state.slices)} section(s) carry no "
        "transform; every section needs one, damaged ones included: give every section a "
        "transform" + _tools_in_brackets(on, ("elastix_affine", "interactive_transform")) + ".",
        missing_ids=missing,
    )


def missing_deformations(
    state: Stack,
    on: ToolsOn,
    exempt: frozenset[str] | set[str] = frozenset(),
) -> Refused | None:
    """``None`` when every section carries a deformation at its current linear
    placement, or else the refusal. The sections in *exempt* (their
    deformation frozen) need none."""
    missing: list[dict[str, str]] = []
    for section in state.in_order():
        if section.id in exempt:
            continue
        held = section.deformation
        if held is None:
            missing.append({"id": section.id, "reason": "no deformation"})
        elif held.linear_key != deformation.linear_key(state, section):
            missing.append(
                {
                    "id": section.id,
                    "reason": "made at a different linear placement",
                }
            )
    if not missing:
        return None
    return _gate(
        RefusalCode.GATE_REFUSED,
        f"{len(missing)} of {len(state.slices)} section(s) carry no deformation at their "
        "current placement. The user requires a deformation on every section: give every "
        "section a deformation" + _tools_in_brackets(on, ("fireants_syn", "trace_borders")) + ".",
        sections=missing,
    )


def submit_errors(
    state: Stack,
    spec: JobSpec,
    breaks: list[int],
) -> Refused | None:
    """Every gate that applies to this run, in order. The deformation gate
    applies only when the host requires one (``NonlinearSpec.require_deformation``)."""
    if spec.has("position"):
        refusal = missing_positions(state, spec.tools_on) or (
            strict_interval_error(state, breaks)
            if spec.position.strict_interval
            else unsupported_breaks(state, breaks)
        )
        if refusal is not None:
            return refusal
    if spec.has("transform"):
        refusal = missing_transforms(state, spec.tools_on)
        if refusal is not None:
            return refusal
    if spec.has("nonlinear") and spec.nonlinear.require_deformation:
        frozen = {
            record.id for record in state.slices if "deformation" in spec.frozen_for(record.id)
        }
        return missing_deformations(state, spec.tools_on, frozen)
    return None
