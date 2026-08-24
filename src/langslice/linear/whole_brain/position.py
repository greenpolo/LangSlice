"""The positioning agent: place the whole stack, with the whole stack in view.

The stack arrives unplaced. The agent gets data tools — look at sections, fetch
atlas levels, read the atlas annotation, interpolate between points it fixes,
read back the positions it has written — and reasons its own way to a placement.
Nothing here prescribes an approach, and no tool payload carries an opinion:
tools report numbers and the agent judges them.

Everything the agent writes goes through ``set_positions`` onto the
:class:`~langslice.linear.whole_brain.state.StackState` the node handed in, so
a pass that runs out of turns still leaves its accepted positions behind.
``interpolate_between`` deliberately does NOT write: it computes, and the agent
decides what to keep.

The refusals in ``submit_positions`` are the exception, and they are
constraints, not advice: a submission that contradicts the atlas annotation, or
reports a gap the agent's own numbers do not contain, is rejected with the
facts that rejected it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.adk.model_resolver import default_http_options, resolve_adk_model
from langslice.atlas.landmarks import (
    axis_range_of,
    find_structure,
    near_misses,
    structure_count,
    structures_at,
)
from langslice.linear.tools import fetch_atlas
from langslice.linear.whole_brain._step_common import (
    build_stack_manifest,
    make_view_slices,
    run_agent_session,
    stack_image_parts,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.signals import interpolate_positions
from langslice.linear.whole_brain.state import StackState, apply_confidence
from langslice.space import Plane

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one positioning pass may spend.
DEFAULT_POSITION_MAX_ITERATIONS = 40

#: Levels one ``atlas_structures_at`` call may report on.
MAX_STRUCTURE_LEVELS = 8

#: Structures one ``structure_range`` call may look up.
MAX_STRUCTURE_QUERIES = 10

#: An end anchor's structure must span no more than this fraction of the
#: atlas's full slicing-axis extent. A structure that runs most of the
#: brain's length (cortex, say) is "present" almost everywhere and so proves
#: nothing about where a section sits — it is gameable by construction. Kept
#: tight on purpose: at 25% a mouse-atlas anchor could still be 3 mm wide,
#: wider than the placement errors the gate exists to catch.
MAX_ANCHOR_SPAN_FRACTION = 0.08

#: A reported interval break must be at least this much wider than the
#: stack's own median written spacing. Below it, the "break" is not in the
#: numbers the agent itself wrote, and reporting it downstream is fiction.
INTERVAL_BREAK_MIN_RATIO = 1.5

DEFAULT_POSITION_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_position"

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

_NUDGE_NO_TOOL = (
    "You did not call a tool. Continue with the tools rather than in prose; "
    "when every section has a position, call `submit_positions`."
)
_NUDGE_CONTINUE = (
    "Continue; when every section has a position, call `submit_positions`."
)


# --- outcome -------------------------------------------------------------


@dataclass
class PositionOutcome:
    """What one positioning session produced.

    ``findings`` is ``None`` when the agent never submitted (turn budget
    exhausted); any positions its tools already wrote stay on the state.
    """

    findings: dict[str, Any] | None = None
    positions_written: int = 0
    tool_calls: int = 0
    turns: int = 0


@dataclass
class PositionToolBox:
    """Tool callables plus the mutable results the node reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)


# --- the stack's own numbers ---------------------------------------------


def position_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order: index, id, position, spacing.

    ``position_mm`` is ``None`` for a section that has none.
    ``spacing_to_next_mm`` is the distance to the next section in corrected
    order that carries a position, and ``None`` when this section has no
    position or no placed section follows it.

    Data only: no comparison against the nominal interval, no verdict. The
    agent does its own arithmetic on these numbers.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        here = record.position_mm
        spacing: float | None = None
        if here is not None:
            following = next(
                (
                    value
                    for r in ordered[index + 1 :]
                    if (value := r.position_mm) is not None
                ),
                None,
            )
            if following is not None:
                spacing = round(following - here, 3)
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "spacing_to_next_mm": spacing,
            }
        )
    return rows


# --- the end-anchor gate -------------------------------------------------


def check_end_anchors(
    state: StackState, atlas: Any, anchors: Any, *, axis_extent_mm: float
) -> dict[str, Any] | None:
    """Check the two end anchors against the atlas annotation.

    Returns ``None`` when both hold, or the error dict ``submit_positions``
    should hand back. This is the one check in the whole positioning step that
    is not the agent marking its own homework: the structure the agent names
    has an existence range in the annotation volume, and a section placed
    outside that range is placed wrong — provided the structure is narrow
    enough to mean anything (see :data:`MAX_ANCHOR_SPAN_FRACTION`).

    ``axis_extent_mm`` is the atlas's full extent along the slicing axis
    (``pos_hi - pos_lo``), the yardstick a structure's own span is judged
    against.
    """
    if not structure_count(atlas):
        # No structure tree: nothing here can answer, and refusing every
        # submission over it would deadlock the run. The caller decides.
        raise ValueError(
            f"atlas {state.atlas!r} exposes no structure tree to check anchors against"
        )

    ordered = state.in_order()
    required = [ordered[0].id, ordered[-1].id]
    entries = (
        [a for a in anchors if isinstance(a, dict)]
        if isinstance(anchors, (list, tuple))
        else []
    )
    ids = {str(entry.get("id", "")) for entry in entries}
    if ids != set(required) or len(entries) != len(set(required)):
        return {
            "status": "error",
            "error": "END_ANCHORS_REQUIRED",
            "required_ids": required,
            "message": (
                "submit_positions requires one end anchor for each END of the "
                f"corrected order — {required[0]} and {required[-1]} — as "
                '{"id": ..., "structure": "<acronym>", "note": "<what you '
                'saw>"}.'
            ),
        }

    plane: Plane = state.plane  # type: ignore[assignment]
    tolerance = max(float(state.thickness_mm), 0.0)
    failures: list[dict[str, Any]] = []
    for entry in entries:
        slice_id = str(entry.get("id", ""))
        record = state.by_id(slice_id)
        if record is None or record.position_mm is None:
            continue  # the missing-positions check already covers this
        position = float(record.position_mm)
        query = str(entry.get("structure", "")).strip()
        structure = find_structure(atlas, query) if query else None
        if structure is None:
            failures.append(
                {
                    "id": slice_id,
                    "structure": query,
                    "error": "UNKNOWN_STRUCTURE",
                    "near_misses": near_misses(atlas, query) if query else [],
                    "reason": f"no structure in {state.atlas} matches {query!r}.",
                }
            )
            continue
        span = axis_range_of(atlas, structure["acronym"], plane)
        if span is None:
            failures.append(
                {
                    "id": slice_id,
                    "structure": structure["acronym"],
                    "error": "STRUCTURE_NOT_ANNOTATED",
                    "reason": (
                        f"{structure['acronym']} ({structure['name']}) has no "
                        f"voxels in the {state.atlas} annotation."
                    ),
                }
            )
            continue
        lo, hi = span
        span_fraction = (hi - lo) / axis_extent_mm if axis_extent_mm > 0 else 0.0
        if span_fraction > MAX_ANCHOR_SPAN_FRACTION:
            failures.append(
                {
                    "id": slice_id,
                    "structure": structure["acronym"],
                    "structure_name": structure["name"],
                    "span_mm": [round(lo, 3), round(hi, 3)],
                    "span_fraction": round(span_fraction, 2),
                    "error": "STRUCTURE_TOO_BROAD",
                    "reason": (
                        f"{structure['acronym']} spans {span_fraction * 100:.0f}% "
                        f"of the slicing axis ({hi - lo:.2f} mm); an end anchor "
                        f"may span at most "
                        f"{MAX_ANCHOR_SPAN_FRACTION * 100:.0f}%."
                    ),
                }
            )
            continue
        if not (lo - tolerance <= position <= hi + tolerance):
            failures.append(
                {
                    "id": slice_id,
                    "structure": structure["acronym"],
                    "structure_name": structure["name"],
                    "position_mm": round(position, 3),
                    "atlas_span_mm": [round(lo, 3), round(hi, 3)],
                    "error": "OUTSIDE_STRUCTURE_SPAN",
                    "reason": (
                        f"{structure['acronym']} exists from {lo:.3f} to "
                        f"{hi:.3f} mm along the slicing axis; you placed "
                        f"{slice_id} at {position:.3f} mm, outside that span "
                        f"(tolerance {tolerance:.3f} mm)."
                    ),
                }
            )

    if not failures:
        return None
    return {
        "status": "error",
        "error": "END_ANCHOR_FAILED",
        "failures": failures,
        "message": (
            f"{len(failures)} end anchor(s) do not hold against the "
            f"{state.atlas} annotation; nothing was submitted."
        ),
    }


def _missing_positions_error(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section has a position, else the rejection dict."""
    missing = [record.id for record in state.in_order() if record.position_mm is None]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_POSITIONS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} section(s) have no position. submit_positions "
            "requires a position for every section, damaged ones included."
        ),
    }


def check_interval_breaks(state: StackState, breaks: Any) -> dict[str, Any] | None:
    """Check reported interval breaks against the spacing the agent WROTE.

    A break at corrected index *i* claims the gap between section *i-1* and
    section *i* is larger than the rest of the stack's. That claim is checkable
    without an image: the positions on the state are the agent's own, and a
    "break" where its own numbers show ordinary spacing is a report of
    something that is not there — it travels downstream as a real finding.
    Refused, not warned, for the same reason the end anchors are.

    Returns ``None`` when every reported index holds (or when the stack is too
    short to have a median spacing), else the error dict ``submit_positions``
    hands back. Call it after :func:`_missing_positions_error`, so every
    section is known to carry a position.
    """
    indices: list[int] = []
    for raw in breaks if isinstance(breaks, (list, tuple)) else []:
        try:
            indices.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not indices:
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
    for index in sorted(set(indices)):
        position = next(
            (i for i, r in enumerate(ordered) if r.index_corrected == index), None
        )
        if position is None or position == 0:
            failures.append(
                {
                    "index": index,
                    "error": "NOT_A_GAP",
                    "reason": (
                        f"corrected index {index} has no section before it, so "
                        "there is no interval there; a break index names the "
                        "section AFTER the gap."
                    ),
                }
            )
            continue
        before, after = ordered[position - 1], ordered[position]
        written = abs(float(after.position_mm) - float(before.position_mm))  # type: ignore[arg-type]
        if written > threshold:
            continue
        failures.append(
            {
                "index": index,
                "error": "NOT_A_GAP",
                "written_interval_mm": round(written, 3),
                "median_interval_mm": round(median, 3),
                "between": [before.id, after.id],
                "reason": (
                    f"{before.id} and {after.id} are {written:.3f} mm apart in "
                    f"the positions you wrote; the stack's median written "
                    f"spacing is {median:.3f} mm."
                ),
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
            "positions you wrote; nothing was submitted. A break index is "
            f"accepted only where the written interval exceeds "
            f"{INTERVAL_BREAK_MIN_RATIO:g}x the stack's median written spacing."
        ),
    }


def _submission_errors(state: StackState, interval_breaks: Any) -> dict[str, Any] | None:
    """The checks both ``submit_positions`` variants run, in order."""
    return _missing_positions_error(state) or check_interval_breaks(
        state, interval_breaks
    )


# --- tools ---------------------------------------------------------------


def build_position_tools(
    state: StackState, ctx: EngineContext, *, pos_lo: float, pos_hi: float
) -> PositionToolBox:
    """Build the positioning tool set, closed over *state* (the working copy)."""
    box = PositionToolBox()
    landmark_tools = bool(ctx.config.landmark_tools)

    view_slices = make_view_slices(state, ctx)
    plane: Plane = state.plane  # type: ignore[assignment]

    def _atlas() -> Any:
        return ctx.atlas_loader(state.atlas or ctx.config.atlas)

    if landmark_tools:

        def atlas_structures_at(positions_mm: list[float]) -> dict[str, Any]:
            """What the atlas annotation carries at up to 8 levels along the slicing axis.

            Args:
                positions_mm: Positions along the slicing axis (max 8 per call).

            Returns:
                One entry per level: the structures present, largest in-plane area
                share first.
            """
            if not positions_mm:
                return {"status": "error", "error": "BAD_ARGS"}
            try:
                atlas = _atlas()
            except Exception as exc:
                logger.warning("position: atlas unavailable for structures: %s", exc)
                return {
                    "status": "error",
                    "error": "ATLAS_UNAVAILABLE",
                    "message": str(exc),
                }

            levels: list[dict[str, Any]] = []
            for raw in list(positions_mm)[:MAX_STRUCTURE_LEVELS]:
                try:
                    position = min(pos_hi, max(pos_lo, float(raw)))
                except (TypeError, ValueError):
                    levels.append({"requested": str(raw), "error": "NOT_A_NUMBER"})
                    continue
                try:
                    found = structures_at(atlas, position, plane)
                except Exception as exc:
                    levels.append(
                        {
                            "position_mm": round(position, 3),
                            "error": "LOOKUP_FAILED",
                            "message": str(exc),
                        }
                    )
                    continue
                levels.append(
                    {"position_mm": round(position, 3), "structures": found}
                )
            return {
                "status": "ok",
                "levels": levels,
                "note": (
                    "Structures the atlas annotation carries at each level, "
                    "ordered by share of the section's in-plane area."
                ),
            }

        def structure_range(acronyms: list[str]) -> dict[str, Any]:
            """Where along the slicing axis each named structure exists, at all.

            The span covers the structure and everything under it in the atlas
            hierarchy.

            Args:
                acronyms: Structure acronyms, or full structure names (max 10 per
                    call). Case-insensitive; a unique name substring also works.

            Returns:
                One entry per query: the resolved structure and its
                ``first_mm``/``last_mm`` span, or an error entry naming the
                closest structures the atlas does have.
            """
            if not acronyms:
                return {"status": "error", "error": "BAD_ARGS"}
            try:
                atlas = _atlas()
            except Exception as exc:
                logger.warning("position: atlas unavailable for ranges: %s", exc)
                return {
                    "status": "error",
                    "error": "ATLAS_UNAVAILABLE",
                    "message": str(exc),
                }

            ranges: list[dict[str, Any]] = []
            for raw in list(acronyms)[:MAX_STRUCTURE_QUERIES]:
                query = str(raw).strip()
                structure = find_structure(atlas, query) if query else None
                if structure is None:
                    ranges.append(
                        {
                            "query": query,
                            "error": "UNKNOWN_STRUCTURE",
                            "near_misses": near_misses(atlas, query) if query else [],
                        }
                    )
                    continue
                try:
                    span = axis_range_of(atlas, structure["acronym"], plane)
                except Exception as exc:
                    ranges.append(
                        {
                            "query": query,
                            "error": "LOOKUP_FAILED",
                            "message": str(exc),
                        }
                    )
                    continue
                if span is None:
                    ranges.append(
                        {
                            "query": query,
                            "acronym": structure["acronym"],
                            "name": structure["name"],
                            "error": "STRUCTURE_NOT_ANNOTATED",
                        }
                    )
                    continue
                entry: dict[str, Any] = {
                    "query": query,
                    "acronym": structure["acronym"],
                    "name": structure["name"],
                    "first_mm": round(span[0], 3),
                    "last_mm": round(span[1], 3),
                }
                # A short acronym resolves silently to ONE structure, which
                # may not be the one meant ("MED" is a thalamic nucleus in one
                # atlas and the start of "medulla" in a reader's head). Name
                # the other candidates so a wrong resolution is visible.
                others = [
                    acronym
                    for acronym in near_misses(atlas, query, limit=4)
                    if acronym != structure["acronym"]
                ][:3]
                if others:
                    entry["also_matches"] = others
                ranges.append(entry)
            return {
                "status": "ok",
                "ranges": ranges,
                "skipped": [str(a) for a in list(acronyms)[MAX_STRUCTURE_QUERIES:]],
                "note": (
                    "Spans include the structure's descendants, in atlas-native mm "
                    "along the slicing axis. `name` is the structure the query "
                    "resolved to; `also_matches` lists other structures the same "
                    "query matched."
                ),
            }

    def stack_positions() -> dict[str, Any]:
        """The stack's current positions and neighbour spacing, in corrected order.

        Returns:
            One row per section: corrected index, filename, ``position_mm``
            (null when the section has none) and ``spacing_to_next_mm``, the
            distance to the next placed section (null when none follows).
        """
        return {"status": "ok", "rows": position_rows(state)}

    def set_positions(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Write positions for one or more sections. Batch: one call, many sections.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back. Writing a
        section again overwrites its position.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>,
                "confidence": "low" | "medium" | "high"}]``. ``confidence`` is
                optional.

        Returns:
            Which sections were written, which values were clamped, unknown
            ids, and the same rows ``stack_positions`` returns, as they stand
            after the write.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        written: list[dict[str, Any]] = []
        clamped: list[dict[str, Any]] = []
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append({"entry": str(entry), "reason": "not an object"})
                continue
            record = state.by_id(str(entry.get("id", "")))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            try:
                requested = float(entry.get("position_mm"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rejected.append(
                    {"id": record.id, "reason": "position_mm is not a number"}
                )
                continue
            value = min(pos_hi, max(pos_lo, requested))
            if value != requested:
                clamped.append(
                    {
                        "id": record.id,
                        "requested_mm": round(requested, 3),
                        "clamped_to_mm": round(value, 3),
                    }
                )
            record.position_mm = value
            record.position_source = "refined"
            apply_confidence(record, entry.get("confidence"))
            written.append({"id": record.id, "position_mm": round(value, 3)})
            box.written.append(record.id)

        result: dict[str, Any] = {
            "status": "ok" if written else "error",
            "written": written,
            "unknown_ids": unknown,
            "rejected": rejected,
            "clamped": clamped,
            "rows": position_rows(state),
        }
        if clamped:
            result["warning"] = (
                f"{len(clamped)} position(s) fell outside the atlas range "
                f"{pos_lo:.2f}-{pos_hi:.2f} mm and were clamped."
            )
        if not written:
            result["error"] = "NOTHING_WRITTEN"
        return result

    def interpolate_between(fixed: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute positions for every section from two or more points you fix.

        Straight-line interpolation in corrected stack order: between two fixed
        points the spacing is spread evenly, and beyond the outermost fixed
        points it steps by the interval those points imply. At least two fixed
        points are required. Nothing is written.

        Args:
            fixed: The points to interpolate between, as
                ``[{"id": "<filename>", "position_mm": <number>}]``.

        Returns:
            One position per section, each marked ``fixed`` or
            ``interpolated``, plus the interval used beyond the outermost
            fixed points.
        """
        if not fixed:
            return {"status": "error", "error": "BAD_ARGS"}
        ordered = state.in_order()
        index_of = {record.id: index for index, record in enumerate(ordered)}
        known: list[float | None] = [None] * len(ordered)
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []
        for entry in fixed:
            if not isinstance(entry, dict):
                rejected.append({"entry": str(entry), "reason": "not an object"})
                continue
            slice_id = str(entry.get("id", ""))
            if slice_id not in index_of:
                unknown.append(slice_id)
                continue
            try:
                known[index_of[slice_id]] = float(entry.get("position_mm"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rejected.append({"id": slice_id, "reason": "position_mm is not a number"})
        anchor_indices = [i for i, value in enumerate(known) if value is not None]
        if not anchor_indices:
            return {
                "status": "error",
                "error": "NO_FIXED_POINTS",
                "unknown_ids": unknown,
                "rejected": rejected,
            }
        if len(anchor_indices) < 2:
            return {
                "status": "error",
                "error": "ONE_FIXED_POINT",
                "unknown_ids": unknown,
                "rejected": rejected,
                "message": (
                    "interpolate_between requires at least two fixed points; "
                    "one was given."
                ),
            }

        # Beyond the outermost fixed points, step by what the fixed points
        # imply, not by the nominal interval. Signed, so a stack cut
        # back-to-front extrapolates the way it actually runs.
        first, last = anchor_indices[0], anchor_indices[-1]
        step = (float(known[last]) - float(known[first])) / (last - first)  # type: ignore[arg-type]
        suggested = interpolate_positions(known, interval_mm=step)
        rows = [
            {
                "id": record.id,
                "position_mm": round(min(pos_hi, max(pos_lo, value)), 3),
                "source": "fixed" if known[index] is not None else "interpolated",
            }
            for index, (record, value) in enumerate(
                zip(ordered, suggested, strict=True)
            )
        ]
        return {
            "status": "ok",
            "suggestions": rows,
            "unknown_ids": unknown,
            "rejected": rejected,
            "implied_interval_mm": round(abs(step), 3),
        }

    if landmark_tools:

        def submit_positions(  # pyright: ignore[reportRedeclaration]
            interval_breaks: list[int],
            notes: list[str],
            summary: str,
            end_anchors: list[dict[str, Any]],
            tool_context: Any = None,
        ) -> dict[str, Any]:
            """Finish the positioning step. Call this exactly once, last.

            Rejected unless every section has a position, the reported
            interval breaks are present in those positions, and both end
            anchors hold against the atlas annotation.

            Args:
                interval_breaks: Corrected indices of the sections AFTER a gap
                    you conclude is real. Empty if there are none. An index is
                    accepted only where the interval between the positions you
                    wrote exceeds 1.5x the stack's median written spacing.
                notes: Short observations worth carrying forward.
                summary: One or two sentences on what you did.
                end_anchors: Exactly two entries, one for the FIRST and one
                    for the LAST section of the corrected order:
                    ``[{"id": "<filename>", "structure": "<acronym>", "note":
                    "<what you saw>"}]``. Each structure's atlas span must
                    contain the position written for that section (within one
                    slice thickness) and must itself cover no more than 8% of
                    the slicing axis.
            """
            refusal = _submission_errors(state, interval_breaks)
            if refusal is not None:
                return refusal

            anchor_note = ""
            if state.slices:
                try:
                    failure = check_end_anchors(
                        state, _atlas(), end_anchors, axis_extent_mm=pos_hi - pos_lo
                    )
                except Exception as exc:
                    # The atlas itself is unusable (no annotation, failed
                    # load). Refusing forever would deadlock the run, so let
                    # it through and say so in the record.
                    logger.warning("position: end-anchor check unavailable: %s", exc)
                    failure = None
                    anchor_note = f"end-anchor check skipped: {exc}"
                if failure is not None:
                    return failure

            # Model output is a trust boundary: a malformed submission must
            # not take the run down.
            submitted_notes = (
                [str(note) for note in notes]
                if isinstance(notes, (list, tuple))
                else []
            )
            if anchor_note:
                submitted_notes.append(anchor_note)
            box.submission.update(
                {
                    "interval_breaks": (
                        list(interval_breaks)
                        if isinstance(interval_breaks, (list, tuple))
                        else []
                    ),
                    "notes": submitted_notes,
                    "summary": str(summary or ""),
                    "end_anchors": [
                        {
                            "id": str(entry.get("id", "")),
                            "structure": str(entry.get("structure", "")),
                            "note": str(entry.get("note", "")),
                        }
                        for entry in end_anchors
                        if isinstance(entry, dict)
                    ],
                }
            )
            if tool_context is not None:
                tool_context.actions.escalate = True
            return {"status": "ok", "positioned": len(state.slices)}

    else:

        def submit_positions(
            interval_breaks: list[int],
            notes: list[str],
            summary: str,
            tool_context: Any = None,
        ) -> dict[str, Any]:
            """Finish the positioning step. Call this exactly once, last.

            Rejected unless every section has a position and the reported
            interval breaks are present in those positions.

            Args:
                interval_breaks: Corrected indices of the sections AFTER a gap
                    you conclude is real. Empty if there are none. An index is
                    accepted only where the interval between the positions you
                    wrote exceeds 1.5x the stack's median written spacing.
                notes: Short observations worth carrying forward.
                summary: One or two sentences on what you did.
            """
            refusal = _submission_errors(state, interval_breaks)
            if refusal is not None:
                return refusal

            # Model output is a trust boundary: a malformed submission must
            # not take the run down.
            box.submission.update(
                {
                    "interval_breaks": (
                        list(interval_breaks)
                        if isinstance(interval_breaks, (list, tuple))
                        else []
                    ),
                    "notes": (
                        [str(note) for note in notes]
                        if isinstance(notes, (list, tuple))
                        else []
                    ),
                    "summary": str(summary or ""),
                }
            )
            if tool_context is not None:
                tool_context.actions.escalate = True
            return {"status": "ok", "positioned": len(state.slices)}

    box.tools = [
        view_slices,
        fetch_atlas,
        *([atlas_structures_at, structure_range] if landmark_tools else []),
        stack_positions,
        interpolate_between,
        set_positions,
        submit_positions,
    ]
    return box


# --- prompt + seed message ----------------------------------------------


def build_position_prompt(
    *,
    state: StackState,
    species: str,
    pos_lo: float,
    pos_hi: float,
    landmark_tools: bool = True,
) -> str:
    """System instruction for the positioning agent, plane-aware.

    The job, the run's facts, the tools and the hard constraints — nothing
    else. No strategy, no rules of thumb, no warnings about failure modes:
    every benchmark failure worth tracing came back to advice the harness put
    in front of the model, so the model reasons and the prompt reports.

    Deliberately atlas-agnostic: it names no region, no landmark and no
    absolute position, because the same prompt runs against every BrainGlobe
    atlas, species and plane.

    ``landmark_tools=False`` (an ablation switch, see
    :attr:`~langslice.linear.whole_brain.state.BrainConfig.landmark_tools`)
    drops `atlas_structures_at`/`structure_range` and the end-anchor
    requirement.
    """
    plane = state.plane
    axis = _PLANE_AXIS_LABEL.get(plane, "AP")
    placed = [s for s in state.in_order() if s.position_mm is not None]
    placed_line = (
        f"- {len(placed)} of {len(state.slices)} sections already carry a "
        f"position.\n"
        if placed
        else "- No section carries a position yet.\n"
    )
    breaks_line = (
        f"- The survey step flagged possible interval breaks at corrected "
        f"indices {state.interval_breaks}.\n"
        if state.interval_breaks
        else ""
    )
    damaged = [s.id for s in state.in_order() if s.damaged]
    damaged_line = (
        f"- Sections marked damaged: {', '.join(damaged)}.\n" if damaged else ""
    )
    order_line = (
        "- The corrected order shown is fixed; it was not open to "
        "reordering.\n"
        if state.keep_order
        else "- The corrected order shown is the survey step's, which was free "
        "to reorder the stack.\n"
    )

    landmark_tool_lines = (
        "- `atlas_structures_at`: the structures the atlas annotation carries "
        "at up to 8 levels you name, by share of in-plane area.\n"
        "- `structure_range`: the slicing-axis span over which each of up to "
        "10 named structures exists, descendants included.\n"
        if landmark_tools
        else ""
    )
    anchor_constraint = (
        "- `submit_positions` requires one end anchor for the FIRST and one "
        "for the LAST section of the corrected order: a structure you can see "
        "in that section, whose span in the atlas annotation contains the "
        "position you wrote for that section (within one slice thickness) and "
        f"itself covers no more than {MAX_ANCHOR_SPAN_FRACTION * 100:.0f}% of "
        "the slicing axis. Both are checked against the annotation volume.\n"
        if landmark_tools
        else ""
    )

    return (
        f"You are an expert neuroanatomist placing a stack of "
        f"{len(state.slices)} {plane} histology sections along the "
        f"{state.atlas} ({species}) atlas.\n\n"
        f"Your job: give every section a position in millimetres along the "
        f"slicing axis, and report the corrected indices where you conclude "
        f"the interval between neighbouring sections is genuinely broken. "
        f"Damaged sections get a position too.\n\n"
        f"Run facts:\n"
        f"- {len(state.slices)} sections, {plane} plane, atlas "
        f"{state.atlas} ({species}).\n"
        f"- Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm along the slicing "
        f"axis, measured from the origin edge of the atlas volume "
        f"({pos_lo:.2f} mm is its first section, {pos_hi:.2f} mm its last).\n"
        f"- Cutting protocol: nominal section interval "
        f"{state.interval_mm:.3f} mm center-to-center, section thickness "
        f"{state.thickness_mm:.3f} mm.\n"
        f"{order_line}"
        f"{placed_line}"
        f"{breaks_line}"
        f"{damaged_line}"
        f"\n"
        f"Tools:\n"
        f"- `view_slices`: up to 8 named sections at higher resolution.\n"
        f"- `fetch_atlas`: atlas sections at the positions you name.\n"
        f"{landmark_tool_lines}"
        f"- `stack_positions`: the positions currently written and the "
        f"spacing between them.\n"
        f"- `interpolate_between`: straight-line positions for every section "
        f"from two or more points you fix; writes nothing.\n"
        f"- `set_positions`: writes positions for one or more sections.\n"
        f"- `submit_positions`: ends the step.\n\n"
        f"Constraints:\n"
        f"- Every section must have a position before `submit_positions` is "
        f"accepted, damaged sections included.\n"
        f"{anchor_constraint}"
        f"- A reported interval break is accepted only at an index where the "
        f"interval between the positions you wrote exceeds "
        f"{INTERVAL_BREAK_MIN_RATIO:g}x the stack's median written spacing.\n\n"
        f"Work with the tools, then call `submit_positions`."
    )


def build_position_seed_message(
    state: StackState, ctx: EngineContext, *, note_limit: int = 12
) -> types.Content:
    """Per-section images + stack manifest: what the positioning step starts from.

    Sections with no position yet read as "unplaced" in the manifest — the
    normal case, since nothing upstream places them.

    The recent run notes ride along because this step is also the loop-back
    target: when the review step refuses a stack it writes what is wrong into
    the notes, and this is where the next pass reads it.
    """
    recent = state.notes[-note_limit:]
    notes_block = "\n".join(f"- {note}" for note in recent) if recent else "- (none)"
    parts: list[types.Part] = stack_image_parts(state, ctx)
    parts.append(
        types.Part.from_text(
            text=(
                "Stack manifest (corrected index, filename, current position "
                "and where it came from — 'unplaced' means no position yet — "
                "and current flags):\n"
                f"{build_stack_manifest(state, with_positions=True)}\n\n"
                f"Run notes so far, oldest first (a 'review:' note was written "
                f"by a review pass that sent this stack back):\n"
                f"{notes_block}\n\n"
                "Place the stack with the tools, write positions with "
                "`set_positions`, then call `submit_positions`."
            )
        )
    )
    return types.Content(role="user", parts=parts)


# --- agent + session driver ---------------------------------------------


def build_position_agent(
    *,
    state: StackState,
    tools: list[Any],
    species: str,
    pos_lo: float,
    pos_hi: float,
    model: str | object = DEFAULT_POSITION_MODEL,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    landmark_tools: bool = True,
) -> LlmAgent:
    """Construct the positioning LlmAgent."""
    # Same shape as build_survey_agent: kwargs dict so the enum-typed
    # media_resolution string is accepted as-is.
    config_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        # A whole-stack set_positions call is a long argument list; leave room
        # for it plus the model's reasoning.
        "max_output_tokens": 8000,
        "media_resolution": media_resolution,
        "http_options": default_http_options(),
    }
    return LlmAgent(
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name="whole_brain_position",
        instruction=build_position_prompt(
            state=state,
            species=species,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            landmark_tools=landmark_tools,
        ),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )


async def run_position_session(
    *,
    state: StackState,
    ctx: EngineContext,
    species: str,
    pos_lo: float,
    pos_hi: float,
    max_iterations: int = DEFAULT_POSITION_MAX_ITERATIONS,
) -> PositionOutcome:
    """Drive one positioning pass and return what it produced.

    The tools mutate *state* as they are called, so accepted positions survive
    even when the agent never reaches ``submit_positions``.
    """
    box = build_position_tools(state, ctx, pos_lo=pos_lo, pos_hi=pos_hi)
    agent = build_position_agent(
        state=state,
        tools=box.tools,
        species=species,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        model=ctx.model or DEFAULT_POSITION_MODEL,
        landmark_tools=bool(ctx.config.landmark_tools),
    )

    tool_calls, turns = await run_agent_session(
        agent=agent,
        state=state,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        seed_message=build_position_seed_message(state, ctx),
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
    )

    return PositionOutcome(
        findings=dict(box.submission) if box.submission else None,
        positions_written=len(set(box.written)),
        tool_calls=tool_calls,
        turns=turns,
    )
