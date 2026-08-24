"""The positioning agent: place the whole stack, with the whole stack in view.

This step owns placement, strategy included. The stack usually arrives
unplaced: nothing upstream picks key sections, because picking good ones needs
intimate atlas knowledge and a badly chosen key section poisons every position
interpolated from it. Instead the agent gets the full toolkit — look at
sections, fetch atlas levels, estimate named sections with the single-slice
worker, interpolate between points it trusts, read the spacing arithmetic —
plus a menu of strategies in its prompt, and decides for itself.

Everything the agent writes goes through ``set_positions`` onto the
:class:`~langslice.linear.whole_brain.state.StackState` the node handed in, so
a pass that runs out of turns still leaves its accepted positions behind.
``estimate_slices`` and ``interpolate_between`` deliberately do NOT write: they
report, and the agent decides what to keep.
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
    split_known_ids,
    stack_image_parts,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.estimation_agents import run_slice_estimation
from langslice.linear.whole_brain.signals import interpolate_positions, monotone_fit
from langslice.linear.whole_brain.state import (
    SliceState,
    StackState,
    apply_confidence,
)
from langslice.space import Plane

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one positioning pass may spend. This
#: step now owns strategy as well as placement, so the budget is generous.
DEFAULT_POSITION_MAX_ITERATIONS = 40

#: Sections one ``estimate_slices`` call may estimate. Each one is a full
#: single-slice agent session, so the cap is about cost, not correctness.
MAX_ESTIMATE_SLICES = 8

#: Levels one ``atlas_structures_at`` call may report on.
MAX_STRUCTURE_LEVELS = 8

#: Structures one ``structure_range`` call may look up.
MAX_STRUCTURE_QUERIES = 10

DEFAULT_POSITION_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_position"

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: compare sections against "
    "the atlas with `view_slices` and `fetch_atlas`, check what the atlas says "
    "is there with `atlas_structures_at` and `structure_range`, estimate "
    "sections with `estimate_slices`, write positions with `set_positions`, "
    "and finish with `submit_positions`."
)
_NUDGE_CONTINUE = (
    "Please continue placing the stack. Check anything still unresolved, write "
    "the positions you have settled on with `set_positions`, verify both ends "
    "against the atlas with `structure_range`, then call `submit_positions` "
    "with an end anchor for the first and last section."
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
    estimated: int = 0
    tool_calls: int = 0
    turns: int = 0


@dataclass
class PositionToolBox:
    """Tool callables plus the mutable results the node reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    estimated: list[str] = field(default_factory=list)


# --- spacing views -------------------------------------------------------


#: Legend carried with every interval table. The nominal interval is the one
#: number in the run that is NOT evidence, and a stack placed to match it
#: exactly is the failure this line is here to name.
INTERVAL_LEGEND = (
    "nominal_interval_mm is the cutting protocol, not a measurement; "
    "implied_interval_mm is what your own placed positions say the spacing "
    "actually is. The realized mean is usually LARGER than nominal, because "
    "sections get lost, torn or skipped during collection. A ladder that "
    "matches the nominal interval exactly while your verified anchors imply "
    "larger spacing means the stack is COMPRESSED: the ends are wrong."
)


def _placed_runs(ordered: list[SliceState]) -> list[list[SliceState]]:
    """Contiguous stretches of positioned sections, in corrected order."""
    runs: list[list[SliceState]] = []
    current: list[SliceState] = []
    for record in ordered:
        if record.position_mm is None:
            if len(current) > 1:
                runs.append(current)
            current = []
            continue
        current.append(record)
    if len(current) > 1:
        runs.append(current)
    return runs


def interval_table(state: StackState) -> dict[str, Any]:
    """Neighbour spacing in corrected order, implied interval beside the nominal.

    ``rows`` is one entry per section (index, id, position, source, delta to
    the next). Around them sit the two numbers that decide absolute placement:
    the NOMINAL interval the cutting protocol claims, and the interval the
    placed sections actually IMPLY — over each contiguous placed stretch and
    over the stack as a whole.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(ordered):
        following = ordered[position + 1] if position + 1 < len(ordered) else None
        delta = (
            round(following.position_mm - record.position_mm, 3)
            if following is not None
            and following.position_mm is not None
            and record.position_mm is not None
            else None
        )
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": (
                    round(record.position_mm, 3)
                    if record.position_mm is not None
                    else None
                ),
                "source": record.position_source,
                "delta_to_next_mm": delta,
            }
        )

    stretches: list[dict[str, Any]] = []
    total_span = 0.0
    total_gaps = 0
    for run in _placed_runs(ordered):
        span = abs(float(run[-1].position_mm) - float(run[0].position_mm))  # type: ignore[arg-type]
        gaps = len(run) - 1
        total_span += span
        total_gaps += gaps
        stretches.append(
            {
                "from_index": run[0].index_corrected,
                "to_index": run[-1].index_corrected,
                "sections": len(run),
                "implied_interval_mm": round(span / gaps, 3),
            }
        )

    return {
        "rows": rows,
        "nominal_interval_mm": round(state.interval_mm, 3),
        "implied_interval_mm": (
            round(total_span / total_gaps, 3) if total_gaps else None
        ),
        "implied_by_stretch": stretches,
        "legend": INTERVAL_LEGEND,
    }


def _positioned(ordered: list[SliceState]) -> list[SliceState]:
    return [record for record in ordered if record.position_mm is not None]


def spacing_advisories(state: StackState) -> dict[str, Any]:
    """The two arithmetic spacing views of the stack, as one dict.

    Shared with the review step: both show the agent the same numbers, and
    both label them advisory — nothing here has looked at an image.
    """
    return {
        "status": "ok",
        "advisory": (
            "Suggestions computed from the current numbers only. Nothing here "
            "has looked at a section image; verify before acting."
        ),
        "interval_table": interval_table(state),
        "monotone_fit": _monotone_suggestion(state),
    }


def _monotone_suggestion(state: StackState) -> dict[str, Any]:
    """A monotone, minimum-spacing curve through the current positions."""
    ordered = _positioned(state.in_order())
    if len(ordered) < 3:
        return {"status": "unavailable", "reason": "fewer than three positioned sections"}
    positions = [float(record.position_mm) for record in ordered]  # type: ignore[arg-type]
    # monotone_fit fits an increasing curve; a stack cut back-to-front is
    # fitted on the negated positions and flipped back.
    sign = 1.0 if positions[-1] >= positions[0] else -1.0
    try:
        fitted = monotone_fit(
            [sign * value for value in positions],
            interval_mm=state.interval_mm,
            thickness_mm=state.thickness_mm,
        )
    except Exception as exc:  # a bad fit must not take the session down
        logger.warning("position: monotone fit failed: %s", exc)
        return {"status": "unavailable", "reason": str(exc)}
    rows = [
        {
            "id": record.id,
            "position_mm": round(value, 3),
            "suggested_mm": round(sign * fit, 3),
            "delta_mm": round(sign * fit - value, 3),
        }
        for record, value, fit in zip(ordered, positions, fitted, strict=True)
    ]
    return {"status": "ok", "rows": rows}


# --- the end-anchor gate -------------------------------------------------


def check_end_anchors(
    state: StackState, atlas: Any, anchors: Any
) -> dict[str, Any] | None:
    """Check the two end anchors against the atlas annotation.

    Returns ``None`` when both hold, or the error dict ``submit_positions``
    should hand back. This is the one check in the whole positioning step that
    is not the agent marking its own homework: the structure the agent names
    has an existence range in the annotation volume, and a section placed
    outside that range is placed wrong.
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
                "submit_positions needs one end anchor for each END of the "
                f"corrected order — {required[0]} and {required[-1]} — as "
                '{"id": ..., "structure": "<acronym>", "note": "<what you '
                'saw>"}. Name a structure you can SEE in that section, and '
                "check with `structure_range` that the atlas agrees it exists "
                "where you have placed the section."
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
                    "reason": (
                        f"no structure in {state.atlas} matches {query!r}; name "
                        "one by its acronym or its full name."
                    ),
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
                        f"voxels in the {state.atlas} annotation, so it cannot "
                        "anchor anything. Pick a structure the atlas draws."
                    ),
                }
            )
            continue
        lo, hi = span
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
                        f"{hi:.3f} mm along the slicing axis, but you placed "
                        f"{slice_id} at {position:.3f} mm — outside that span "
                        f"(tolerance {tolerance:.3f} mm). Either the placement "
                        "is wrong or the structure is not what you saw."
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
            f"The atlas contradicts {len(failures)} of your end anchors, so "
            "nothing was submitted. A placement that a structure's existence "
            "range rules out is wrong, full stop: move the section into the "
            "structure's span, or name a structure that is really there. Do "
            "not resubmit the same numbers with a different structure unless "
            "you have looked again."
        ),
    }


# --- tools ---------------------------------------------------------------


def build_position_tools(
    state: StackState, ctx: EngineContext, *, pos_lo: float, pos_hi: float
) -> PositionToolBox:
    """Build the positioning tool set, closed over *state* (the working copy)."""
    box = PositionToolBox()

    view_slices = make_view_slices(state, ctx)
    plane: Plane = state.plane  # type: ignore[assignment]

    def _atlas() -> Any:
        return ctx.atlas_loader(state.atlas or ctx.config.atlas)

    def atlas_structures_at(positions_mm: list[float]) -> dict[str, Any]:
        """What the ATLAS says is present at up to 8 levels along the slicing axis.

        Read straight out of the atlas annotation volume — not a model opinion
        and not arithmetic over your own writes. Use it two ways: to check what
        anatomy should be there at a level you are considering, and to find
        where a structure you can SEE in the tissue can and cannot be.

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
                "Structures the atlas annotation carries at each level, by "
                "share of the section's tissue area. Anything you SEE that "
                "this list rules out means the level is wrong."
            ),
        }

    def structure_range(acronyms: list[str]) -> dict[str, Any]:
        """Where along the slicing axis each named structure exists, at all.

        The span covers the structure and everything under it in the atlas
        hierarchy. A section placed outside a span cannot contain that
        structure — this is how you check an absolute placement against
        something other than your own spacing arithmetic.

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
            ranges.append(
                {
                    "query": query,
                    "acronym": structure["acronym"],
                    "name": structure["name"],
                    "first_mm": round(span[0], 3),
                    "last_mm": round(span[1], 3),
                }
            )
        return {
            "status": "ok",
            "ranges": ranges,
            "skipped": [str(a) for a in list(acronyms)[MAX_STRUCTURE_QUERIES:]],
            "note": (
                "Spans include the structure's descendants, in atlas-native mm "
                "along the slicing axis."
            ),
        }

    def set_positions(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Write positions for one or more sections. Batch: one call, many sections.

        Positions are in atlas-native millimetres along the slicing axis.
        Anything outside the atlas range is clamped and reported back as a
        warning. Calling this again for the same section overwrites it.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>,
                "confidence": "low" | "medium" | "high"}]``. ``confidence`` is
                optional.

        Returns:
            Which sections were written, which values were clamped, unknown
            ids, and the resulting neighbour-interval table so you can see the
            spacing your edit produced.
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
            "interval_table": interval_table(state),
        }
        if clamped:
            result["warning"] = (
                f"{len(clamped)} position(s) fell outside the atlas range "
                f"{pos_lo:.2f}-{pos_hi:.2f} mm and were clamped."
            )
        if not written:
            result["error"] = "NOTHING_WRITTEN"
        return result

    async def estimate_slices(slice_ids: list[str]) -> dict[str, Any]:
        """Estimate named sections independently against the whole atlas.

        Each section gets its own full atlas sweep by an independent estimator
        that sees only that section — no stack context, no neighbours. This is
        how you place key sections, or resolve a section you cannot place by
        eye. It is the slow, expensive tool: one sweep per section, run one
        after another, up to 8 sections per call.

        Nothing is written. Judge the numbers against each other and against
        the stack — independent estimates disagree, and the disagreement is
        information — then write what you accept with `set_positions`.

        Args:
            slice_ids: Filenames from the stack manifest (max 8 per call).

        Returns:
            One entry per section: its estimated position and the estimator's
            reasoning, or an error for that section if its estimate failed.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        requested = list(slice_ids)
        wanted, unknown = split_known_ids(state, requested[:MAX_ESTIMATE_SLICES])
        if not wanted:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}

        estimates: list[dict[str, Any]] = []
        for slice_id in wanted:
            record = state.by_id(slice_id)
            assert record is not None
            current = (
                round(record.position_mm, 3) if record.position_mm is not None else None
            )
            try:
                result = await run_slice_estimation(
                    image_path=ctx.image_path(record.id),
                    atlas_name=state.atlas,
                    plane=state.plane,
                    model_name=ctx.model,
                    apply_clahe=ctx.config.preprocess == "auto",
                )
            except Exception as exc:
                logger.warning("position: estimate failed for %s: %s", record.id, exc)
                estimates.append(
                    {
                        "id": record.id,
                        "status": "error",
                        "error": "ESTIMATE_FAILED",
                        "message": str(exc),
                    }
                )
                continue
            box.estimated.append(record.id)
            estimates.append(
                {
                    "id": record.id,
                    "status": "ok",
                    "position_mm": round(float(result.position_mm), 3),
                    "reasoning": result.reasoning,
                    "current_position_mm": current,
                }
            )

        return {
            "status": "ok" if any(e["status"] == "ok" for e in estimates) else "error",
            "estimates": estimates,
            "unknown_ids": unknown,
            "skipped_ids": requested[MAX_ESTIMATE_SLICES:],
            "note": (
                "Nothing was written. Call `set_positions` for the estimates "
                "you accept."
            ),
        }

    def interpolate_between(fixed: list[dict[str, Any]]) -> dict[str, Any]:
        """Suggest positions for every other section from points you fix.

        Straight-line interpolation in corrected stack order: between two fixed
        points the spacing is spread evenly, and beyond the outermost ones it
        steps by the interval those fixed points IMPLY — never by the nominal
        interval, which would quietly compress the ends of the stack back onto
        the cutting protocol. Two fixed points are the minimum: one cannot
        place a stack, only offset it.

        It is arithmetic — it has never looked at an image and knows nothing
        about sections lost during collection, so a stretch where the anatomy
        advances faster than the suggestion says is exactly where you should
        look.

        Nothing is written. Write what you accept with `set_positions`.

        Args:
            fixed: The points to interpolate between, as
                ``[{"id": "<filename>", "position_mm": <number>}]``. At least
                two, ideally one near each end of the stack.

        Returns:
            One suggested position per section, each marked as fixed by you or
            interpolated, plus the implied interval used beyond the ends.
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
                    "one fixed point cannot place the stack; fix a second "
                    "point near the other end. With a single point every "
                    "section outside it would be stepped at the nominal "
                    "interval, which is the cutting protocol, not a "
                    "measurement — that is how a stack ends up compressed."
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
            "nominal_interval_mm": round(state.interval_mm, 3),
            "note": (
                "Suggestions only, nothing written, no image looked at. "
                f"Sections beyond your outermost fixed points were stepped at "
                f"{abs(step):.3f} mm, the interval your fixed points imply "
                f"(nominal is {state.interval_mm:.3f} mm). Check the stretches "
                "between your fixed points before accepting them."
            ),
        }

    def get_advisories() -> dict[str, Any]:
        """Computed spacing signals for the stack as it currently stands.

        ADVISORY ONLY — these are arithmetic, not anatomy. They cannot see the
        images and know nothing about missing sections. Use them to spot where
        the stack disagrees with itself, then check those sections yourself.

        Returns:
            The neighbour-interval table and a monotone minimum-spacing curve
            fitted through the positions the stack currently carries.
        """
        return spacing_advisories(state)

    def submit_positions(
        interval_breaks: list[int],
        notes: list[str],
        summary: str,
        end_anchors: list[dict[str, Any]],
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish the positioning step. Call this exactly once, last.

        Every section must have a position first, damaged sections included,
        and both ends of the stack must be anchored to real anatomy — the call
        is rejected otherwise.

        Args:
            interval_breaks: Corrected indices where the spacing between
                neighbouring sections breaks the nominal interval — the index
                of the section AFTER the gap. Empty if the stack is regular.
            notes: Short observations worth carrying forward.
            summary: One or two sentences on what you changed and why.
            end_anchors: Exactly two entries, one for the FIRST and one for the
                LAST section of the corrected order:
                ``[{"id": "<filename>", "structure": "<acronym>", "note":
                "<what you saw>"}]``. The structure must be one you can SEE in
                that section, and the atlas must agree it exists where you have
                placed it — checked here against the annotation volume, so a
                placement that contradicts the structure's range is refused.
        """
        missing = [
            record.id for record in state.in_order() if record.position_mm is None
        ]
        if missing:
            return {
                "status": "error",
                "error": "MISSING_POSITIONS",
                "missing_ids": missing,
                "message": (
                    f"{len(missing)} section(s) still have no position. Every "
                    "section needs one, damaged sections included: write them "
                    "with `set_positions`, then submit again."
                ),
            }

        anchor_note = ""
        if state.slices:
            try:
                failure = check_end_anchors(state, _atlas(), end_anchors)
            except Exception as exc:
                # The atlas itself is unusable (no annotation, failed load).
                # Refusing forever would deadlock the run, so let it through
                # and say so in the record.
                logger.warning("position: end-anchor check unavailable: %s", exc)
                failure = None
                anchor_note = f"end-anchor check skipped: {exc}"
            if failure is not None:
                return failure

        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
        submitted_notes = (
            [str(note) for note in notes] if isinstance(notes, (list, tuple)) else []
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

    box.tools = [
        view_slices,
        fetch_atlas,
        atlas_structures_at,
        structure_range,
        estimate_slices,
        interpolate_between,
        set_positions,
        get_advisories,
        submit_positions,
    ]
    return box


# --- prompt + seed message ----------------------------------------------


def build_position_prompt(
    *, state: StackState, species: str, pos_lo: float, pos_hi: float
) -> str:
    """System instruction for the positioning agent, plane-aware.

    Deliberately atlas-agnostic: it names no region, no landmark and no
    absolute position, because the same prompt runs against every BrainGlobe
    atlas, species and plane. What it does carry is a MENU of strategies and
    the failure modes that bite whichever one the agent picks.
    """
    plane = state.plane
    axis = _PLANE_AXIS_LABEL.get(plane, "AP")
    placed = [s for s in state.in_order() if s.position_mm is not None]
    placed_line = (
        f"- {len(placed)} of {len(state.slices)} sections already carry a "
        f"position; check them rather than trusting them.\n"
        if placed
        else "- No section has a position yet: the stack is unplaced and "
        "placing it is your job.\n"
    )
    breaks_line = (
        f"- The survey step flagged possible interval breaks at corrected "
        f"indices {state.interval_breaks}.\n"
        if state.interval_breaks
        else ""
    )
    damaged = [s.id for s in state.in_order() if s.damaged]
    damaged_line = (
        f"- Damaged sections ({', '.join(damaged)}) still need positions — "
        "place them from their neighbours if their own anatomy is "
        "unreadable.\n"
        if damaged
        else ""
    )

    return (
        f"You are an expert neuroanatomist placing a stack of "
        f"{len(state.slices)} {plane} histology sections along the "
        f"{state.atlas} ({species}) atlas.\n\n"
        f"Stack facts:\n"
        f"- Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm along the slicing "
        f"axis, measured from the origin edge of the atlas volume "
        f"({pos_lo:.2f} mm is its first section, {pos_hi:.2f} mm its last)\n"
        f"- Nominal section interval from the cutting protocol: "
        f"{state.interval_mm:.3f} mm center-to-center — expect the REALIZED "
        f"mean spacing to be >= this, because sections get lost. It is a "
        f"starting guess, not a measurement, and nothing in the stack has to "
        f"match it.\n"
        f"- Slice thickness: {state.thickness_mm:.3f} mm\n"
        f"{placed_line}"
        f"{breaks_line}"
        f"{damaged_line}"
        f"\n"
        f"YOU CHOOSE THE STRATEGY. Nothing upstream picked sections for you and "
        f"no strategy is prescribed here. Look at the stack, decide how to "
        f"place it, and say which approach you took in your submission.\n\n"
        f"STRATEGIES THAT WORK — pick one or mix them:\n\n"
        f"A. KEY SECTIONS, THEN INTERPOLATE. Choose a few sections whose "
        f"anatomy is distinctive and unambiguous, estimate them with "
        f"`estimate_slices`, VERIFY each one yourself against the atlas, then "
        f"fill the rest with `interpolate_between` and investigate the "
        f"stretches in between. Fast, and only as good as the key sections: a "
        f"key section placed wrong drags every section interpolated from it.\n\n"
        f"B. FULL COVERAGE. For a small stack, estimate every section with "
        f"`estimate_slices` (up to 8 per call) and reconcile the results "
        f"against each other and against the slicing interval. Slower, but no "
        f"section inherits another's error.\n\n"
        f"C. A MIX. Place a few sections, interpolate, then estimate more "
        f"wherever the result looks weak — around a suspected break, at the "
        f"ends, or anywhere the anatomy stops matching.\n\n"
        f"RULES THAT APPLY WHATEVER YOU CHOOSE:\n\n"
        f"1. PLACE KEY SECTIONS WHERE THE FEATURES ARE UNAMBIGUOUS. A good key "
        f"section is one whose atlas level is identifiable at a glance: some "
        f"structure appears, disappears, or changes shape sharply within a "
        f"short span of the slicing axis. Do NOT anchor on a section that sits "
        f"inside a long span of levels that look alike — that span is where "
        f"you interpolate through, not where you take your bearings.\n\n"
        f"2. DISAGREEING ESTIMATES ARE A STOP SIGN, NOT AN AVERAGE. Estimates "
        f"are made independently, one section at a time, so they can disagree "
        f"about where the stack sits as a whole. When they do, RE-VERIFY the "
        f"disagreeing sections before you commit to any global placement: "
        f"`fetch_atlas` around EACH candidate position, `view_slices` the "
        f"sections around each one, and check that the neighbours make sense "
        f"at that placement too. Then discard the estimate that does not hold "
        f"up. Never resolve the conflict by sliding a self-consistent set of "
        f"sections to match a minority reading, and never split the "
        f"difference. BUT: when MANY independent estimates disagree with your "
        f"tidy ladder by a CONSISTENT amount and in the same direction, it is "
        f"the ladder's absolute placement that is suspect, not the estimates. "
        f"That pattern is what a compressed or offset stack looks like from "
        f"the inside — check it with `atlas_structures_at` and "
        f"`structure_range` before discarding a single estimate.\n\n"
        f"3. CHECK BOTH ENDS BEFORE YOU SUBMIT — AGAINST THE ATLAS, NOT "
        f"AGAINST YOUR OWN ARITHMETIC. A ladder with plausible spacing hung at "
        f"the wrong absolute position is perfectly consistent from the inside: "
        f"every interval looks right and every section still sits wrong. The "
        f"only way out of that loop is evidence from outside it. For the FIRST "
        f"and the LAST section of the corrected order: NAME a structure you "
        f"can actually SEE in that section, call `structure_range` on it, and "
        f"check that the position you have written falls inside the span the "
        f"atlas reports. `atlas_structures_at` goes the other way — it tells "
        f"you what the atlas says is there at a level you are considering. A "
        f"placement that contradicts a structure's existence range is WRONG, "
        f"full stop; move the section or pick a landmark you can defend, never "
        f"a structure you have not looked for. `submit_positions` asks for "
        f"both end anchors and re-checks them, and refuses the submission when "
        f"they do not hold.\n\n"
        f"4. THE SPACING IS NOT THE NOMINAL INTERVAL, IN EITHER DIRECTION. A "
        f"consistent slicing interval does NOT mean no sections are missing: "
        f"sections get lost, torn or skipped during collection, and the "
        f"sections on either side of the loss still look evenly spaced on the "
        f"slide. So the realized spacing is usually LARGER than the nominal "
        f"{state.interval_mm:.3f} mm, occasionally smaller where the protocol "
        f"drifted, and a ladder that matches the nominal interval EXACTLY end "
        f"to end is a warning sign, not a success — it usually means the stack "
        f"has been compressed onto the protocol and its ends pulled inward. "
        f"The interval table shows the implied interval next to the nominal "
        f"one: when your anchors imply larger spacing than your ladder uses, "
        f"believe the anchors. Where the anatomy advances faster than the "
        f"suggestion says, sections are missing and any interpolation across "
        f"that stretch is wrong — investigate it, reposition it, and report "
        f"every break you confirm in `submit_positions`.\n\n"
        f"5. WRITE IN BATCHES. `set_positions` takes many sections in one call "
        f"and hands back the resulting neighbour-interval table, so you see "
        f"immediately what your edit did to the spacing.\n\n"
        f"6. THE ARITHMETIC IS ADVICE; THE ATLAS IS EVIDENCE. "
        f"`interpolate_between` and `get_advisories` compute from numbers that "
        f"have never seen an image and know nothing about missing sections. "
        f"Use them to find suspicious stretches, never as the answer. "
        f"`atlas_structures_at` and `structure_range` are the other kind of "
        f"tool: they read the atlas annotation itself, so they can contradict "
        f"you. Let them.\n\n"
        f"Finish with `submit_positions`. It is rejected unless every section "
        f"in the stack has a position (damaged sections included) AND both end "
        f"anchors hold up against the atlas."
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
                f"Run notes so far (a 'review:' note means an earlier pass "
                f"sent this stack back — read it first):\n{notes_block}\n\n"
                "Decide how you want to place this stack, work through it with "
                "the tools, write positions with `set_positions`, verify the "
                "first and last section against the atlas with "
                "`structure_range`, then call `submit_positions` with an end "
                "anchor for each of them."
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
            state=state, species=species, pos_lo=pos_lo, pos_hi=pos_hi
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
        estimated=len(box.estimated),
        tool_calls=tool_calls,
        turns=turns,
    )
