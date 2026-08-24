"""The positioning agent: refine the seeded positions with the whole stack in view.

The seed step put a globally consistent but coarse set of positions on the
stack — a few anchor estimates with everything else interpolated between them.
This step is the expert's "reposition slices": one agent, whole stack in
context, verifying the anchors against the atlas and hunting for BREAKS in the
interval, because a constant slicing interval does not mean no sections went
missing.

Everything the agent writes goes through ``set_positions`` onto the
:class:`~langslice.linear.whole_brain.state.StackState` the node handed in, so
a pass that runs out of turns still leaves its accepted positions behind.
``estimate_slice`` is the escalation hatch — a full single-slice sweep for one
stubborn section — and deliberately does NOT write: the agent looks at the
number and decides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.adk.model_resolver import default_http_options, resolve_adk_model
from langslice.linear.tools import fetch_atlas
from langslice.linear.whole_brain._step_common import (
    build_stack_manifest,
    contact_sheet_parts,
    make_view_slices,
    run_agent_session,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.estimation_agents import run_slice_estimation
from langslice.linear.whole_brain.signals import interpolate_positions, monotone_fit
from langslice.linear.whole_brain.state import (
    CONFIDENCE_LEVELS,
    SliceState,
    StackState,
)

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one positioning pass may spend.
DEFAULT_POSITION_MAX_ITERATIONS = 25

DEFAULT_POSITION_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_position"

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: compare sections against "
    "the atlas with `view_slices` and `fetch_atlas`, write positions with "
    "`set_positions`, and finish with `submit_positions`."
)
_NUDGE_CONTINUE = (
    "Please continue repositioning. Check anything still unresolved, write the "
    "positions you have settled on with `set_positions`, then call "
    "`submit_positions`."
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
    escalations: int = 0
    tool_calls: int = 0
    turns: int = 0


@dataclass
class PositionToolBox:
    """Tool callables plus the mutable results the node reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)


# --- spacing views -------------------------------------------------------


def interval_table(state: StackState) -> list[dict[str, Any]]:
    """Neighbour spacing in corrected order: index, id, position, delta to next."""
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
    return rows


def _positioned(ordered: list[SliceState]) -> list[SliceState]:
    return [record for record in ordered if record.position_mm is not None]


def _interpolation_residuals(state: StackState) -> dict[str, Any]:
    """Per-section gap between the current position and the anchor interpolation."""
    ordered = state.in_order()
    known: list[float | None] = [
        record.position_mm if record.position_source == "anchor" else None
        for record in ordered
    ]
    anchors = [value for value in known if value is not None]
    if len(anchors) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer than two anchor sections to interpolate between",
        }
    step = state.interval_mm if anchors[-1] >= anchors[0] else -state.interval_mm
    expected = interpolate_positions(known, interval_mm=step)
    rows = [
        {
            "id": record.id,
            "position_mm": round(record.position_mm, 3),
            "interpolated_mm": round(value, 3),
            "residual_mm": round(record.position_mm - value, 3),
        }
        for record, value in zip(ordered, expected, strict=True)
        if record.position_mm is not None
    ]
    return {"status": "ok", "rows": rows}


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


# --- tools ---------------------------------------------------------------


def build_position_tools(
    state: StackState, ctx: EngineContext, *, pos_lo: float, pos_hi: float
) -> PositionToolBox:
    """Build the positioning tool set, closed over *state* (the working copy)."""
    box = PositionToolBox()

    view_slices = make_view_slices(state, ctx)

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
            confidence = str(entry.get("confidence", "")).strip().lower()
            if confidence in CONFIDENCE_LEVELS:
                record.confidence = confidence
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

    async def estimate_slice(slice_id: str) -> dict[str, Any]:
        """Run a full single-slice atlas sweep on ONE section. Escalation only.

        This is the slow path: an independent agent sweeps the whole atlas for
        that one section and reports what it found. Use it for a section you
        cannot place by eye against its neighbours. The result is NOT written —
        judge it against the rest of the stack and, if you accept it, write it
        with `set_positions`.

        Args:
            slice_id: One filename from the stack manifest.

        Returns:
            The estimated position and the estimator's reasoning, or
            ``{"status": "error", ...}`` if the estimate failed.
        """
        record = state.by_id(str(slice_id))
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_ID", "id": str(slice_id)}
        try:
            result = await run_slice_estimation(
                image_path=ctx.image_path(record.id),
                atlas_name=state.atlas,
                plane=state.plane,
                model_name=ctx.model,
            )
        except Exception as exc:
            logger.warning("position: escalation failed for %s: %s", record.id, exc)
            return {
                "status": "error",
                "error": "ESTIMATE_FAILED",
                "id": record.id,
                "message": str(exc),
            }
        box.escalations.append(record.id)
        return {
            "status": "ok",
            "id": record.id,
            "position_mm": round(float(result.position_mm), 3),
            "reasoning": result.reasoning,
            "current_position_mm": (
                round(record.position_mm, 3) if record.position_mm is not None else None
            ),
            "note": (
                "Not written. Call `set_positions` if you accept this estimate."
            ),
        }

    def get_advisories() -> dict[str, Any]:
        """Computed spacing signals for the stack as it currently stands.

        ADVISORY ONLY — these are arithmetic, not anatomy. They cannot see the
        images and know nothing about missing sections. Use them to spot where
        the stack disagrees with itself, then check those sections yourself.

        Returns:
            The neighbour-interval table, the residual between each section and
            what interpolating between the anchor sections would predict, and a
            monotone minimum-spacing curve fitted through the current positions.
        """
        return {
            "status": "ok",
            "advisory": (
                "Suggestions computed from the current numbers only. Nothing "
                "here has looked at a section image; verify before acting."
            ),
            "interval_table": interval_table(state),
            "interpolation_residuals": _interpolation_residuals(state),
            "monotone_fit": _monotone_suggestion(state),
        }

    def submit_positions(
        interval_breaks: list[int],
        notes: list[str],
        summary: str,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish the positioning step. Call this exactly once, last.

        Every section must have a position first, damaged sections included —
        the call is rejected if any is missing.

        Args:
            interval_breaks: Corrected indices where the spacing between
                neighbouring sections breaks the nominal interval — the index
                of the section AFTER the gap. Empty if the stack is regular.
            notes: Short observations worth carrying forward.
            summary: One or two sentences on what you changed and why.
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
        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
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
        set_positions,
        estimate_slice,
        get_advisories,
        submit_positions,
    ]
    return box


# --- prompt + seed message ----------------------------------------------


def build_position_prompt(
    *, state: StackState, species: str, pos_lo: float, pos_hi: float
) -> str:
    """System instruction for the positioning agent, plane-aware."""
    plane = state.plane
    axis = _PLANE_AXIS_LABEL.get(plane, "AP")
    anchors = [s.id for s in state.in_order() if s.position_source == "anchor"]
    anchor_line = (
        "The key sections (estimated directly, everything else interpolated "
        f"between them) are: {', '.join(anchors)}.\n"
        if anchors
        else "No section was estimated directly; every position is a guess.\n"
    )
    breaks_line = (
        f"The survey step flagged possible interval breaks at corrected "
        f"indices {state.interval_breaks}.\n"
        if state.interval_breaks
        else ""
    )
    damaged = [s.id for s in state.in_order() if s.damaged]
    damaged_line = (
        f"Damaged sections ({', '.join(damaged)}) still need positions — place "
        "them from their neighbours if their own anatomy is unreadable.\n"
        if damaged
        else ""
    )

    return (
        f"You are an expert neuroanatomist placing a stack of "
        f"{len(state.slices)} {plane} histology sections along the "
        f"{state.atlas} ({species}) atlas.\n\n"
        f"Stack facts:\n"
        f"- Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm, measured from "
        f"the anterior edge of the atlas volume\n"
        f"- Nominal section interval: {state.interval_mm:.3f} mm "
        f"center-to-center; slice thickness {state.thickness_mm:.3f} mm\n"
        f"- {anchor_line}"
        f"{breaks_line}"
        f"{damaged_line}\n"
        f"Every section already has a starting position. Your job is to check "
        f"and correct them, not to re-derive the stack from scratch.\n\n"
        f"HOW TO WORK:\n\n"
        f"1. VERIFY THE KEY SECTIONS FIRST. They carry the whole stack: every "
        f"other position was interpolated between them, so an error in a key "
        f"section is an error in all its neighbours. Compare each one against "
        f"the atlas — `fetch_atlas` around its current position, `view_slices` "
        f"to see the section itself — and correct it if it is off.\n\n"
        f"2. THEN LOOK FOR BREAKS IN THE INTERVAL. A consistent slicing "
        f"interval does NOT mean no sections are missing: sections get lost, "
        f"torn or skipped during collection, and the sections on either side "
        f"of the loss still look evenly spaced on the slide. Between two "
        f"verified key sections, the anatomy must advance by roughly "
        f"{state.interval_mm:.3f} mm per section. Where it advances faster "
        f"than that, sections are missing and the interpolated positions in "
        f"that stretch are wrong — investigate that stretch and reposition it. "
        f"Report every break you confirm in `submit_positions`.\n\n"
        f"3. WRITE POSITIONS IN BATCHES. `set_positions` takes many sections "
        f"in one call and hands back the resulting neighbour-interval table, "
        f"so you immediately see what your edit did to the spacing.\n\n"
        f"4. ESCALATE ONLY WHEN STUCK. `estimate_slice` runs a full atlas "
        f"sweep on one section. It is slow, so save it for a section you "
        f"cannot place from its neighbours. It does not write anything: judge "
        f"the number it returns against the rest of the stack, then write it "
        f"with `set_positions` if you accept it.\n\n"
        f"5. `get_advisories` gives you arithmetic — interval table, "
        f"interpolation residuals, a monotone spacing fit. It is advice from "
        f"numbers that have never seen an image. Use it to find suspicious "
        f"stretches, never as the answer.\n\n"
        f"Finish with `submit_positions`. It is rejected unless every section "
        f"in the stack has a position, damaged sections included."
    )


def build_position_seed_message(state: StackState) -> types.Content:
    """Contact sheet + positioned manifest: what the positioning step starts from."""
    parts: list[types.Part] = contact_sheet_parts(state)
    parts.append(
        types.Part.from_text(
            text=(
                "Stack manifest (corrected index, filename, current position "
                "and where it came from, current flags):\n"
                f"{build_stack_manifest(state, with_positions=True)}\n\n"
                "Verify the key ('anchor') sections against the atlas, hunt "
                "for breaks in the interval between them, correct positions "
                "with `set_positions`, then call `submit_positions`."
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
        seed_message=build_position_seed_message(state),
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
    )

    return PositionOutcome(
        findings=dict(box.submission) if box.submission else None,
        positions_written=len(set(box.written)),
        escalations=len(box.escalations),
        tool_calls=tool_calls,
        turns=turns,
    )
