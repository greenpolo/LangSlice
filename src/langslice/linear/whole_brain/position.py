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
from langslice.linear.tools import fetch_atlas
from langslice.linear.whole_brain._step_common import (
    build_stack_manifest,
    contact_sheet_parts,
    make_view_slices,
    run_agent_session,
    split_known_ids,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.estimation_agents import run_slice_estimation
from langslice.linear.whole_brain.signals import interpolate_positions, monotone_fit
from langslice.linear.whole_brain.state import (
    SliceState,
    StackState,
    apply_confidence,
)

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one positioning pass may spend. This
#: step now owns strategy as well as placement, so the budget is generous.
DEFAULT_POSITION_MAX_ITERATIONS = 40

#: Sections one ``estimate_slices`` call may estimate. Each one is a full
#: single-slice agent session, so the cap is about cost, not correctness.
MAX_ESTIMATE_SLICES = 8

DEFAULT_POSITION_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_position"

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: compare sections against "
    "the atlas with `view_slices` and `fetch_atlas`, estimate sections with "
    "`estimate_slices`, write positions with `set_positions`, and finish with "
    "`submit_positions`."
)
_NUDGE_CONTINUE = (
    "Please continue placing the stack. Check anything still unresolved, write "
    "the positions you have settled on with `set_positions`, then call "
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
        steps by the nominal section interval. It is arithmetic — it has never
        looked at an image and knows nothing about sections lost during
        collection, so a stretch where the anatomy advances faster than the
        suggestion says is exactly where you should look.

        Nothing is written. Write what you accept with `set_positions`.

        Args:
            fixed: The points to interpolate between, as
                ``[{"id": "<filename>", "position_mm": <number>}]``. Two or
                more is normal; one only fixes an offset.

        Returns:
            One suggested position per section, each marked as fixed by you or
            interpolated.
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
        anchors = [value for value in known if value is not None]
        if not anchors:
            return {
                "status": "error",
                "error": "NO_FIXED_POINTS",
                "unknown_ids": unknown,
                "rejected": rejected,
            }

        # Direction comes from the fixed points themselves: a stack cut
        # back-to-front runs the other way and its extrapolation must too.
        step = state.interval_mm if anchors[-1] >= anchors[0] else -state.interval_mm
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
            "note": (
                "Suggestions only, nothing written, no image looked at. Check "
                "the stretches between your fixed points before accepting them."
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
        f"- Nominal section interval: {state.interval_mm:.3f} mm "
        f"center-to-center; slice thickness {state.thickness_mm:.3f} mm\n"
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
        f"difference.\n\n"
        f"3. CHECK BOTH ENDS BEFORE YOU SUBMIT. A ladder with plausible "
        f"spacing hung at the wrong absolute position is perfectly consistent "
        f"from the inside — every interval looks right and every section still "
        f"sits wrong. The ends are where that shows: verify the first and the "
        f"last section of the stack against the atlas independently, and if "
        f"either one does not match, the whole placement is offset, not just "
        f"that section.\n\n"
        f"4. WATCH FOR BREAKS IN THE INTERVAL. A consistent slicing interval "
        f"does NOT mean no sections are missing: sections get lost, torn or "
        f"skipped during collection, and the sections on either side of the "
        f"loss still look evenly spaced on the slide. Between two verified "
        f"sections the anatomy must advance by roughly {state.interval_mm:.3f} "
        f"mm per section; where it advances faster, sections are missing and "
        f"any interpolation across that stretch is wrong. Investigate the "
        f"stretch, reposition it, and report every break you confirm in "
        f"`submit_positions`.\n\n"
        f"5. WRITE IN BATCHES. `set_positions` takes many sections in one call "
        f"and hands back the resulting neighbour-interval table, so you see "
        f"immediately what your edit did to the spacing.\n\n"
        f"6. THE ARITHMETIC IS ADVICE. `interpolate_between` and "
        f"`get_advisories` compute from numbers that have never seen an image "
        f"and know nothing about missing sections. Use them to find suspicious "
        f"stretches, never as the answer.\n\n"
        f"Finish with `submit_positions`. It is rejected unless every section "
        f"in the stack has a position, damaged sections included."
    )


def build_position_seed_message(state: StackState, *, note_limit: int = 12) -> types.Content:
    """Contact sheet + stack manifest: what the positioning step starts from.

    Sections with no position yet read as "unplaced" in the manifest — the
    normal case, since nothing upstream places them.

    The recent run notes ride along because this step is also the loop-back
    target: when the review step refuses a stack it writes what is wrong into
    the notes, and this is where the next pass reads it.
    """
    recent = state.notes[-note_limit:]
    notes_block = "\n".join(f"- {note}" for note in recent) if recent else "- (none)"
    parts: list[types.Part] = contact_sheet_parts(state)
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
                "the tools, write positions with `set_positions`, then call "
                "`submit_positions`."
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
        estimated=len(box.estimated),
        tool_calls=tool_calls,
        turns=turns,
    )
