"""The survey agent: one fused triage pass over the whole stack.

Damage, hemisphere flips, ordering and interval breaks are one agent step, not
four — a reader who is already looking at the whole stack sees all four at
once. This module owns the tools, the prompt, the agent, and a lean session
driver; the routing decision lives with the other nodes in
:mod:`langslice.linear.whole_brain.nodes`.

Corrections are DATA: ``flip_slices``/``reorder_slices``/``mark_damaged``
write to the :class:`~langslice.linear.whole_brain.state.StackState` the node
handed in (the tools close over it) and never touch the user's files.
``view_slices`` renders through that same corrected view, so what the agent
sees is what the state says.
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
from langslice.linear.whole_brain.state import StackState

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one survey pass may spend.
DEFAULT_SURVEY_MAX_ITERATIONS = 15

DEFAULT_SURVEY_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_survey"

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

_PLANE_DIRECTION_HINT: dict[str, str] = {
    "coronal": '{"ap": "anterior_to_posterior"} or {"ap": "posterior_to_anterior"}',
    "sagittal": '{"ml": "lateral_to_medial"} or {"ml": "medial_to_lateral"}',
    "horizontal": '{"dv": "dorsal_to_ventral"} or {"dv": "ventral_to_dorsal"}',
}

#: (section heading, what a mirrored section looks like, where the orientation
#: notch is cut) per plane — a sagittal section has no in-plane hemispheres.
_PLANE_MIRROR_CUE: dict[str, tuple[str, str, str]] = {
    "coronal": (
        "HEMISPHERE FLIPS",
        "a left-right hemisphere flip: the section is mirrored about the "
        "midline relative to the rest of the stack",
        "one hemisphere",
    ),
    "horizontal": (
        "HEMISPHERE FLIPS",
        "a left-right hemisphere flip: the section is mirrored about the "
        "midline relative to the rest of the stack",
        "one hemisphere",
    ),
    "sagittal": (
        "MIRRORED SECTIONS",
        "an anterior-posterior mirror: the section faces the opposite way "
        "from the rest of the stack",
        "one side of the section",
    ),
}

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: inspect with "
    "`view_slices` or `fetch_atlas`, apply corrections with `flip_slices`, "
    "`reorder_slices` or `mark_damaged`, and finish with `submit_survey`."
)
_NUDGE_CONTINUE = (
    "Please continue the survey. Inspect anything still unclear with "
    "`view_slices`, apply any remaining corrections, then call "
    "`submit_survey` with your findings."
)


# --- outcome -------------------------------------------------------------


@dataclass
class SurveyOutcome:
    """What one survey session produced.

    ``findings`` is ``None`` when the agent never submitted (turn budget
    exhausted); any corrections its tools already applied stay on the state.
    """

    findings: dict[str, Any] | None = None
    corrections_applied: bool = False
    tool_calls: int = 0
    turns: int = 0


# --- tools ---------------------------------------------------------------


@dataclass
class SurveyToolBox:
    """Tool callables plus the mutable results the node reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)


def build_survey_tools(state: StackState, ctx: EngineContext) -> SurveyToolBox:
    """Build the survey tool set, closed over *state* (the working copy)."""
    box = SurveyToolBox()

    view_slices = make_view_slices(state, ctx)

    def _known(slice_ids: list[str]) -> tuple[list[str], list[str]]:
        return split_known_ids(state, slice_ids)

    def flip_slices(slice_ids: list[str]) -> dict[str, Any]:
        """Toggle the hemisphere flip on the named sections.

        Call once with every mirrored section. Flips are recorded as data —
        the user's image files are never modified. Calling it twice on the
        same section undoes the flip.

        Args:
            slice_ids: Filenames from the stack manifest.

        Returns:
            Which sections were toggled and the full flipped set afterwards.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        wanted, unknown = _known(list(slice_ids))
        for slice_id in wanted:
            record = state.by_id(slice_id)
            assert record is not None
            record.flip = not record.flip
        if wanted:
            box.corrections.append(f"flipped {len(wanted)} section(s)")
        return {
            "status": "ok",
            "toggled": wanted,
            "unknown_ids": unknown,
            "flipped_now": [s.id for s in state.in_order() if s.flip],
        }

    def reorder_slices(new_order: list[str]) -> dict[str, Any]:
        """Set the corrected order of the WHOLE stack in one call.

        Args:
            new_order: Every section filename exactly once, in the order the
                sections were cut. This replaces the current order; there is
                no incremental move operation.

        Returns:
            The accepted order, or an error if the stack order is locked or
            the list is not a permutation of the stack.
        """
        if state.keep_order:
            return {
                "status": "error",
                "error": "ORDER_LOCKED",
                "message": (
                    "The host locked the section order for this run "
                    "(keep_order). Report ordering problems in your survey "
                    "notes instead; flips and damage can still be applied."
                ),
            }
        ids = [s.id for s in state.slices]
        if sorted(new_order) != sorted(ids):
            missing = [sid for sid in ids if sid not in new_order]
            unknown = [sid for sid in new_order if sid not in ids]
            return {
                "status": "error",
                "error": "NOT_A_PERMUTATION",
                "message": (
                    f"new_order must list all {len(ids)} section filenames "
                    "exactly once."
                ),
                "missing_ids": missing,
                "unknown_ids": unknown,
            }
        for index, slice_id in enumerate(new_order):
            record = state.by_id(slice_id)
            assert record is not None
            record.index_corrected = index
        box.corrections.append("reordered the stack")
        return {"status": "ok", "order": list(new_order)}

    def mark_damaged(entries: list[dict[str, str]]) -> dict[str, Any]:
        """Record damaged sections and why, for the later registration steps.

        Args:
            entries: ``[{"id": "<filename>", "note": "<short description>"}]``
                — tears, folds, missing tissue, bubbles, knife chatter.

        Returns:
            Which sections are marked damaged afterwards.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        marked: list[str] = []
        unknown: list[str] = []
        for entry in entries:
            slice_id = str(entry.get("id", ""))
            record = state.by_id(slice_id)
            if record is None:
                unknown.append(slice_id)
                continue
            record.damaged = True
            record.damage_note = str(entry.get("note", "")).strip()
            marked.append(slice_id)
        if marked:
            box.corrections.append(f"marked {len(marked)} section(s) damaged")
        return {
            "status": "ok",
            "marked": marked,
            "unknown_ids": unknown,
            "damaged_now": [s.id for s in state.in_order() if s.damaged],
        }

    def submit_survey(
        axis_directions: dict[str, str],
        interval_breaks: list[int],
        notes: list[str],
        clean: bool,
        summary: str,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish the survey. Call this exactly once, last.

        Args:
            axis_directions: Direction the stack runs along its axis, e.g.
                ``{"ap": "anterior_to_posterior"}``.
            interval_breaks: Corrected indices where the spacing between
                neighbouring sections breaks the nominal interval — the index
                of the section AFTER the gap. Empty if the stack is regular.
            notes: Short observations worth carrying forward.
            clean: False if you applied ANY correction in this pass (flip,
                reorder, damage) — the stack then gets one re-check pass.
                True only if the stack needed nothing.
            summary: One or two sentences on what you found.
        """
        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
        box.submission.update(
            {
                "axis_directions": (
                    {str(k): str(v) for k, v in axis_directions.items()}
                    if isinstance(axis_directions, dict)
                    else {}
                ),
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
                "clean": bool(clean),
                "summary": str(summary or ""),
            }
        )
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", "clean": bool(clean)}

    box.tools = [
        view_slices,
        fetch_atlas,
        flip_slices,
        reorder_slices,
        mark_damaged,
        submit_survey,
    ]
    return box


# --- prompt + seed message ----------------------------------------------


def build_survey_prompt(
    *, state: StackState, species: str, pos_lo: float, pos_hi: float
) -> str:
    """System instruction for the survey agent, plane-aware."""
    plane = state.plane
    axis = _PLANE_AXIS_LABEL.get(plane, "AP")
    flip_heading, mirror_cue, notch_site = _PLANE_MIRROR_CUE.get(
        plane, _PLANE_MIRROR_CUE["coronal"]
    )
    direction_hint = _PLANE_DIRECTION_HINT.get(plane, _PLANE_DIRECTION_HINT["coronal"])
    order_rule = (
        "The host locked the section order for this run, so `reorder_slices` "
        "will refuse. If the order still looks wrong, say so in your notes."
        if state.keep_order
        else "You may reorder the stack with `reorder_slices`."
    )
    known_directions = (
        f"Already known: {state.axis_directions}.\n" if state.axis_directions else ""
    )

    return (
        f"You are an expert neuroanatomist triaging a stack of "
        f"{len(state.slices)} {plane} histology sections before they are "
        f"registered to a reference atlas.\n\n"
        f"Stack facts:\n"
        f"- Atlas: {state.atlas} ({species}); valid {axis} range "
        f"{pos_lo:.2f}-{pos_hi:.2f} mm\n"
        f"- Nominal section interval: {state.interval_mm:.3f} mm "
        f"center-to-center; slice thickness {state.thickness_mm:.3f} mm\n"
        f"- {order_rule}\n"
        f"{known_directions}\n"
        f"You are shown a contact sheet of the whole stack in its current "
        f"corrected order (each thumbnail is labelled "
        f"'<index>: <filename>') plus the same stack as a text manifest. "
        f"The filenames are context for the intended order — they usually "
        f"encode the order the sections were cut.\n\n"
        f"In ONE pass, triage the stack for four things at once:\n\n"
        f"1. DAMAGE. Tears, folds, missing tissue, bubbles, knife chatter. "
        f"Record each one with a short note via `mark_damaged`. This list is "
        f"needed later: damaged sections take a different registration path.\n\n"
        f"2. {flip_heading}. Sections are rarely mounted in a consistent "
        f"orientation, so expect some to be mirrored. A flipped section shows "
        f"{mirror_cue}. Two cues:\n"
        f"   - A notch, cut or score deliberately made into {notch_site} as "
        f"an orientation marker. It should sit on the SAME side in every "
        f"section; a section where it has jumped to the other side is "
        f"flipped.\n"
        f"   - Asymmetry from oblique slicing: when the block was cut at an "
        f"angle, the same structure appears at a slightly different level on "
        f"the two sides of a section. That bias is consistent down the stack, "
        f"so a section whose asymmetry runs the other way is flipped.\n"
        f"   Call `flip_slices` once with every section that is mirrored "
        f"relative to the majority.\n\n"
        f"3. ORDER. Sections whose index is wrong. This is about ORDER ONLY — "
        f"you are NOT estimating where any section sits in the brain here, "
        f"only whether the stack runs in a consistent sequence. Anatomy "
        f"should change smoothly and in one direction from each section to "
        f"the next; a section that breaks that progression and fits somewhere "
        f"else belongs there. Submit the whole corrected order in one "
        f"`reorder_slices` call.\n\n"
        f"4. INTERVAL BREAKS. Watch for breaks in the interval: neighbouring "
        f"sections whose anatomy jumps much further than the nominal "
        f"{state.interval_mm:.3f} mm step (missing sections), or barely "
        f"changes at all. Nothing is fixed here — report them as corrected "
        f"indices in `submit_survey(interval_breaks=...)`, using the index of "
        f"the section AFTER the gap. Later steps consume the list.\n\n"
        f"Also determine which way the stack runs along its axis and report "
        f"it as `axis_directions`, e.g. {direction_hint}. `fetch_atlas` "
        f"gives you reference sections to compare against if the direction "
        f"is not obvious from the stack alone.\n\n"
        f"HOW TO WORK:\n"
        f"- Start from the contact sheet. Call `view_slices` (up to 8 "
        f"sections per call) on anything the thumbnails are too small to "
        f"judge.\n"
        f"- Batch your corrections: one `flip_slices` call with every "
        f"mirrored id, one `reorder_slices` call with the full corrected "
        f"order, one `mark_damaged` call with every damaged section.\n"
        f"- Corrections are recorded as data; the user's image files are "
        f"never modified.\n"
        f"- Finish with `submit_survey`. Set clean=false if you applied any "
        f"correction in this pass (you will get one re-check pass with a "
        f"refreshed contact sheet); clean=true only if the stack needed "
        f"nothing."
    )


def build_seed_message(state: StackState) -> types.Content:
    """Contact sheet + manifest: everything the survey starts from."""
    parts: list[types.Part] = contact_sheet_parts(state)
    parts.append(
        types.Part.from_text(
            text=(
                "Stack manifest (corrected index, filename, current flags):\n"
                f"{build_stack_manifest(state)}\n\n"
                "Survey this stack: damage, hemisphere flips, order, interval "
                "breaks. Apply what you find with the tools, then call "
                "`submit_survey`."
            )
        )
    )
    return types.Content(role="user", parts=parts)


# --- agent + session driver ---------------------------------------------


def build_survey_agent(
    *,
    state: StackState,
    tools: list[Any],
    species: str,
    pos_lo: float,
    pos_hi: float,
    model: str | object = DEFAULT_SURVEY_MODEL,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
) -> LlmAgent:
    """Construct the survey LlmAgent."""
    # Same shape as build_single_slice_agent: kwargs dict so the enum-typed
    # media_resolution string is accepted as-is.
    config_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        # A full reorder of a large stack is a long argument list; leave room
        # for it plus the model's reasoning.
        "max_output_tokens": 8000,
        "media_resolution": media_resolution,
        "http_options": default_http_options(),
    }
    return LlmAgent(
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name="whole_brain_survey",
        instruction=build_survey_prompt(
            state=state, species=species, pos_lo=pos_lo, pos_hi=pos_hi
        ),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )


async def run_survey_session(
    *,
    state: StackState,
    ctx: EngineContext,
    species: str,
    pos_lo: float,
    pos_hi: float,
    max_iterations: int = DEFAULT_SURVEY_MAX_ITERATIONS,
) -> SurveyOutcome:
    """Drive one survey pass and return what it produced.

    The tools mutate *state* as they are called, so corrections survive even
    when the agent never reaches ``submit_survey``.
    """
    box = build_survey_tools(state, ctx)
    agent = build_survey_agent(
        state=state,
        tools=box.tools,
        species=species,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        model=ctx.model or DEFAULT_SURVEY_MODEL,
    )

    tool_calls, turns = await run_agent_session(
        agent=agent,
        state=state,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        seed_message=build_seed_message(state),
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
    )

    return SurveyOutcome(
        findings=dict(box.submission) if box.submission else None,
        corrections_applied=bool(box.corrections),
        tool_calls=tool_calls,
        turns=turns,
    )
