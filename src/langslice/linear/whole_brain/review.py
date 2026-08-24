"""The review agent: one last look at the whole stack before hand-back.

Everything upstream worked on part of the problem — a section at a time, a
signal at a time. This step is the only one that sees the finished article:
positions, where each came from, confidences, damage, and which sections ended
up with a transform. It can attach caveats, and it can send the stack back to
the positioning step once if the order or the spacing does not hold together.

It cannot block hand-back. A review that runs out of turns approves with a note
— an un-reviewed result is still a result, and the caveats say so.
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
    contact_sheet_parts,
    make_view_slices,
    run_agent_session,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.position import spacing_advisories
from langslice.linear.whole_brain.state import StackState, apply_confidence

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one review pass may spend.
DEFAULT_REVIEW_MAX_ITERATIONS = 10

DEFAULT_REVIEW_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_review"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: check anything doubtful "
    "with `view_slices` and `fetch_atlas`, attach caveats with `flag_slice`, "
    "and finish with `submit_review`."
)
_NUDGE_CONTINUE = (
    "Please finish the review. Flag anything that needs a caveat, then call "
    "`submit_review` with your verdict."
)


# --- outcome -------------------------------------------------------------


@dataclass
class ReviewOutcome:
    """What one review pass produced.

    ``findings`` is ``None`` when the agent never submitted (turn budget
    exhausted); any flags its tools already wrote stay on the state.
    """

    findings: dict[str, Any] | None = None
    flagged: int = 0
    tool_calls: int = 0
    turns: int = 0


@dataclass
class ReviewToolBox:
    """Tool callables plus the mutable results the node reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)


# --- manifest ------------------------------------------------------------


def review_manifest(state: StackState) -> str:
    """One line per section: position, provenance, flags, transform, caveats."""
    lines: list[str] = []
    for record in state.in_order():
        position = (
            f"{record.position_mm:.3f} mm" if record.position_mm is not None else "NO POSITION"
        )
        bits = [record.position_source or "unknown source"]
        if record.confidence:
            bits.append(f"{record.confidence} confidence")
        if record.flip:
            bits.append("flipped")
        if record.damaged:
            bits.append(
                f"damaged: {record.damage_note}" if record.damage_note else "damaged"
            )
        if record.affine is not None:
            bits.append("affine")
        if record.interactive_transform is not None:
            bits.append("interactive transform")
        if record.affine is None and record.interactive_transform is None:
            bits.append("NO TRANSFORM")
        if record.caveats:
            bits.append("caveats: " + "; ".join(record.caveats))
        lines.append(
            f"{record.index_corrected:>3}  {record.id}  {position}  [{', '.join(bits)}]"
        )
    return "\n".join(lines)


# --- tools ---------------------------------------------------------------


def build_review_tools(state: StackState, ctx: EngineContext) -> ReviewToolBox:
    """Build the review tool set, closed over *state* (the working copy)."""
    box = ReviewToolBox()

    view_slices = make_view_slices(state, ctx)

    def flag_slice(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Attach a caveat to one or more sections. Batch: one call, many sections.

        Use it for anything a downstream user should know: a position you are
        not sure of, a section whose transform could not cover the damage, a
        stretch of the stack whose spacing looks wrong but that you could not
        resolve. Caveats accumulate; they never overwrite each other.

        Args:
            entries: ``[{"id": "<filename>", "caveat": "<short warning>",
                "confidence": "low" | "medium" | "high"}]``. ``confidence`` is
                optional and replaces the section's current confidence.

        Returns:
            Which sections were flagged and which ids were not recognised.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        flagged: list[str] = []
        unknown: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.by_id(str(entry.get("id", "")))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            caveat = str(entry.get("caveat", "")).strip()
            if caveat and caveat not in record.caveats:
                record.caveats.append(caveat)
            apply_confidence(record, entry.get("confidence"))
            flagged.append(record.id)
            box.flagged.append(record.id)
        return {
            "status": "ok" if flagged else "error",
            "flagged": flagged,
            "unknown_ids": unknown,
            **({} if flagged else {"error": "NOTHING_FLAGGED"}),
        }

    def submit_review(
        approved: bool,
        notes: list[str],
        summary: str,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish the review. Call this exactly once, last.

        Args:
            approved: True to hand the stack back as it stands. False sends it
                back to the positioning step for ONE more pass — only worth it
                if you can say in your notes what specifically has to change,
                because the positioning step reads them.
            notes: Short, concrete observations. If you are not approving, this
                is the instruction the next positioning pass works from.
            summary: One or two sentences on the state of the stack.
        """
        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
        box.submission.update(
            {
                "approved": bool(approved),
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
        return {"status": "ok", "approved": bool(approved)}

    box.tools = [view_slices, fetch_atlas, flag_slice, submit_review]
    return box


# --- prompt + seed message ----------------------------------------------


def build_review_prompt(*, state: StackState, species: str) -> str:
    """System instruction for the review agent."""
    return (
        f"You are an expert neuroanatomist signing off on a stack of "
        f"{len(state.slices)} {state.plane} histology sections that has been "
        f"ordered, positioned along the {state.atlas} ({species}) atlas, and "
        f"given a proposed in-plane transform per section.\n\n"
        f"Nominal section interval: {state.interval_mm:.3f} mm "
        f"center-to-center; slice thickness {state.thickness_mm:.3f} mm.\n\n"
        f"This is the last look before the results go back to the user. You "
        f"are checking the stack as a WHOLE, not re-doing the work:\n\n"
        f"1. SERIAL ORDER. Positions must advance in one direction down the "
        f"stack. A section that goes backwards is either misordered or "
        f"misplaced.\n\n"
        f"2. SPACING. Neighbouring sections should be about "
        f"{state.interval_mm:.3f} mm apart. Isolated jumps are worth a look: "
        f"they are either a real break in the interval (sections lost during "
        f"collection — legitimate, and already reported) or a misplaced "
        f"section. Do not expect perfect regularity.\n\n"
        f"3. CAVEATS. Damaged sections, low-confidence positions, sections "
        f"whose transform is missing or weak — say so with `flag_slice`. A "
        f"caveat costs nothing; a silent bad section costs the user their "
        f"analysis.\n\n"
        f"4. VERDICT. `submit_review(approved=True)` hands the stack back. "
        f"`approved=False` sends it to ONE more positioning pass — use it only "
        f"when something concrete is wrong AND your notes say what, because "
        f"those notes are what the next pass reads. If the stack is merely "
        f"imperfect, approve it with caveats instead.\n\n"
        f"The spacing signals in the manifest are arithmetic: they have never "
        f"seen an image and know nothing about missing sections. Use "
        f"`view_slices` and `fetch_atlas` to check anything they flag before "
        f"you act on it."
    )


def build_review_seed_message(state: StackState) -> types.Content:
    """Contact sheet + full manifest + advisory spacing signals."""
    advisories = spacing_advisories(state)
    parts: list[types.Part] = contact_sheet_parts(state)
    parts.append(
        types.Part.from_text(
            text=(
                "Final stack manifest (corrected index, filename, position, "
                "provenance and flags):\n"
                f"{review_manifest(state)}\n\n"
                "Advisory spacing signals (arithmetic only — no image has been "
                "looked at):\n"
                f"- neighbour intervals: {advisories['interval_table']}\n"
                f"- interpolation residuals: {advisories['interpolation_residuals']}\n"
                f"- monotone spacing fit: {advisories['monotone_fit']}\n\n"
                f"Run notes so far:\n{_recent_notes(state)}\n\n"
                "Review the stack, flag what needs flagging, then call "
                "`submit_review`."
            )
        )
    )
    return types.Content(role="user", parts=parts)


def _recent_notes(state: StackState, limit: int = 12) -> str:
    recent = state.notes[-limit:]
    return "\n".join(f"- {note}" for note in recent) if recent else "- (none)"


# --- agent + session driver ---------------------------------------------


def build_review_agent(
    *,
    state: StackState,
    tools: list[Any],
    species: str,
    model: str | object = DEFAULT_REVIEW_MODEL,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
) -> LlmAgent:
    """Construct the review LlmAgent."""
    # Same shape as the other whole-brain agents: kwargs dict so the enum-typed
    # media_resolution string is accepted as-is.
    config_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        "max_output_tokens": 4000,
        "media_resolution": media_resolution,
        "http_options": default_http_options(),
    }
    return LlmAgent(
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name="whole_brain_review",
        instruction=build_review_prompt(state=state, species=species),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )


async def run_review_session(
    *,
    state: StackState,
    ctx: EngineContext,
    species: str,
    pos_lo: float,
    pos_hi: float,
    max_iterations: int = DEFAULT_REVIEW_MAX_ITERATIONS,
) -> ReviewOutcome:
    """Drive one review pass and return what it produced."""
    box = build_review_tools(state, ctx)
    agent = build_review_agent(
        state=state,
        tools=box.tools,
        species=species,
        model=ctx.model or DEFAULT_REVIEW_MODEL,
    )

    tool_calls, turns = await run_agent_session(
        agent=agent,
        state=state,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        seed_message=build_review_seed_message(state),
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
    )

    return ReviewOutcome(
        findings=dict(box.submission) if box.submission else None,
        flagged=len(set(box.flagged)),
        tool_calls=tool_calls,
        turns=turns,
    )
