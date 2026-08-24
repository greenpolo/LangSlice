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
    make_view_slices,
    run_agent_session,
    stack_image_parts,
)
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.position import position_rows
from langslice.linear.whole_brain.state import StackState, apply_confidence

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one review pass may spend.
DEFAULT_REVIEW_MAX_ITERATIONS = 10

DEFAULT_REVIEW_MODEL = "gemini-3-flash-preview"

_RUN_LABEL = "whole_brain_review"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Continue with the tools rather than in prose; "
    "when you have a verdict, call `submit_review`."
)
_NUDGE_CONTINUE = "Continue; when you have a verdict, call `submit_review`."


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

        Caveats accumulate on a section; they never overwrite each other.
        Caveats ride out with the results and are what a downstream user sees.

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
            approved: True hands the stack back as it stands. False sends it
                back to the positioning step for ONE more pass, which reads
                ``notes``.
            notes: Short, concrete observations. They are carried on the run
                and are what a returned stack's next positioning pass reads.
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
    """System instruction for the review agent.

    Same lean treatment as the positioning prompt: the job, the run's facts,
    the tools and what the verdict does. No checklist, no rules of thumb.
    """
    return (
        f"You are an expert neuroanatomist signing off on a stack of "
        f"{len(state.slices)} {state.plane} histology sections that has been "
        f"ordered, positioned along the {state.atlas} ({species}) atlas, and "
        f"given a proposed in-plane transform per section.\n\n"
        f"Your job: look at the finished stack as a whole and decide whether "
        f"it goes back to the user as it stands.\n\n"
        f"Run facts:\n"
        f"- {len(state.slices)} sections, {state.plane} plane, atlas "
        f"{state.atlas} ({species}).\n"
        f"- Cutting protocol: nominal section interval "
        f"{state.interval_mm:.3f} mm center-to-center, section thickness "
        f"{state.thickness_mm:.3f} mm.\n"
        f"- Interval breaks reported by the positioning step: "
        f"{state.interval_breaks or 'none'}.\n\n"
        f"Tools:\n"
        f"- `view_slices`: up to 8 named sections at higher resolution.\n"
        f"- `fetch_atlas`: atlas sections at the positions you name.\n"
        f"- `flag_slice`: attaches a caveat, and optionally a confidence, to "
        f"one or more sections. Caveats travel out with the results.\n"
        f"- `submit_review`: ends the step. `approved=True` hands the stack "
        f"back; `approved=False` sends it to ONE more positioning pass, which "
        f"reads your notes.\n\n"
        f"Call `submit_review` when you have a verdict."
    )


def build_review_seed_message(state: StackState, ctx: EngineContext) -> types.Content:
    """Per-section images, the full manifest, and the stack's own spacing."""
    parts: list[types.Part] = stack_image_parts(state, ctx)
    parts.append(
        types.Part.from_text(
            text=(
                "Final stack manifest (corrected index, filename, position, "
                "provenance and flags):\n"
                f"{review_manifest(state)}\n\n"
                "Positions and neighbour spacing (corrected index, filename, "
                "position in mm, spacing to the next placed section):\n"
                f"{position_rows(state)}\n\n"
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
        seed_message=build_review_seed_message(state, ctx),
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
