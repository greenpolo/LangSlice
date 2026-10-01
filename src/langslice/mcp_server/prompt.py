"""Claude's job statement: the ADK agent's, plus how the opening pictures arrive."""
from __future__ import annotations

from langslice.linear.display import display_facts
from langslice.linear.engine import EngineContext
from langslice.linear.prompt import build_job_statement
from langslice.linear.render import status_text
from langslice.linear.spec import JobSpec
from langslice.linear.state import StackState


def job_statement(spec: JobSpec, state: StackState, ctx: EngineContext, pages: int,
                  notes: str = "", tool_names: list[str] | None = None) -> str:
    """The same job, facts, tools, constraints and method the ADK agent is given.

    Only the delivery differs: the ADK agent's seed message carries the opening
    pictures, while here they arrive through ``show_stack`` pages.
    """
    low, high = ctx.position_range
    lines = [
        build_job_statement(
            spec, state, tool_names=tool_names or [], species=ctx.species,
            pos_lo=low, pos_hi=high, axis_ends=ctx.axis_ends,
            **display_facts(ctx, state),
        ),
        "",
        "Opening pictures: read every page with show_stack(page=1) through "
        f"show_stack(page={pages}) before any write. The last page is the atlas "
        "reference strip; the pages before it show every section, individually "
        "labelled, in the stack's current order.",
    ]
    if notes.strip():
        lines.extend(["", "User notes:", notes])
    lines.extend(["", "Status table:", status_text(state)])
    return "\n".join(lines)
