"""Claude's concise job statement, separate from the ADK model's method."""
from __future__ import annotations

from langslice.linear.engine import EngineContext
from langslice.linear.prompt import run_facts, task_notes
from langslice.linear.render import status_text
from langslice.linear.spec import JobSpec
from langslice.linear.state import StackState


def job_statement(spec: JobSpec, state: StackState, ctx: EngineContext, pages: int,
                  notes: str = "") -> str:
    low, high = ctx.position_range
    labels = {"reorder": "correct section order and orientation",
              "position": "position every section in the atlas",
              "transform": "align every unlocked section in-plane"}
    lines = ["Register this histology stack against its reference atlas.",
             "Selected work: " + "; ".join(labels[task] for task in spec.tasks) + ".",
             "Work only through the LangSlice tools and finish with submit.",
             "Read every opening-picture page with show_stack(page=1) through "
             f"show_stack(page={pages}) before any write.",
             "The last page is the atlas reference strip; preceding pages show individually "
             "labelled sections in corrected order.",
             "Run facts:",
             *run_facts(spec, state, species=ctx.species, pos_lo=low, pos_hi=high,
                        axis_ends=ctx.axis_ends),
             *task_notes(spec)]
    if notes.strip():
        lines.extend(["User notes:", notes])
    lines.extend(["Status table:", status_text(state)])
    return "\n".join(lines)
