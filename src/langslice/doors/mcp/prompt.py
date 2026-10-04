"""Claude's job statement: the ADK agent's, plus how the opening pictures arrive."""
from __future__ import annotations

from dataclasses import replace

from langslice.agent.engine import EngineContext
from langslice.agent.prompt import build_job_statement, display_facts
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.core.status import status_text

#: Said to Claude when the job's image model is not connected to LangSlice.
IMAGE_MODEL_OFF = (
    "The image-model tool (trace_borders) is off: no image model is connected to "
    "LangSlice (no key or login for this job's image provider). Fit each "
    "section's deformation to its stain with fit_deformable."
)


def job_statement(spec: JobSpec, state: StackState, ctx: EngineContext, pages: int,
                  notes: str = "", tool_names: list[str] | None = None,
                  max_resolution: int | None = None, image_model_off: bool = False) -> str:
    """The same job, facts, tools, constraints and method the ADK agent is given.

    Only the delivery differs: the ADK agent's seed message carries the opening
    pictures, while here they arrive through ``show_stack`` pages.
    *image_model_off*: the job's image model is not connected here
    (``server.image_model_off``), so the statement is that of a run without
    one, plus a line saying the image-model tool is off and why.
    """
    low, high = ctx.position_range
    if image_model_off:
        spec = replace(spec, nonlinear=replace(spec.nonlinear, provider="none"))
    lines = [
        build_job_statement(
            spec, state, tool_names=tool_names or [], species=ctx.species,
            pos_lo=low, pos_hi=high, axis_ends=ctx.axis_ends,
            max_resolution=max_resolution, **display_facts(ctx, state),
        ),
        "",
        "Opening pictures: read every page with show_stack(page=1) through "
        f"show_stack(page={pages}) before any write. They show every section in "
        "strips, in the stack's current order, each labelled, with the atlas at "
        "its current position beneath it when the stack has positions; when a "
        "section has none, atlas reference strips at evenly spaced positions "
        "follow.",
    ]
    if image_model_off:
        lines.extend(["", IMAGE_MODEL_OFF])
    if notes.strip():
        lines.extend(["", "User notes:", notes])
    lines.extend(["", "Status table:", status_text(state)])
    return "\n".join(lines)
