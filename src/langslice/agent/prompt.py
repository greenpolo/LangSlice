"""The job statement: the job, the run facts, the tools by name, the constraints, the method.

The statement reports; the model reasons. Each tool is described once, by
its own description (:mod:`langslice.doors.declarations`); the statement
names the tools and describes none. It gives no rules of thumb and no
warnings about failure modes. The one place for strategy is the short
"Method:" section per task that is on (look before writing, review the
stack, inspect each fit against internal anatomy); ``position.playbook``
replaces the positioning lines with a step-by-step method for models that
do not find one on their own.

Deliberately atlas- and plane-agnostic — it names no region, no landmark and no
absolute position, because the same text runs against every BrainGlobe atlas,
species and plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langslice.core.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.core.state import StackState

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

#: ``SliceState.position_source`` of a starting position the job gave a
#: section (``job.job.DEFAULT_POSITION``).
STARTING_POSITION = "default"


def display_facts(
    ctx: Workspace, state: StackState,
) -> dict[str, Any]:
    """The job statement's display facts: raw channels and atlas layers here.

    ``channels`` is one list when every section shares it, else a mapping
    filename -> names. Unreadable files contribute nothing (the seed will
    report them).
    """
    from langslice.core.display import available_atlas_channels

    named: dict[str, list[str]] = {}
    for record in state.in_order():
        try:
            named[record.id] = list(ctx.section_channels(record.id)[0])
        except Exception:
            continue
    distinct = {tuple(names) for names in named.values()}
    channels: Any = list(next(iter(distinct))) if len(distinct) == 1 else named
    return {"channels": channels or None, "atlas_channels": available_atlas_channels(ctx)}


def channel_facts(
    channels: list[str] | dict[str, list[str]] | None,
    atlas_channels: tuple[str, ...] | None,
    resolution: int | None,
) -> list[str]:
    """The run facts on what the pictures can show: the raw channels, the
    atlas layers offered here and, where the agent sizes pictures, the
    largest ``look`` picture."""
    lines: list[str] = []
    raw = ("the planes of each section's file as read; the names come from the file or "
           "the host and may not say which is the stain")
    if isinstance(channels, list) and channels:
        lines.append(f"- Raw image channels of every section ({raw}): "
                     + ", ".join(channels) + ".")
    elif isinstance(channels, dict) and channels:
        lines.append(f"- Raw image channels per section ({raw}): "
                     + "; ".join(f"{name}: {', '.join(names)}"
                                 for name, names in channels.items()) + ".")
    if atlas_channels:
        lines.append("- Atlas layers on this host: " + ", ".join(atlas_channels) + ".")
    if resolution:
        from langslice.core.sizes import MIN_RESOLUTION

        lines.append(f"- Picture size is yours to choose: look's resolution runs from "
                     f"{MIN_RESOLUTION} to {resolution} pixels.")
    return lines


#: Heading of each task's user notes, in the order the tasks are listed.
_TASK_NOTES: tuple[tuple[str, str, str], ...] = (
    ("position", "position", "Positioning"),
    ("transform", "transform", "In-plane alignment"),
    ("nonlinear", "nonlinear", "Nonlinear deformation"),
)


def task_notes(spec: JobSpec) -> list[str]:
    """The user's notes for each task that is on, each under its task's name.

    Verbatim, one ``- `` line per non-empty line of the note. A task that is
    off, or has no note, contributes nothing.
    """
    lines: list[str] = []
    for task, section, title in _TASK_NOTES:
        if not spec.has(task):
            continue
        text = str(getattr(getattr(spec, section), "notes", "") or "")
        body = [line.strip() for line in text.splitlines() if line.strip()]
        if body:
            lines += ["", f"{title} notes from the user:", *(f"- {line}" for line in body)]
    return lines


def ants_fact(tool_names: list[str]) -> str | None:
    """The job statement's line when ANTs cannot run here (None when it can,
    or when no tool of the run needs it)."""
    from langslice.core.deformation import ants_available

    needing = [name for name in ("ants_syn", "trace_borders") if name in tool_names]
    if not needing or ants_available():
        return None
    return (f"- ANTs is not installed on this host, so {' and '.join(needing)} cannot run; "
            "a section left without a deformation is named in submit's left_linear.")


def build_job_statement(
    spec: JobSpec,
    state: StackState,
    *,
    tool_names: list[str],
    species: str,
    pos_lo: float,
    pos_hi: float,
    axis_ends: tuple[str, str],
    channels: list[str] | dict[str, list[str]] | None = None,
    atlas_channels: tuple[str, ...] | None = None,
    max_resolution: int | None = None,
    door: str = "agent",
    auto: bool | None = None,
    gates: bool = True,
) -> str:
    """The system instruction for one run, built from the spec and the state.

    *max_resolution* is the largest picture the driver model takes, the
    top of ``look``'s ``resolution`` at image resolution "auto" (None: the
    OpenAI lanes', ``opening.DEFAULT_IMAGE_LIMIT``). *door* is who reads it
    (``agent``, ``mcp``, ``cli``; the doors' own paragraphs are
    ``doors.statement``'s); *auto* whether the caller sizes each picture
    (None: the spec's image resolution is "auto", the agent CLI always
    does); *gates* False leaves out the look-before-commit gates
    (``position.gated``), which a door without them (the agent CLI) never
    applies. *axis_ends* is ``(low, high)`` from
    :func:`langslice.core.space.slice_axis_ends` — what the two ends of the slicing
    axis are anatomically in THIS atlas.
    """
    del door  # every door reads the same statement; its own lines are added after it
    gated = bool(spec.position.gated and gates)
    jobs: list[str] = []
    if spec.has("position"):
        jobs.append(
            "give every section its own position in millimetres along the "
            "slicing axis, each one inspected and checked against the atlas, "
            "damaged sections included, and report, by filename, each section "
            "after a gap where you conclude the interval between neighbouring "
            "sections is genuinely broken"
        )
    if spec.has("transform"):
        turned = "mirrored or turned" if spec.transform.flip else "turned"
        jobs.append(
            "give every section one in-plane transform onto its atlas section, "
            f"correcting the orientation of any section that is {turned}"
        )
    if spec.has("nonlinear"):
        jobs.append(
            "give every section a deformation onto the atlas, on top of its linear "
            "placement, with the section's stain as the evidence"
            + (", or, for a section you choose to trace, the borders the image model "
               "traces on it"
               if "trace_borders" in tool_names else "")
        )
    job = "; ".join(jobs) if jobs else "review the stack"

    facts = run_facts(spec, state, species=species, pos_lo=pos_lo, pos_hi=pos_hi,
                      axis_ends=axis_ends)
    sized = spec.image_resolution == "auto" if auto is None else auto
    if sized and max_resolution is None:
        from langslice.core.opening import DEFAULT_IMAGE_LIMIT

        max_resolution = DEFAULT_IMAGE_LIMIT[0]
    facts += channel_facts(channels, atlas_channels, max_resolution if sized else None)

    tools = ("Tools (each one's own description says what it does and returns): "
             + ", ".join(f"`{name}`" for name in tool_names) + ".")

    constraints: list[str] = []
    if spec.has("position"):
        constraints.append(
            "- `submit` is refused unless every section has a position of its own, "
            "damaged sections included; a starting position the job gave a section "
            "does not count."
        )
        if gated:
            constraints.append(
                "- `position_sections` is refused for a section that has not been looked "
                "at with `look` in mode overlay or positioning since it was last written."
            )
            constraints.append(
                "- `submit` is refused until `look` in mode positioning has shown every "
                "section after the last `position_sections` write."
            )
        if spec.position.strict_interval:
            constraints.append(
                f"- Every consecutive spacing must be within 10% of "
                f"{state.interval_mm:.3f} mm, and no interval break may be "
                f"reported."
            )
        else:
            constraints.append(
                "- A reported interval break is accepted only at a section "
                "whose interval from the section before it, in the positions "
                "you wrote, exceeds 1.5x the stack's median written spacing."
            )
    if spec.has("transform"):
        constraints.append(
            "- `submit` is refused unless every section carries a transform, "
            "damaged sections included."
        )
        ways = ((["by hand with `interactive_transform`"]
                 if "interactive_transform" in tool_names else [])
                + (["with `elastix_affine`, which leaves the marked regions out of the fit"]
                   if "elastix_affine" in tool_names else []))
        if ways:
            constraints.append(
                "- A damaged section (one with marked regions) still needs a transform "
                "made for its surviving anatomy: set it " + ", or fit it ".join(ways)
                + ". Inspect each overlay against the surviving internal anatomy before "
                "submitting."
            )
    if spec.has("nonlinear"):
        if "trace_borders" in tool_names:
            constraints.append(
                "- `trace_borders` is optional: you decide which sections, if any, to "
                "trace. A trace requires a position and a linear transform."
            )
        if spec.nonlinear.require_deformation:
            constraints.append(
                "- `submit` is refused unless every section carries a deformation applied "
                "at its current placement, damaged sections included; the user requires "
                "one on every section."
            )
        else:
            constraints.append(
                "- `submit` is refused unless every section carries a deformation applied "
                "at its current placement, or is named in its `left_linear` with the "
                "reason its linear placement stands, damaged sections included."
            )
        missing = ants_fact(tool_names)
        if missing:
            constraints.append(missing)
    constraints.append(
        "- Corrections are recorded as data; the user's image files are never "
        "modified."
    )

    method: list[str] = []
    if spec.has("position") and spec.position.playbook:
        # A complete hypothesis first, then one confirmation sweep, one
        # write, targeted re-checks, a review: coaching, only for the models
        # that do not find this method on their own.
        method = [
            "",
            "Method:",
            "- First, from the opening images alone — every section and the "
            "atlas beneath it — form a complete hypothesis: a position for every "
            "section. Look for the structure of how the sections were cut (series "
            "that interleave, missing sections) and use it.",
            "- Then confirm the hypothesis: `look` in mode positioning at each "
            "section with the atlas at its hypothesised position and its neighbours, "
            "walking the stack in order; where the atlas at that position does not "
            "match the section, change the position.",
            "- Write every position in one `position_sections`, then re-check the "
            "sections you were unsure about and correct them.",
            "- Mark damaged sections' lost regions, run `look` in mode positioning "
            "over the whole stack, look again at anything mis-spaced, then `submit`.",
        ]
    elif spec.has("position"):
        method = [
            "",
            "Method:",
            "- Place each section on its own evidence: compare it against "
            "candidate atlas positions before writing, and do not let the "
            "nominal interval stand in for a look.",
            "- After writing, review the whole stack against the atlas, watch "
            "for spacings that differ from their neighbours, and re-check the "
            "sections on either side of any gap before reporting an interval break.",
            "- Submit when the work is complete; address any missing requirements it returns.",
        ]

    if spec.has("transform") and spec.transform.interactive:
        if not method:
            method = ["", "Method:"]
        method.append(
            "- After each automatic fit or manual adjustment, inspect the returned "
            "overlay against surviving internal anatomy. Refine each slice's "
            "alignment, damaged slices included, until no further improvement is "
            "possible with the available transforms. Keep changes only if they "
            "improve the alignment."
        )

    if spec.has("nonlinear"):
        if not method:
            method = ["", "Method:"]
        if "mark_damage" in tool_names:
            method.append(
                "- Before fitting a damaged section, mark the regions it has lost with "
                "`mark_damage`, so that every region it still has drives its fits."
            )
        method.append(
            "- Fit each section whole first, then refine regions. Inspect each fit's "
            "borders against the section's internal anatomy"
            + (" and, where traced, its traced borders"
               if "trace_borders" in tool_names else "")
            + "; undo a fit that does not improve the alignment"
            + ("." if spec.nonlinear.require_deformation else
               ", and name in submit's `left_linear` only a section that no fit improves.")
        )

    return "\n".join(
        [
            f"You are an expert neuroanatomist working on a stack of "
            f"{len(state.slices)} {state.plane} histology sections against the "
            f"{state.atlas} ({species}) reference atlas.",
            "",
            f"Your job: {job}.",
            *task_notes(spec),
            "",
            "Run facts:",
            *facts,
            "",
            tools,
            "",
            "Constraints:",
            *constraints,
            *method,
            "",
            "Work with the tools, then call `submit`.",
        ]
    )


def stack_angles_fact(state: StackState) -> str:
    """The job statement's line on the cutting angles: the stack's one
    angle, or, when the sections differ (a registration supplied per
    section), that each status row carries its own."""
    if state.mixed_angles:
        return ("- Cutting angles: these differ between sections, as supplied; each "
                "status row carries its own cutting_angles_deg. position_sections sets "
                "one angle for every section.")
    pitch, yaw = state.stack_angles
    return f"- Stack-wide cutting angles: pitch {pitch:.2f} deg, yaw {yaw:.2f} deg."


def run_facts(
    spec: JobSpec, state: StackState, *, species: str, pos_lo: float,
    pos_hi: float, axis_ends: tuple[str, str],
) -> list[str]:
    """Shared factual briefing, without any model-specific method advice."""
    axis = _PLANE_AXIS_LABEL.get(state.plane, "AP")
    ordered = state.in_order()
    placed = [s for s in ordered if s.position_mm is not None]
    starting = [s.id for s in ordered if s.position_source == STARTING_POSITION]
    damaged = [s.id for s in ordered if s.damaged]
    inputs = spec.inputs or {}
    noted = [s.id for s in ordered if s.id in (inputs.get("damaged") or {})]
    locked_ids = {str(name) for name in inputs.get("locked") or []}
    locked = [s.id for s in ordered if s.id in locked_ids]

    facts: list[str] = [
        f"- {len(state.slices)} sections, {state.plane} plane, atlas "
        f"{state.atlas} ({species}).",
        f"- Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm along the slicing "
        f"axis, measured from the origin edge of the atlas volume "
        f"({pos_lo:.2f} mm is its first section, {pos_hi:.2f} mm its last).",
        f"- {pos_lo:.2f} mm is the {axis_ends[0]} edge of the volume; "
        f"positions increase toward {axis_ends[1]}.",
        f"- Cutting protocol: nominal section interval {state.interval_mm:.3f} "
        f"mm center-to-center, section thickness {state.thickness_mm:.3f} mm. "
        f"The nominal interval is a protocol value, not a measurement: sections "
        f"can be missing anywhere in the stack, so the spacing between "
        f"neighbours may differ from it.",
        stack_angles_fact(state),
    ]
    if starting:
        facts.append(
            f"- {len(starting)} of {len(state.slices)} sections are at evenly spaced "
            "starting positions the job gave them, in file order; they are not placed "
            "yet (status shows position_source \"default\", the opening labels their "
            "atlas \"start\")."
        )
        if len(placed) > len(starting):
            facts.append(f"- {len(placed) - len(starting)} sections carry a position "
                         "of their own.")
    elif placed:
        facts.append(f"- {len(placed)} of {len(state.slices)} sections carry a position.")
    else:
        facts.append("- No section carries a position yet.")
    if damaged:
        facts.append(f"- Sections with marked damage regions: {', '.join(damaged)}.")
    if noted:
        facts.append(
            "- The user noted damage on these sections (damage_note in status; no "
            f"regions are marked from it): {', '.join(noted)}."
        )
    if locked:
        facts.append(
            "- In-plane alignment of these sections was already done by the "
            "user, so their orientation and transform are locked and count "
            f"as done: {', '.join(locked)}."
            + (" Their positions can still be changed." if spec.has("position") else "")
        )
    if spec.has("position"):
        facts.append("- The stack's order follows the positions: writing positions "
                     "renumbers the sections, so use filenames to name them.")
    else:
        facts.append("- The positions shown are given; this run does not change them.")
    if not spec.has("transform"):
        facts.append(
            "- Existing linear transforms are supplied and fixed for this run, "
            "orientation included."
            if spec.has("nonlinear")
            else "- The orientation shown is fixed; transforms are not part of this run."
        )
    else:
        if not spec.transform.flip:
            facts.append("- Flipping sections is switched off for this run.")
        facts.append(
            "- Alignment canvas: each section is drawn at its TRUE physical size "
            "from its pixel size (read from the file, or given by the host, or "
            "estimated — the tool payload says which), and the atlas at its "
            "voxel size; scale 1.0 is the section's calibrated size."
        )
        facts.append(
            "- Transform frame: rotation and scales act about the pivot (the "
            "canvas centre unless the section's transform holds another), x runs "
            "right and y runs down, shifts are millimetres."
        )
        cap = spec.transform.max_parallel
        if cap < MAX_PARALLEL_TRANSFORMS:
            facts.append(
                f"- Each transform tool call takes at most {cap} "
                f"section{'s' if cap != 1 else ''}."
            )
    # Mirroring belongs to in-plane alignment: the cue is shown only where the
    # agent can flip, never in a positioning-only run.
    cue = spec.transform.hemisphere_cue.strip()
    if cue and spec.has("transform") and spec.transform.flip:
        facts.append(f"- What marks a hemisphere, from the user: {cue}")
    facts.extend(f"- {fact.strip()}" for fact in spec.facts if str(fact).strip())

    return facts
