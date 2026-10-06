"""The job statement: job, run facts, one factual line per tool, constraints, method.

The statement reports; the model reasons. It gives no rules of thumb and no
warnings about failure modes. The one place for strategy is the short
"Method:" section per task that is on (look before writing, review the
stack, inspect each fit against internal anatomy); ``position.playbook``
replaces the positioning lines with a step-by-step method for models that do
not find one on their own.

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

#: One factual line per tool. Only the tools actually built are listed.
TOOL_LINES: dict[str, str] = {
    "status": "the stack as it stands, one row per section in corrected order; "
    "every write returns only the rows it changed, this returns them all.",
    "view_slices": "up to 4 named sections at higher resolution, as corrected; "
    "view mode channels shows each section's raw channels side by side instead, "
    "unmodified and labelled.",
    "view_atlas": "up to 4 atlas sections at the positions you name, rendered "
    "at the stack's cutting angles.",
    "note": "appends one line to the run notes.",
    "undo": "reverses the last write; one tool call undoes as one step.",
    "redo": "reapplies the write `undo` reversed.",
    "mark_damaged": "records sections whose outline would break an "
    "outline-based fit, with a note each; damaged=False clears the flag and note.",
    "preprocess": "sets how sections look, from their raw channels: channel "
    "weights (the counterstain that lights all tissue, e.g. DAPI or Nissl, "
    "usually deserves the most), CLAHE clip and tiles, ANTs N4 and denoising; "
    "for target view (what you are shown), fit (what a deformable fit reads) "
    "or both, independently, for the stack or named sections; returns each "
    "pictured section (up to 4) twice, labelled: BEFORE this call and AFTER it. "
    "Undoable.",
    "orient_slices": "sets the flip and the rotation of named sections and "
    "returns them rendered as they now stand; a section whose orientation "
    "changes loses its transform.",
    "reorder_slices": "places the filenames in `slices` together, in that "
    "order, after a named section or at start (default). A one-item list moves "
    "one section; a complete list sets the whole order. Unlisted sections keep "
    "their relative order. Positions and transforms are kept.",
    "view_placement": "shows sections in their complete current registration "
    "(position, in-plane transform and any applied deformation), or tests "
    "candidate positions before you commit to one: name a section with several "
    "positions (or none for its current one), up to 4 pairs per call; e.g. one "
    "section at 4.6, 4.8 and 5.0 mm. view modes: template (default: the atlas at "
    "that position on the section's own canvas and scale; the section itself is "
    "in {opening}), overlay, checkerboard, outlines or section (the "
    "section under its registration on that canvas, one image per pair), "
    "stacked (the section as corrected over the atlas, each tissue-framed) or "
    "side_by_side (separate original section plus atlas references, one section "
    "per distinct id and one atlas per pair, up to 8 images; full view only); "
    "writes nothing.",
    "view_stack": "whole-stack review, meant for after the positions are "
    "written and before `submit`: one contact sheet of every section in the "
    "order of its written position with the atlas at that position beneath "
    "it, each captioned with index, filename, position and the distance to "
    "the next, plus a plot of position against corrected index; writes "
    "nothing.",
    "set_positions": "writes positions for one or more sections, clamped to "
    "the atlas range, and returns a placement picture (any `view_placement` "
    "mode; default stacked) unless that exact section, position, orientation "
    "and cutting-angle combination was already seen in a full-canvas "
    "atlas-bearing placement view.",
    "search_position": "searches the atlas around one section's current position "
    "and reports the best it found; writes nothing.",
    "set_cutting_angles": "sets the stack-wide cutting angles.",
    "fit_affine": "fits an in-plane affine per section against its atlas "
    "section, writes it as the section's transform, and returns the overlap, "
    "the transform as the same five physical parameters `adjust_transforms` "
    "takes, and a picture of the section under it at true physical scale. "
    "The default method, elastix, refines the section's current transform by "
    "matching the section's fit appearance against an atlas image (`fit_atlas`: "
    "ara, or nissl where offered), inner anatomy included, starting where the "
    "section is; method silhouette fits "
    "the tissue outline to the atlas outline from scratch. "
    "`exclude` regions (acronyms or ids, descendants included) are removed "
    "from the atlas side, and the tissue the fit lays on them from the "
    "section side; `include` restricts the fit to those regions plus 300 um "
    "(with silhouette, counting only where they reach the outline). Damaged "
    "sections are refused unless regions are given.",
    "adjust_transforms": "sets and shows one to four independent positioned "
    "sections in one undoable call. Each entry supplies rotation_deg, scale_x, "
    "scale_y, translate_x_mm and translate_y_mm, plus optional pivot and note; "
    "one `view` draws every entry (modes overlay, side_by_side, checkerboard, "
    "outlines, section, template or ab). ab shows new and previous transforms; "
    "side_by_side shows section and atlas. Results map their images with "
    "zero-based image_indexes. Each section may appear once; inspect before a "
    "dependent correction in a later call. This replaces the complete "
    "transform, including any shear.",
    "trace_borders": "runs the image-model border-correction prompt on one "
    "section's existing linear placement, with your edited copy of the prompt "
    "for that section. The image call runs in the background and the tool returns "
    "at once; the result is saved for the user; `submit` waits "
    "for running calls, and `fit_deformable` with a traced `fit_section` waits for "
    "it too. The first result at each placement is saved and reused. "
    "This records an annotation; it does not fit or change the transform.",
    "grep_atlas": "looks regions up in the atlas hierarchy by acronym, name "
    "substring or numeric id (at most 40 rows). Each row gives acronym, id, name, "
    "ancestry as acronyms from the root and the number of descendants. With a "
    "section that has a position it also says whether the region, or any "
    "descendant, appears in the atlas plane at that placement. Text only; writes nothing.",
    "fit_deformable": "fits a deformation of the placed atlas onto one or more "
    "positioned, transformed sections with a library engine (ANTs SyN or Elastix "
    "B-spline), on top of the linear placement. Choose what of the section it "
    "reads, `fit_section` {fit_sections}, what of the atlas it reads, "
    "`fit_atlas`, the stiffness, regions to include (fit only them and a margin) and "
    "to exclude (removed from the atlas side; \"CTX:left\" or \"CTX:right\" names "
    "one side of the section as shown), and start (linear, or current to "
    "compose onto the applied deformation, region by region). Several candidates "
    "(2 to 4 setting variants, run concurrently) preview and write nothing; exactly "
    "one setting applies it, reusing an identical earlier result; `keep_linear` "
    "with a reason instead records, without a fit, that a section's linear "
    "placement stands. Returns per result "
    "the final borders drawn on the section image (included regions strong, "
    "excluded in pink), displacement, fold fraction, plausibility flags and the "
    "engine numbers used{trace_picture}; view modes borders (default) or ab "
    "(then what the fit started from). A change "
    "to a section's position, orientation, cutting angles or transform clears "
    "its deformation or keep_linear record. Undoable.",
    "submit": "checks requirements and ends the run if they pass; otherwise "
    "returns the missing requirements without ending or changing the run.",
}


#: ``fit_deformable``'s fit sections, with and without the image model.
_FIT_SECTIONS_TRACED = (
    "(the fit appearance, or the section's trace_borders result at "
    "this placement: traced_borders as named regions, traced_lines as lines; a call "
    "waits for a trace that is still running)"
)
_FIT_SECTIONS_STAIN = "(the fit appearance)"


#: Where each door's opening pictures are, as the tool lines name them
#: (:data:`langslice.doors.declarations.DOORS`: the ADK agent's seed message,
#: the MCP door's ``show_stack`` pages, the agent CLI's ``brief`` files).
OPENING_PLACES: dict[str, str] = {
    "agent": "the opening message",
    "mcp": "the opening pictures (show_stack)",
    "cli": "the opening pictures (brief)",
}

#: Tool lines worded for the agent CLI, which answers an image-model verb
#: once its call has landed (``langslice job FOLDER trace_borders``) unless
#: it runs with ``--background``.
_CLI_LINES: dict[str, tuple[str, str]] = {
    "trace_borders": (
        "The image call runs in the background and the tool returns "
        "at once; the result is saved for the user; `submit` waits "
        "for running calls, and `fit_deformable` with a traced `fit_section` waits for "
        "it too.",
        "The command answers once the image call has landed (with --background it "
        "answers at once and `wait` collects the answer); the result is saved for the "
        "user, and `submit` and `fit_deformable` with a traced `fit_section` wait for "
        "a call still running.",
    ),
}


def tool_line(name: str, spec: JobSpec, door: str = "agent") -> str:
    """The job statement's line for one tool, worded for this run's settings
    and for the *door* that reads it (``agent``, ``mcp`` or ``cli``)."""
    line = TOOL_LINES[name]
    if name == "view_placement":
        line = line.format(opening=OPENING_PLACES[door])
    if door == "cli" and name in _CLI_LINES:
        old, new = _CLI_LINES[name]
        line = line.replace(old, new)
    if name == "fit_deformable":
        traced = spec.nonlinear.uses_image_model
        line = line.format(
            fit_sections=_FIT_SECTIONS_TRACED if traced else _FIT_SECTIONS_STAIN,
            trace_picture=(", plus each traced section's trace drawn on the section"
                           if traced else ""),
        )
    return line


def deformable_engine_fact(engine: str, *, traced: bool = True) -> str:
    """The job statement's line on the deformable-fit engine and its availability.

    *traced* (the image model is on) names what a missing ANTs costs the
    traced section images; without the image model there are none to name.
    """
    from langslice.core.deformation import ants_available

    if engine == "either":
        if ants_available():
            return ("`fit_deformable` engine: ants or elastix, your choice per call "
                    "(default ants).")
        if not traced:
            return "`fit_deformable` engine: elastix (ANTs is not installed on this host)."
        return ("`fit_deformable` engine: elastix (ANTs is not installed on this host, so "
                "traced_borders is unavailable).")
    if engine == "ants" and not ants_available():
        return ("`fit_deformable` engine: ants, set by the user, but ANTs is not installed on "
                "this host, so `fit_deformable` cannot run.")
    return f"`fit_deformable` engine: {engine}, set by the user for this run."


#: Tools that return pictures and so take the shared ``view`` argument.
PICTURE_TOOLS: tuple[str, ...] = (
    "view_slices", "view_atlas", "view_placement", "view_stack", "set_positions",
    "orient_slices", "fit_affine", "adjust_transforms", "preprocess", "fit_deformable",
)

#: One line per atlas channel, as the job statement describes it.
ATLAS_CHANNEL_LINES: dict[str, str] = {
    "ara": "the atlas's reference image, the template its regions were drawn on",
    "nissl": "a Nissl-stained reference, ABBA's cached Allen atlas",
    "borders": "the atlas regions, drawn as lines",
}


def display_facts(
    ctx: Workspace, state: StackState,
) -> dict[str, Any]:
    """The job statement's display facts: raw channels and atlas channels here.

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


def display_lines(
    tool_names: list[str],
    *,
    channels: list[str] | dict[str, list[str]] | None = None,
    atlas_channels: tuple[str, ...] | None = None,
    resolution: int | None = None,
) -> list[str]:
    """``view``, described once, with this run's raw and atlas channels.

    Tool docstrings name their own modes and point here. *resolution* (the
    host left picture size to the agent, level "auto": the largest picture
    the driver model takes) adds the ``resolution`` key and its range;
    otherwise picture size is never mentioned.
    """
    if not any(name in PICTURE_TOOLS for name in tool_names):
        return []
    from langslice.core.display import DEFAULT_ATLAS_OPACITY, DEFAULT_BORDER_THICKNESS

    lines = [
        "- Picture options: every tool that returns a picture takes them in ONE "
        "argument, `view` (an object; this call only, nothing is stored). Its keys: "
        "`mode` (each tool lists its own; the first is its default); "
        "`channels`, what of the section is shown: one or more raw channel names "
        "(each stretched by percentile; one is shown in gray, several are added in "
        "distinct colours, and the reply's `view.channel_colors` says which is "
        "which), or one version, [\"view\"] (your viewing appearance, the default) or "
        "[\"fit\"] (the appearance registration reads); "
        "`atlas_channels`, what of the atlas is shown, any of the atlas channels "
        "below: images are drawn under the section at `atlas_opacity` (0..1, "
        f"default {DEFAULT_ATLAS_OPACITY:g}) in overlay, ab, outlines and borders modes "
        "and as the atlas picture in the others (two images are added in two "
        "colours); leave out borders for no lines. Defaults: [borders] in overlay, ab, "
        "outlines and borders; [ara] in template, stacked and the framed "
        "side_by_side of view_placement/set_positions; [ara, borders] in checkerboard "
        "and the physical side_by_side; "
        "`regions` (atlas acronyms or ids, descendants included, \"CTX:left\" / "
        "\"CTX:right\" for one side of the section: their borders at full strength, "
        "other lines faint); `outlines` (all or outer: which lines borders draws); "
        f"`border_color` (named or #RRGGBB, default yellow); `border_thickness` "
        f"(0.25..8 output pixels, default {DEFAULT_BORDER_THICKNESS:g}); `zoom` "
        "([x0, y0, x1, y1] fractions; the crop comes before the resize, so it "
        "magnifies); `deformation` (view_placement and set_positions: applied, the "
        "default, draws the section's applied deformation; none, the linear placement "
        "alone). A key that means nothing for a tool or a mode is refused with the "
        "reason, and so is any unknown argument.",
    ]
    if resolution:
        from langslice.core.sizes import AUTO_RESOLUTION, MIN_RESOLUTION, PICTURE_EDGES

        low, high = MIN_RESOLUTION, resolution
        lines.append(
            "- `view` also takes `resolution`: the long edge in pixels of each picture "
            f"the call returns (of each section tile in `view_stack`), {low} to {high}; "
            f"0 or omitted is {PICTURE_EDGES[AUTO_RESOLUTION][1]}. A picture is never "
            "drawn larger than the image it comes from."
        )
    if atlas_channels:
        lines.append(
            "- Atlas channels on this host: "
            + "; ".join(f"{name} ({ATLAS_CHANNEL_LINES.get(name, name)})"
                        for name in atlas_channels) + "."
        )
    raw = ("the planes of each section's file as read; the names come from the file or "
           "the host and may not say which is the stain, so `view_slices` with mode "
           "channels shows each one, unmodified, side by side")
    if isinstance(channels, list) and channels:
        lines.append(f"- Raw image channels of every section ({raw}): "
                     + ", ".join(channels) + ".")
    elif isinstance(channels, dict) and channels:
        lines.append(
            f"- Raw image channels per section ({raw}): "
            + "; ".join(f"{name}: {', '.join(names)}" for name, names in channels.items())
            + "."
        )
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
    top of ``view.resolution`` at image resolution "auto" (None: the OpenAI
    lanes', ``opening.DEFAULT_IMAGE_LIMIT``). *door* is who reads it
    (``agent``, ``mcp``, ``cli``: where the opening pictures are, how a long
    call answers); *auto* whether the caller sizes each picture
    (``view.resolution``; None: the spec's image resolution is "auto", the
    agent CLI always does); *gates* False leaves out the look-before-commit
    gates (``position.gated``), which a door without them (the agent CLI)
    never applies. *axis_ends* is ``(low, high)`` from
    :func:`langslice.core.space.slice_axis_ends` — what the two ends of the slicing
    axis are anatomically in THIS atlas.
    """
    jobs: list[str] = []
    if spec.has("reorder"):
        jobs.append("put the stack in the order the sections were cut")
    if spec.has("position"):
        jobs.append(
            "give every section its own position in millimetres along the "
            "slicing axis, each one inspected and checked against the atlas, "
            "damaged sections included, and report the corrected indices where "
            "you conclude the interval between neighbouring sections is "
            "genuinely broken"
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
               if spec.nonlinear.uses_image_model else "")
        )
    job = "; ".join(jobs) if jobs else "review the stack"

    facts = run_facts(spec, state, species=species, pos_lo=pos_lo, pos_hi=pos_hi,
                      axis_ends=axis_ends)

    tools = [f"- `{name}`: {tool_line(name, spec, door)}" for name in tool_names
             if name in TOOL_LINES]
    sized = spec.image_resolution == "auto" if auto is None else auto
    if sized and max_resolution is None:
        from langslice.core.opening import DEFAULT_IMAGE_LIMIT

        max_resolution = DEFAULT_IMAGE_LIMIT[0]
    tools += display_lines(
        tool_names, channels=channels, atlas_channels=atlas_channels,
        resolution=max_resolution if sized else None,
    )

    constraints: list[str] = []
    if spec.has("position"):
        constraints.append(
            "- `submit` is refused unless every section has a position, "
            "damaged sections included."
        )
        constraints.append(
            "- `submit` is refused unless the positions run one way along the "
            "corrected order."
        )
        if spec.position.gated and gates:
            constraints.append(
                "- `set_positions` is refused for a section that has not been "
                "compared since it was last written."
            )
            constraints.append(
                "- `submit` is refused until `view_stack` has run after the last "
                "`set_positions` write."
            )
        if spec.position.strict_interval:
            constraints.append(
                f"- Every consecutive spacing must be within 10% of "
                f"{state.interval_mm:.3f} mm, and no interval break may be "
                f"reported."
            )
        else:
            constraints.append(
                "- A reported interval break is accepted only at an index "
                "where the interval between the positions you wrote exceeds "
                "1.5x the stack's median written spacing."
            )
    if spec.has("transform"):
        if spec.transform.automatic:
            constraints.append(
                "- Damaged sections are refused by `fit_affine` unless it is given "
                "regions to include or exclude."
            )
        constraints.append(
            "- `submit` is refused unless every section carries a transform, "
            "damaged sections included."
        )
        constraints.append(
            "- Every damaged section requires a non-identity manual (interactive) "
            "transform based on surviving anatomy; marking damage does not exempt "
            "it from alignment. An automatic fit or an identity transform does "
            "not satisfy this requirement."
        )
        if spec.transform.interactive:
            constraints.append(
                "- Align damaged sections with "
                "`adjust_transforms`; "
                "inspect the overlays before submitting."
            )
        else:
            constraints.append(
                "- Interactive transforms are disabled; unresolved damaged "
                "sections require the host to enable them before submission."
            )
    if spec.has("nonlinear"):
        traced = spec.nonlinear.uses_image_model
        if traced:
            constraints.append(
                "- `trace_borders` is optional: you decide which sections, if any, to "
                "trace. A trace requires a position and an existing linear transform. "
                "Edit the base image prompt below for each section you trace: its format "
                "is good and tested, so make small changes or add a special instruction "
                "for that particular section."
            )
        constraints.append(
            "- `submit` is refused unless every section carries a deformation applied "
            "at its current placement, or a `keep_linear` reason saying its linear "
            "placement stands, damaged sections included."
        )
        if "fit_deformable" in tool_names:
            constraints.append(
                "- " + deformable_engine_fact(spec.nonlinear.engine, traced=traced))
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
            "atlas reference strips — form a complete hypothesis: the corrected order of "
            "the whole stack and a position for every section. Look for the "
            "structure of how the sections were cut (series that interleave, "
            "missing sections) and use it.",
            "- Then confirm the hypothesis: `view_placement` every section "
            "at its hypothesised position, four sections per call, walking the "
            "stack in order; where the atlas at that position does not match "
            "the section, change the position.",
            "- Write every position in one `set_positions`, then re-check the "
            "sections you were unsure about with `view_placement` and "
            "correct them.",
            *(
                [
                    "- After the first write, run `search_position` on every "
                    "section (window 3 mm, angles false) and write its best "
                    "position where the fit disagrees with yours; confirm "
                    "with `view_placement`."
                ]
                if "search_position" in tool_names
                else []
            ),
            "- Mark damaged sections with a note each, set the order, run "
            "`view_stack`, look again at anything out of sequence or "
            "mis-spaced, then `submit`.",
        ]
    elif spec.has("position"):
        method = [
            "",
            "Method:",
            "- Place each section on its own evidence: compare it against "
            "candidate atlas positions before writing, and do not let the "
            "nominal interval stand in for a look.",
            "- After writing, review the whole stack against the atlas, watch "
            "for a section that sits out of sequence and for spacings that "
            "differ from their neighbours, and re-check the sections on either "
            "side of any gap before reporting an interval break.",
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

    if spec.has("nonlinear") and "fit_deformable" in tool_names:
        if not method:
            method = ["", "Method:"]
        method.append(
            "- Let every region a section still has drive its deformable fit, "
            "damaged sections included: exclude the regions it has lost rather "
            "than restricting the fit to a few that survive."
        )
        method.append(
            "- Compare candidates before applying, and inspect each returned fit's "
            "borders against the section's internal anatomy"
            + (" and, where traced, its traced borders"
               if spec.nonlinear.uses_image_model else "")
            + ". Apply the fit that matches best; keep the linear placement only "
            "where no fit improves on it."
        )

    image_task: list[str] = []
    if spec.has("nonlinear") and spec.nonlinear.uses_image_model:
        from typing import cast

        from langslice.core.nonlinear.registration_tool import correction_instructions
        from langslice.core.space import Plane

        image_task = [
            "", "Base image-model prompt (image numbers refer to the tool's attachments):",
            correction_instructions(cast(Plane, state.plane), spec.nonlinear.provider),
        ]

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
            "Tools:",
            *tools,
            "",
            "Constraints:",
            *constraints,
            *method,
            *image_task,
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
                "status row carries its own cutting_angles_deg. set_cutting_angles sets "
                "one angle for every section.")
    pitch, yaw = state.stack_angles
    return f"- Stack-wide cutting angles: pitch {pitch:.2f} deg, yaw {yaw:.2f} deg."

def run_facts(
    spec: JobSpec, state: StackState, *, species: str, pos_lo: float,
    pos_hi: float, axis_ends: tuple[str, str],
) -> list[str]:
    """Shared factual briefing, without any model-specific method advice."""
    axis = _PLANE_AXIS_LABEL.get(state.plane, "AP")
    placed = [s for s in state.in_order() if s.position_mm is not None]
    damaged = [s.id for s in state.in_order() if s.damaged]
    inputs = spec.inputs or {}
    host_damaged = [s.id for s in state.in_order() if s.id in (inputs.get("damaged") or {})]
    locked_ids = {str(name) for name in inputs.get("locked") or []}
    locked = [s.id for s in state.in_order() if s.id in locked_ids]

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
    facts.append(
        f"- {len(placed)} of {len(state.slices)} sections carry a position."
        if placed
        else "- No section carries a position yet."
    )
    if damaged:
        facts.append(f"- Sections marked damaged: {', '.join(damaged)}.")
    if host_damaged:
        facts.append(
            f"- The user marked these sections damaged, and that flag cannot be "
            f"cleared: {', '.join(host_damaged)}."
        )
    if locked:
        facts.append(
            "- In-plane alignment of these sections was already done by the "
            "user, so their orientation and transform are locked and count "
            f"as done: {', '.join(locked)}."
            + (" Their positions can still be changed." if spec.has("position") else "")
        )
    if not spec.has("reorder"):
        facts.append("- The order shown is fixed for this run.")
    if not spec.has("position"):
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
        # The alignment frame, as facts: these lived in the deleted
        # sub-session prompt and the fold-in dropped them.
        facts.append(
            "- Alignment canvas: each section is drawn at its TRUE physical size "
            "from its pixel size (read from the file, or given by the host, or "
            "estimated — the tool payload says which), and the atlas at its "
            "voxel size; scale 1.0 is the section's calibrated size."
        )
        facts.append(
            "- Transform frame: rotation and scales act about the pivot (the "
            "canvas centre unless another is chosen), x runs right and y runs "
            "down, shifts are millimetres."
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
