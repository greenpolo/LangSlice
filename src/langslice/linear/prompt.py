"""The job statement: job, run facts, one factual line per tool, constraints.

Nothing else. No strategy, no rules of thumb, no warnings about failure modes:
every benchmark failure worth tracing came back to advice the harness put in
front of the model, so the harness reports and the model reasons.

Deliberately atlas- and plane-agnostic — it names no region, no landmark and no
absolute position, because the same text runs against every BrainGlobe atlas,
species and plane.
"""

from __future__ import annotations

from langslice.linear.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.linear.state import StackState

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

#: One factual line per tool. Only the tools actually built are listed.
TOOL_LINES: dict[str, str] = {
    "status": "the stack as it stands, one row per section in corrected order; "
    "every write returns only the rows it changed, this returns them all.",
    "view_slices": "up to 4 named sections at higher resolution, as corrected.",
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
    "or both, independently, for the stack or named sections; returns the "
    "sections as they now look. Undoable.",
    "orient_slices": "sets the flip and the rotation of named sections and "
    "returns them rendered as they now stand; a section whose orientation "
    "changes loses its transform.",
    "reorder_slices": "places the filenames in new_order together, in that "
    "order, after a named section or at start (default). A one-item list moves "
    "one section; a complete list sets the whole order. Unlisted sections keep "
    "their relative order. Positions and transforms are kept.",
    "view_placement": "shows sections in their full current placement (position "
    "and in-plane transform), or tests candidate positions before you commit "
    "to one: name a section with several positions (or none for its current "
    "one), up to 4 pairs per call; e.g. one section at 4.6, 4.8 and 5.0 mm. "
    "`mode` is template (default: the atlas at that position on the section's "
    "own canvas and scale; the section itself is in the opening message), "
    "stacked (the section over the atlas, each tissue-framed), side_by_side "
    "(separate original section plus atlas references, one section per "
    "distinct id and one atlas per pair, up to 8 images; full view only), "
    "overlay, checkerboard, outlines or section (one physical-canvas image per "
    "pair); writes nothing.",
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
    "run_deepslice": "seeds positions (and optionally angles) with DeepSlice.",
    "search_position": "searches the atlas around one section's current position "
    "and reports the best it found; writes nothing.",
    "set_cutting_angles": "sets the stack-wide cutting angles.",
    "fit_affine": "fits an in-plane affine per section against its atlas "
    "section, writes it as the section's transform, and returns the overlap, "
    "the transform as the same five physical parameters `adjust_transforms` "
    "takes, and a picture of the section under it at true physical scale. "
    "The fit matches outlines only: the tissue's against the atlas's. "
    "`exclude` regions (acronyms or ids, descendants included) are removed "
    "from the atlas side, and the tissue the fit lays on them from the "
    "section side; `include` restricts the fit to those regions plus 300 um, "
    "which counts only where they reach the outline. Damaged sections are "
    "refused unless regions are given.",
    "adjust_transforms": "sets and shows one to four independent positioned "
    "sections in one undoable call. Each entry supplies rotation_deg, scale_x, "
    "scale_y, translate_x_mm and translate_y_mm, plus optional pivot, note "
    "and display options (mode: overlay, side_by_side, checkerboard, outlines, "
    "section, template or ab). ab shows new and previous transforms; "
    "side_by_side shows section and atlas. Results map their images with "
    "zero-based image_indexes. Each section may appear once; inspect before a "
    "dependent correction in a later call. This replaces the complete "
    "transform, including any spline or shear.",
    "trace_borders": "runs the image-model border-correction prompt on one "
    "section's existing linear placement, with your edited copy of the prompt "
    "for that section. The image call runs in the background and the tool returns "
    "at once; the result is saved for the user and checked at submit, which waits "
    "for running calls, and `fit_deformable` with a traced section image waits for "
    "it too. The first result at each placement is saved and reused. "
    "This records an annotation; it does not fit or change the transform.",
    "grep_atlas": "looks regions up in the atlas hierarchy by acronym, name "
    "substring or numeric id (at most 40 rows). Each row gives acronym, id, name, "
    "ancestry as acronyms from the root and the number of descendants. With a "
    "section that has a position it also says whether the region, or any "
    "descendant, appears in the atlas plane at that placement. Text only; writes nothing.",
    "fit_deformable": "fits a deformation of the placed atlas onto one or more "
    "positioned, transformed sections with a library engine (ANTs SyN or Elastix "
    "B-spline), on top of the linear placement. Choose the section image "
    "{section_images}, the atlas "
    "image, stiffness, detail, regions to include (fit only them and a margin) and "
    "to exclude (removed from the atlas side), and start (linear, or current to "
    "compose onto the applied deformation, region by region). Several candidates "
    "(2 to 4 setting variants, run concurrently) preview and write nothing; exactly "
    "one setting applies it, reusing an identical earlier result; `keep_linear` "
    "with a reason instead records, without a fit, that a section's linear "
    "placement stands. Returns per result "
    "the final borders drawn on the section image (included regions strong, "
    "excluded in pink), displacement, fold fraction, plausibility flags and the "
    "engine numbers used{trace_picture}; display options mode (borders or ab), zoom, "
    "atlas_opacity, regions, outlines, border_color, border_thickness. A change "
    "to a section's position, orientation, cutting angles or transform clears "
    "its deformation or keep_linear record. Undoable.",
    "submit": "checks requirements and ends the run if they pass; otherwise "
    "returns the missing requirements without ending or changing the run.",
}


#: ``fit_deformable``'s section images, with and without the image model.
_SECTION_IMAGES_TRACED = (
    "(the fit appearance, a raw channel, or the section's trace_borders result at "
    "this placement: traced_borders as named regions, traced_lines as lines; a call "
    "waits for a trace that is still running)"
)
_SECTION_IMAGES_STAIN = "(the fit appearance or a raw channel)"


def tool_line(name: str, spec: JobSpec) -> str:
    """The job statement's line for one tool, worded for this run's settings."""
    line = TOOL_LINES[name]
    if name == "fit_deformable":
        traced = spec.nonlinear.uses_image_model
        line = line.format(
            section_images=_SECTION_IMAGES_TRACED if traced else _SECTION_IMAGES_STAIN,
            trace_picture=(", plus each traced section's trace drawn on the section"
                           if traced else ""),
        )
    return line


def deformable_engine_fact(engine: str, *, traced: bool = True) -> str:
    """The job statement's line on the deformable-fit engine and its availability.

    *traced* (the image model is on) names what a missing ANTs costs the
    traced section images; without the image model there are none to name.
    """
    from langslice.linear.deformation import ants_available

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


#: Tools that return pictures and so take the shared display options.
PICTURE_TOOLS: tuple[str, ...] = (
    "view_slices", "view_atlas", "view_placement", "view_stack", "set_positions",
    "orient_slices", "fit_affine", "adjust_transforms", "preprocess",
)


def display_lines(
    tool_names: list[str],
    *,
    channels: list[str] | dict[str, list[str]] | None = None,
    atlas_images: tuple[str, ...] | None = None,
    resolution: bool = False,
) -> list[str]:
    """The shared display options, once, with this run's channels and atlas images.

    *resolution* (the host left picture size to the agent, level "auto") adds
    the ``resolution`` argument; otherwise picture size is never mentioned.
    """
    if not any(name in PICTURE_TOOLS for name in tool_names):
        return []
    lines = [
        "- Every tool that returns a picture also takes the same display "
        "options, for that call only: `mode` (per tool), `zoom` ([x0, y0, x1, "
        "y1] fractions; the crop comes before the resize, so it magnifies), "
        "`section_image` (current, or one raw channel by name), `atlas_image`, "
        "`atlas_opacity` (0..1, the atlas image under the lines in overlay), "
        "`regions` (atlas acronyms or ids, descendants included: only their "
        "borders at full strength, the outlines layer faint behind them), "
        "`outlines` (all, outer or none), `border_color` (named or #RRGGBB) "
        "and `border_thickness` (0.25..8 output pixels).",
    ]
    if resolution:
        from langslice.linear.render import AUTO_RESOLUTION, PICTURE_EDGES, RESOLUTION_RANGE

        low, high = RESOLUTION_RANGE
        deformable = " (`fit_deformable` included)" if "fit_deformable" in tool_names else ""
        lines.append(
            f"- Every tool that returns a picture{deformable} also "
            "takes `resolution`: the long edge in pixels of each picture the call "
            f"returns (of each section tile in `view_stack`), {low} to {high}; 0 or "
            f"omitted is {PICTURE_EDGES[AUTO_RESOLUTION][1]}. A picture is never "
            "drawn larger than the image it comes from."
        )
    if atlas_images:
        lines.append(
            "- Atlas images on this host: " + ", ".join(atlas_images)
            + " (ara: the atlas reference image; borders: the atlas regions as "
            "lines" + ("; nissl: the Allen Nissl stain" if "nissl" in atlas_images else "")
            + ")."
        )
    if isinstance(channels, list) and channels:
        lines.append("- Raw channels of every section: " + ", ".join(channels) + ".")
    elif isinstance(channels, dict) and channels:
        lines.append(
            "- Raw channels per section: "
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
    atlas_images: tuple[str, ...] | None = None,
) -> str:
    """The system instruction for one run, built from the spec and the state.

    *axis_ends* is ``(low, high)`` from
    :func:`langslice.space.slice_axis_ends` — what the two ends of the slicing
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
            "placement, with the section's stain"
            + (" and the borders the image model traces on it"
               if spec.nonlinear.uses_image_model else "")
            + " as the evidence"
        )
    job = "; ".join(jobs) if jobs else "review the stack"

    facts = run_facts(spec, state, species=species, pos_lo=pos_lo, pos_hi=pos_hi,
                      axis_ends=axis_ends)

    tools = [f"- `{name}`: {tool_line(name, spec)}" for name in tool_names
             if name in TOOL_LINES]
    tools += display_lines(tool_names, channels=channels, atlas_images=atlas_images,
                           resolution=spec.image_resolution == "auto")

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
        if spec.position.gated:
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
                "- Image correction requires a position and an existing linear transform. "
                "Edit the base image prompt below for each section: its format is good and "
                "tested, so make small changes or add a special instruction for that "
                "particular section."
            )
            constraints.append(
                "- `submit` requires a completed image correction for every section at "
                "its current placement. Completion does not certify anatomical quality."
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
        # GPT-6 Astra's own method, read off its run-8 trace (M11, 34 of 36
        # within 0.25 mm): a complete hypothesis first, then one confirmation
        # sweep, one write, targeted re-checks, a review. Coaching text: only
        # for the models that do not find this on their own.
        method = [
            "",
            "Method:",
            "- First, from the opening images alone — every section and the "
            "atlas strip — form a complete hypothesis: the corrected order of "
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
            "alignment until no further improvement is possible with the available "
            "transforms. Keep changes only if they improve the alignment."
        )

    if spec.has("nonlinear") and "fit_deformable" in tool_names:
        if not method:
            method = ["", "Method:"]
        method.append(
            "- After each deformable fit, inspect the returned borders against the "
            "section's internal anatomy"
            + (" and its traced borders" if spec.nonlinear.uses_image_model else "")
            + ". Apply the fit that matches best; keep the linear placement only "
            "where no fit improves on it."
        )

    image_task: list[str] = []
    if spec.has("nonlinear") and spec.nonlinear.uses_image_model:
        from typing import cast

        from langslice.registration_tool import correction_instructions
        from langslice.space import Plane

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
        f"- Stack-wide cutting angles: pitch {state.pitch_deg:.2f} deg, yaw "
        f"{state.yaw_deg:.2f} deg.",
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
