"""The job statement: job, run facts, one factual line per tool, constraints.

Nothing else. No strategy, no rules of thumb, no warnings about failure modes:
every benchmark failure worth tracing came back to advice the harness put in
front of the model, so the harness reports and the model reasons.

Deliberately atlas- and plane-agnostic — it names no region, no landmark and no
absolute position, because the same text runs against every BrainGlobe atlas,
species and plane.
"""

from __future__ import annotations

from langslice.linear.spec import JobSpec
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
    "fetch_atlas": "up to 4 atlas sections at the positions you name, rendered "
    "at the stack's cutting angles.",
    "note": "appends one line to the run notes.",
    "undo": "reverses the last write; one tool call undoes as one step.",
    "redo": "reapplies the write `undo` reversed.",
    "mark_damaged": "records sections whose outline would break an "
    "outline-based fit, with a note each; damaged=False clears the flag and note.",
    "orient_slices": "sets the flip and the rotation of named sections and "
    "returns them rendered as they now stand; a section whose orientation "
    "changes loses its transform.",
    "reorder_slices": "places the filenames in new_order together, in that "
    "order, after a named section or at start (default). A one-item list moves "
    "one section; a complete list sets the whole order. Unlisted sections keep "
    "their relative order. Positions and transforms are kept.",
    "compare_placement": "tests candidate positions before you commit to one: "
    "name a section with several positions (or none for its current one) and "
    "it is compared with the atlas at each, up to 4 pairs per call; "
    "e.g. one section at 4.6, 4.8 and "
    "5.0 mm. `mode` "
    "is template (default: the atlas at that position on the section's own "
    "canvas and scale; the section itself is in the opening message), "
    "side_by_side (separate original section plus atlas references, "
    "one section per distinct id and one atlas per pair, up to 8 images; "
    "independently tissue-framed, full view only, no outlines/opacity), "
    "overlay, checkerboard, outlines or section (one physical-canvas image per pair), "
    "`zoom` is [x0, y0, x1, y1] of the canvas and magnifies (the crop comes "
    "before the resize, so small structures get more pixels), "
    "`template_opacity` is 0..1, `border_color` is a named color or #RRGGBB "
    "(default yellow), `border_thickness` is 0.25..8 output pixels (default 0.5), "
    "and `outlines` is all, outer or none; border controls affect only drawn "
    "atlas outlines, not separate reference images; "
    "writes nothing.",
    "view_stack": "whole-stack review, meant for after the positions are "
    "written and before `submit`: one contact sheet of every section in the "
    "order of its written position with the atlas at that position beneath "
    "it, each captioned with index, filename, position and the distance to "
    "the next, plus a plot of position against corrected index; writes "
    "nothing.",
    "set_positions": "writes positions for one or more sections, clamped to "
    "the atlas range, and returns a placement image unless that exact section, "
    "position, orientation and cutting-angle combination was already seen in "
    "a full-canvas atlas-bearing placement view.",
    "run_deepslice": "seeds positions (and optionally angles) with DeepSlice.",
    "fit_position": "searches the atlas around one section's current position "
    "and reports the best it found; writes nothing.",
    "set_cutting_angles": "sets the stack-wide cutting angles.",
    "fit_affine": "fits an in-plane affine per section against its atlas "
    "section, writes it as the section's transform, and returns the overlap, "
    "the transform as the same five physical parameters `adjust_transforms` "
    "takes, and an image of the section under the atlas outlines at true "
    "physical scale; damaged sections are refused.",
    "adjust_transforms": "sets and shows one to four independent positioned "
    "sections in one undoable call. Each entry supplies rotation_deg, scale_x, "
    "scale_y, translate_x_mm and translate_y_mm, plus optional mode (overlay, "
    "side_by_side, checkerboard, outlines, section, template or ab), zoom, "
    "template_opacity, pivot, outlines, note, border_color and border_thickness. "
    "ab shows new and previous transforms; side_by_side shows section and atlas. "
    "Results map their images with zero-based image_indexes. Each section may "
    "appear once; inspect before a dependent correction in a later call. "
    "This replaces the complete transform, including any spline or shear.",
    "correct_slice_borders": "uses the fixed image-model border-correction prompt "
    "on one section's existing linear placement, with optional additional_notes "
    "for that slice; returns the raw generated image and extracted borders on "
    "the original. The first result at each placement is saved and reused. "
    "This records an annotation; it does not fit or change the transform.",
    "submit": "checks requirements and ends the run if they pass; otherwise "
    "returns the missing requirements without ending or changing the run.",
}


def build_job_statement(
    spec: JobSpec,
    state: StackState,
    *,
    tool_names: list[str],
    species: str,
    pos_lo: float,
    pos_hi: float,
    axis_ends: tuple[str, str],
) -> str:
    """The system instruction for one run, built from the spec and the state.

    *axis_ends* is ``(low, high)`` from
    :func:`langslice.space.slice_axis_ends` — what the two ends of the slicing
    axis are anatomically in THIS atlas.
    """
    axis = _PLANE_AXIS_LABEL.get(state.plane, "AP")
    jobs: list[str] = []
    if spec.has("reorder"):
        jobs.append(
            "put the stack in the order the sections were cut and correct the "
            "orientation of any section that is mirrored or turned"
        )
    if spec.has("position"):
        jobs.append(
            "give every section its own position in millimetres along the "
            "slicing axis, each one inspected and checked against the atlas, "
            "damaged sections included, and report the corrected indices where "
            "you conclude the interval between neighbouring sections is "
            "genuinely broken"
        )
    if spec.has("transform"):
        jobs.append("give every section one in-plane transform onto its atlas section")
    if spec.has("nonlinear"):
        jobs.append("use the image model to correct every section's placed atlas borders")
    job = "; ".join(jobs) if jobs else "review the stack"

    placed = [s for s in state.in_order() if s.position_mm is not None]
    damaged = [s.id for s in state.in_order() if s.damaged]

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
    if not spec.has("reorder"):
        facts.append("- The order and orientation shown are fixed for this run.")
    elif not spec.reorder.flip:
        facts.append("- Flipping sections is switched off for this run.")
    if not spec.has("position"):
        facts.append("- The positions shown are given; this run does not change them.")
    if not spec.has("transform"):
        facts.append(
            "- Existing linear transforms are supplied and fixed for this run."
            if spec.has("nonlinear") else "- Transforms are not part of this run."
        )
    else:
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
    if spec.reorder.hemisphere_cue.strip():
        facts.append(f"- {spec.reorder.hemisphere_cue.strip()}")
    facts.extend(f"- {fact.strip()}" for fact in spec.facts if str(fact).strip())

    tools = [f"- `{name}`: {TOOL_LINES[name]}" for name in tool_names if name in TOOL_LINES]

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
                "- Damaged sections are refused by `fit_affine`."
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
        constraints.append(
            "- Image correction requires a position and an existing linear transform. "
            "Additional notes supplement the fixed correction prompt; no replacement "
            "prompt or anatomical rejection step is available."
        )
        constraints.append(
            "- Check each sentence of additional_notes against the fixed correction task: "
            "describe this slice's displacement, damage or artifacts without selecting "
            "a different set of atlas borders or omitting regions merely because they are faint."
        )
        constraints.append(
            "- `submit` requires a completed image correction for every section at "
            "its current placement. Completion does not certify anatomical quality."
        )
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
            "- Then confirm the hypothesis: `compare_placement` every section "
            "at its hypothesised position, four sections per call, walking the "
            "stack in order; where the atlas at that position does not match "
            "the section, change the position.",
            "- Write every position in one `set_positions`, then re-check the "
            "sections you were unsure about with `compare_placement` and "
            "correct them.",
            *(
                [
                    "- After the first write, run `fit_position` on every "
                    "section (window 3 mm, angles false) and write its best "
                    "position where the fit disagrees with yours; confirm "
                    "with `compare_placement`."
                ]
                if "fit_position" in tool_names
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

    image_task: list[str] = []
    if spec.has("nonlinear"):
        from typing import cast

        from langslice.registration_tool import correction_instructions
        from langslice.space import Plane

        image_task = [
            "", "Fixed image-model task (image numbers refer to the tool's attachments):",
            correction_instructions(cast(Plane, state.plane)),
        ]

    return "\n".join(
        [
            f"You are an expert neuroanatomist working on a stack of "
            f"{len(state.slices)} {state.plane} histology sections against the "
            f"{state.atlas} ({species}) reference atlas.",
            "",
            f"Your job: {job}.",
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
