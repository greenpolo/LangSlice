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
    "validate": "runs the submit checks without submitting; writes nothing.",
    "view_slices": "up to 4 named sections at higher resolution, as corrected.",
    "fetch_atlas": "up to 4 atlas sections at the positions you name, rendered "
    "at the stack's cutting angles.",
    "note": "appends one line to the run notes.",
    "undo": "reverses the last write; one tool call undoes as one step.",
    "redo": "reapplies the write `undo` reversed.",
    "mark_damaged": "records sections whose outline would break an "
    "outline-based fit, with a note each.",
    "unmark_damaged": "clears the damaged flag.",
    "orient_slices": "sets the flip and the rotation of named sections and "
    "returns them rendered as they now stand; a section whose orientation "
    "changes loses its transform.",
    "reorder_slices": "sets the corrected order of the whole stack in one "
    "call, by filename; positions and transforms are kept.",
    "move_slice": "moves one section in the corrected order; positions and "
    "transforms are kept.",
    "compare_placement": "tests candidate positions before you commit to one: "
    "name a section with several positions (or none for its current one) and "
    "it is drawn against the atlas at each, up to 4 pairs per call, one image "
    "per pair, on one physical-scale canvas; e.g. one section at 4.6, 4.8 and "
    "5.0 mm. `mode` "
    "is template (default: the atlas at that position on the section's own "
    "canvas and scale; the section itself is in the opening message), "
    "side_by_side, overlay, checkerboard, outlines or section, "
    "`zoom` is [x0, y0, x1, y1] of the canvas and magnifies (the crop comes "
    "before the resize, so small structures get more pixels), "
    "`template_opacity` is 0..1 and `outlines` is all, outer or none; "
    "writes nothing.",
    "view_stack": "whole-stack review, meant for after the positions are "
    "written and before `submit`: one contact sheet of every section in the "
    "order of its written position with the atlas at that position beneath "
    "it, each captioned with index, filename, position and the distance to "
    "the next, plus a plot of position against corrected index; writes "
    "nothing.",
    "set_positions": "writes positions for one or more sections, clamped to "
    "the atlas range, and returns each newly placed or moved section over "
    "the atlas section at the position it was given.",
    "run_deepslice": "seeds positions (and optionally angles) with DeepSlice.",
    "fit_position": "searches the atlas around one section's current position "
    "and reports the best it found; writes nothing.",
    "set_cutting_angles": "sets the stack-wide cutting angles.",
    "fit_affine": "fits an in-plane affine per section against its atlas "
    "section, writes it as the section's transform, and returns the overlap, "
    "the transform as the same five physical parameters `adjust_transform` "
    "takes, and an image of the section under the atlas outlines at true "
    "physical scale; damaged sections are refused.",
    "adjust_transform": "sets one positioned section's in-plane transform — a "
    "rotation, per-axis scales and millimetre shifts — and returns the section "
    "drawn under it with the atlas outlines at true physical scale; every "
    "call writes and the last call stays. `mode` is overlay, side_by_side, "
    "checkerboard, outlines, section, template or ab (these parameters and "
    "the transform the section carried before, at one crop), `zoom` is "
    "[x0, y0, x1, y1] of the canvas, `template_opacity` is 0..1, `pivot` — "
    "what the rotation and scales turn about — is canvas, tissue or [fx, fy] "
    "of the canvas, `outlines` is all, outer or none, and `note` is a remark "
    "for the record; it does not change the section's flip or rotation.",
    "landmarks": "measures point pairs (a section point and the atlas point it "
    "belongs on, as fractions of the canvas) under given parameters: the "
    "distance in millimetres per pair, their RMS, a transform fitted to them "
    "(similarity from 2 pairs, affine from 3), and the pairs drawn on the "
    "view; writes nothing.",
    "submit": "ends the run.",
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
        facts.append("- Transforms are not part of this run.")
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
        constraints.append(
            "- Damaged sections are refused by `fit_affine`."
        )
        constraints.append(
            "- `submit` is refused unless every section carries a transform, "
            "damaged sections included."
        )
    constraints.append(
        "- Corrections are recorded as data; the user's image files are never "
        "modified."
    )

    method: list[str] = []
    if spec.has("position"):
        method = [
            "",
            "Method:",
            "- Place each section on its own evidence: compare it against "
            "candidate atlas positions before writing, and do not let the "
            "nominal interval stand in for a look. Work in batches: several "
            "sections per `compare_placement` or `set_positions` call cost "
            "less than one call each.",
            "- After writing, review the whole stack against the atlas, watch "
            "for a section that sits out of sequence and for spacings that "
            "differ from their neighbours, and re-check the sections on either "
            "side of any gap before reporting an interval break.",
            "- Validate, then submit.",
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
            "",
            "Work with the tools, then call `submit`.",
        ]
    )
