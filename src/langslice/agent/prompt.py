"""The job statement: the job, the run facts, the tools by name, the constraints, the method.

The statement reports; the model reasons. Each tool is described once, by
its own description (:mod:`langslice.doors.declarations`); the statement
names the tools and describes none, except the deformable methods when a job
has more than one (:data:`DEFORMABLE_METHODS`: what each published method is,
so that none reads as the default). It gives no rules of thumb and no
warnings about failure modes. The short "Method:" section per task that is
on holds facts and first principles only, never an order of steps (compare
before writing, review the stack, judge each fit against internal anatomy,
what a fit can and cannot correct, what makes a fit work); ``position.playbook``
replaces the positioning lines with a step-by-step method for models that
do not find one on their own.

Deliberately atlas- and plane-agnostic — it names no region, no landmark and no
absolute position, because the same text runs against every BrainGlobe atlas,
species and plane.

A line that names a tool or an option is written with what it needs
(:mod:`langslice.core.needs`) and rendered for the job's disabled list
(``JobSpec.disabled``); the user's own words are never rendered.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from langslice.core import frozen as freeze
from langslice.core.needs import render
from langslice.core.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.core.state import StackState
from langslice.job.job import INTERVAL_BREAK_MIN_RATIO, STRICT_INTERVAL_TOLERANCE

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

#: The deformable methods a job can hold (the native fit and the in-tree engine
#: plugins), in the order the Method lists them, each with what it is: the
#: claims are the methods' own papers' (Nash approved the wording 2026-10-21).
DEFORMABLE_METHODS: dict[str, str] = {
    "fireants_syn": (
        "ANTs SyN (Avants et al. 2008), the standard symmetric diffeomorphic registration "
        "in neuroimaging, run on the GPU by FireANTs. It matches brightness (mutual "
        "information or local correlation) and optionally edges, takes masks, and "
        "`restrict_to` refines single regions."),
    "stalign_lddmm": (
        "STalign (Clifton et al. 2023), LDDMM built to align single-cell spatial "
        "transcriptomics and histology to the Allen CCF. It fits a contrast map between "
        "the two images and sorts every pixel into tissue, background or artifact "
        "itself, so partial sections and debris count less."),
    "spomialign": (
        "SPOmiAlign (Wang et al. 2026), built to align spatial-omics sections across "
        "modalities. It matches points with RoMa, a pretrained dense feature matcher, "
        "instead of comparing brightness, so it needs only shared tissue shape and "
        "structure, not similar staining; seconds per call. Its authors report it "
        "degrades when the two images share little tissue and in repetitive or "
        "ambiguous tissue."),
    "emlddmm_stack": (
        "emlddmm (Tward et al. 2025), built to bring whole-brain series of any modality "
        "into the Allen CCF. It fits one smooth 3-D warp of the atlas to the whole stack, "
        "with each section's own mounting, and sorts out missing tissue and artifacts "
        "itself. It suits dense, evenly spaced series and gives placements that agree "
        "across the stack; it makes no single-section bends."),
}

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

def display_facts(
    ctx: Workspace, state: StackState,
) -> dict[str, Any]:
    """The job statement's display facts: the raw channels, the atlas layers
    here, and the atlas image the fits' default ``nissl`` is.

    ``channels`` is one list when every section shares it, else a mapping
    filename -> names. Unreadable files contribute nothing (the seed will
    report them).
    """
    from langslice.core.atlas.images import default_image
    from langslice.core.display import available_atlas_channels

    named: dict[str, list[str]] = {}
    for record in state.in_order():
        try:
            named[record.id] = list(ctx.section_channels(record.id)[0])
        except Exception:
            continue
    distinct = {tuple(names) for names in named.values()}
    channels: Any = list(next(iter(distinct))) if len(distinct) == 1 else named
    return {"channels": channels or None, "atlas_channels": available_atlas_channels(ctx),
            "nissl_image": default_image(ctx.atlas)}


#: The run fact on section images, written with what it needs
#: (:mod:`langslice.core.needs`); ``{opening}`` is how ``preprocessed`` was made.
SECTION_IMAGES_FACT = (
    "- Section images (status lists them[[?list_files , list_files shows their files under "
    "sections/<name>/images/]]): the raw channels of each file[[?make_image , and the images "
    "made from them with make_image]]. `preprocessed` was made when the job opened: "
    "{opening}. Every fit[[?trace_borders  and the image model read|| reads]] a section image "
    "by name, and a picture of it is exactly what they read."
)
#: The one fact about a channel's signal the agent cannot see from the atlas side.
SPARSE_SIGNAL_FACT = ("- A channel whose signal is sparse (a tracer, a reporter) shows little "
                      "of the anatomy the atlas draws.")


def channel_facts(
    channels: list[str] | dict[str, list[str]] | None,
    atlas_channels: tuple[str, ...] | None,
    resolution: int | None,
    *,
    opening: str = "",
    nissl_image: str | None = None,
    off: Collection[str] = (),
) -> list[str]:
    """The run facts on what the pictures and fits read: the raw channels, the
    section images (*opening*: how ``preprocessed`` was made), the atlas
    images (*nissl_image*: what the fits' default ``nissl`` is) and, where the
    agent sizes pictures, the largest ``look`` picture."""
    from langslice.core.atlas.images import NISSL

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
    if opening:
        lines.append(render(SECTION_IMAGES_FACT.replace("{opening}", opening), off))
        lines.append(SPARSE_SIGNAL_FACT)
    images = [name for name in atlas_channels or () if name not in ("borders", NISSL)]
    if images:
        lines.append("- Atlas images: " + ", ".join(images) + "."
                     + (f" The fits' default `{NISSL}` is `{nissl_image}` on this atlas."
                        if nissl_image else ""))
    if resolution:
        from langslice.core.sizes import MIN_RESOLUTION

        lines.append(f"- Picture size is yours to choose: look's resolution runs from "
                     f"{MIN_RESOLUTION} to {resolution} pixels.")
    return lines


def atlas_facts(atlas_name: str, atlas_channels: tuple[str, ...] | None) -> list[str]:
    """The run facts on what the atlas and its images are: BrainGlobe's own
    description of the atlas (:mod:`langslice.core.atlas.details`) and, where
    the atlas is offered it, what the borrowed Nissl is."""
    from langslice.core.atlas.details import atlas_description
    from langslice.core.atlas.images import AUGMENTED_ATLAS, BORROWED_NISSL

    lines: list[str] = []
    described = atlas_description(atlas_name)
    if described:
        lines.append(f"- The atlas as BrainGlobe describes it: {described}")
    if atlas_channels and BORROWED_NISSL in atlas_channels:
        lines.append(f"- `{BORROWED_NISSL}`: the Nissl template of {AUGMENTED_ATLAS} (the "
                     "Blue Brain averaged Nissl, registered to the CCFv3 annotation), "
                     "resampled onto this atlas's grid.")
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


def engine_fact(tool_names: list[str]) -> str | None:
    """The job statement's line when the deformable engine cannot run here (None when it can,
    or when no tool of the run needs it)."""
    from langslice.core.deformation import engine_available

    needing = [name for name in ("fireants_syn", "trace_borders") if name in tool_names]
    if not needing or engine_available():
        return None
    return (f"- FireANTs (or torch) is not installed on this host, so "
            f"{' and '.join(needing)} cannot run.")


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
    auto: bool | None = None,
    nissl_image: str | None = None,
) -> str:
    """The system instruction for one run, built from the spec and the state.

    *max_resolution* is the largest picture the driver model takes, the
    top of ``look``'s ``resolution`` at image resolution "auto" (None: the
    OpenAI lanes', ``opening.DEFAULT_IMAGE_LIMIT``); every door reads the
    same statement, its own lines added after it (``doors.statement``).
    *auto* whether the caller sizes each picture (None: the spec's image
    resolution is "auto", the agent CLI always does); *axis_ends* is ``(low, high)`` from
    :func:`langslice.core.space.slice_axis_ends` — what the two ends of the slicing
    axis are anatomically in THIS atlas.
    """
    off = spec.switched_off
    held = freeze.everywhere(spec.frozen)
    #: Positions are the agent's to write (the Positioning task is on and its
    #: positions are not frozen): else the task, when on, is the stack's angles.
    places = spec.has("position") and "position" not in held
    jobs: list[str] = []
    if spec.has("position") and not places:
        jobs.append("give the stack its cutting angles, checked against the atlas")
    if places:
        jobs.append(
            "give every section its own position in millimetres along the "
            "slicing axis, each one inspected and checked against the atlas, "
            "damaged sections included, and report, by filename, each section "
            "after a gap where you conclude the interval between neighbouring "
            "sections is genuinely broken"
        )
    if spec.has("transform"):
        turned = "turned" if spec.off("interactive_transform:flip") else "mirrored or turned"
        if "orientation" in held:
            jobs.append("give every section one in-plane transform onto its atlas section")
        elif "affine" in held:
            jobs.append(f"correct the orientation of any section that is {turned}")
        else:
            jobs.append(
                "give every section one in-plane transform onto its atlas section, "
                f"correcting the orientation of any section that is {turned}"
            )
    if spec.has("nonlinear"):
        jobs.append(
            ("give every section a deformation onto the atlas, on top of its linear "
             "placement, with the section's stain as the evidence"
             if spec.nonlinear.require_deformation else
             "give a section a deformation onto the atlas where it improves on its "
             "linear placement, with the section's stain as the evidence")
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
    from langslice.core.section_images import opening_source, opening_words

    absent = {name for name in ("make_image", "trace_borders") if name not in tool_names}
    facts += channel_facts(channels, atlas_channels, max_resolution if sized else None,
                           opening=opening_words(opening_source(spec)), nissl_image=nissl_image,
                           off=set(off) | absent)
    facts += atlas_facts(state.atlas, atlas_channels)

    tools = ("Tools (each one's own description says what it does and returns): "
             + ", ".join(f"`{name}`" for name in tool_names) + ".")

    constraints: list[str] = []
    if places:
        constraints.append(
            "- `submit` is refused unless every section has a position of its own, "
            "damaged sections included."
        )
        if spec.position.strict_interval:
            constraints.append(
                f"- Every consecutive spacing must be within {STRICT_INTERVAL_TOLERANCE:.0%} of "
                f"{state.interval_mm:.3f} mm, and no interval break may be "
                f"reported."
            )
        else:
            constraints.append(
                "- A reported interval break is accepted only at a section "
                "whose interval from the section before it, in the positions "
                f"you wrote, exceeds {INTERVAL_BREAK_MIN_RATIO:g}x the stack's median written "
                "spacing."
            )
    if spec.has("transform") and "affine" not in held:
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
        missing = engine_fact(tool_names)
        if missing:
            constraints.append(missing)
    constraints.append(
        "- Corrections are recorded as data; the user's image files are never "
        "modified."
    )
    constraints.append(
        "- LangSlice sets no limit on tokens, time or tool calls. Quality is the "
        "absolute priority: this run may be evaluated on the quality of its "
        "registration alone, without regard to the time or tokens it spends."
    )

    method: list[str] = []
    if places and spec.position.playbook:
        # Optional coaching: use the current viewing tools and revisit
        # uncertain placements without prescribing a single batch write.
        method = [
            "",
            "Method:",
            "- Start from the opening stack and identify anatomical progression, "
            "possible interleaved series, missing sections and uncertain orientations. "
            "Form initial position hypotheses from the anatomy.",
            *_bullet(
                "[[?look:positioning Use `look` in mode positioning to compare sections "
                "with candidate atlas positions and their neighbours. ]]"
                "[[?grep_atlas|grep_atlas_view Use [[#grep_atlas=`grep_atlas`||"
                "grep_atlas_view=`grep_atlas_view`]] to locate distinguishing anatomy; ]]"
                "[[?look:section|look:atlas|zoom [[?grep_atlas|grep_atlas_view u||U]]se "
                "[[#look:section|look:atlas=section or atlas views||zoom=`zoom`]] when "
                "you need more detail.]]", off),
            "- Write supported placements with `position_sections`, in groups "
            "that you can assess together. Revisit uncertain sections as their "
            "neighbours become clearer; the nominal interval informs the search "
            "but does not replace anatomical evidence.",
            render("- Review the whole stack[[?look:positioning  in positioning mode]] "
                   "after changes. Check "
                   "both sides of an apparent gap before reporting an interval break. "
                   "Complete the enabled alignment and deformation tasks before "
                   "submitting.", off),
        ]
    elif places:
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

    if places and (spec.has("transform") or spec.has("nonlinear")):
        method.append(render(
            "- Positions can be revisited throughout registration. If [[?zoom zooms or ]]"
            "repeated looks suggest that a section's placement is wrong, return to atlas "
            "comparisons and re-place it, even after beginning linear or nonlinear "
            "alignment.", off
        ))

    fits = [name for name in ("elastix_affine", *DEFORMABLE_METHODS) if name in tool_names]
    if spec.has("transform"):
        if not method:
            method = ["", "Method:"]
        if fits:
            method.append(
                "- Fitting algorithms refine a placement that is already close; they do not "
                "correct a large offset.")
        if not spec.off("interactive_transform"):
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
                "- Every fit leaves the regions marked with `mark_damage` out of the atlas, "
                "so that the regions a damaged section still has drive its fits."
            )
        engines = [name for name in DEFORMABLE_METHODS if name in tool_names]
        method.append(
            "- Inspect each fit's borders against the section's internal anatomy"
            + (" and, where traced, its traced borders"
               if "trace_borders" in tool_names else "")
            + "; undo a fit that does not improve the alignment."
        )
        if len(engines) > 1:
            method.append(
                "- Several published deformable methods are available; none is the default, "
                "and each suits different sections. Each fit builds on the section's current "
                "registration, so methods can be chained."
            )
            method.extend(f"  - `{name}`: {DEFORMABLE_METHODS[name]}" for name in engines)

    if any(name in tool_names for name in ("elastix_affine", "fireants_syn")):
        if not method:
            method = ["", "Method:"]
        method.append("- A fit's reply names the section image, atlas image and mask it used.")
        method.append(
            "- A fit compares one section image with one atlas image. It does best when both "
            "show as much internal structure as possible (layers, nuclei, fibre tracts, "
            "ventricle walls) and when the two look alike. Choose, make and look at images by "
            "that principle; there is no fixed recipe.")

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


def _bullet(text: str, off: Collection[str]) -> list[str]:
    """A method bullet written with what it needs, rendered; none when the
    job disables everything it says."""
    line = render("- " + text, off).rstrip()
    return [line] if line.strip(" -") else []


def stack_angles_fact(state: StackState, off: Collection[str] = ()) -> str:
    """The job statement's line on the cutting angles: the stack's one
    angle, or, when the sections differ (a registration supplied per
    section), that each status row carries its own. *off*: the job's
    disabled list."""
    if state.mixed_angles:
        return render("- Cutting angles: these differ between sections, as supplied; each "
                      "status row carries its own cutting_angles_deg.[[?position_sections "
                      " position_sections sets one angle for every section.]]", off)
    pitch, yaw = state.stack_angles
    return f"- Stack-wide cutting angles: pitch {pitch:.2f} deg, yaw {yaw:.2f} deg."


def cutting_protocol_fact(state: StackState) -> str:
    """Supplied protocol values, with unknowns explicit."""
    interval = state.interval_mm
    thickness = state.thickness_mm
    known_interval = interval is not None
    known_thickness = thickness is not None
    interval_text = f"{interval:.3f} mm center-to-center" if known_interval else "not supplied"
    thickness_text = f"{thickness:.3f} mm" if known_thickness else "not supplied"
    text = (f"- Cutting protocol: nominal section interval {interval_text}, "
            f"section thickness {thickness_text}. ")
    if known_interval and known_thickness:
        return text + "These are protocol values; spacing may vary where sections are missing."
    return (text + "Infer positions and spacing from anatomy. Estimate missing protocol "
            "values where the evidence supports them, and record estimates or unresolved "
            "uncertainty in your notes. Spacing may vary where sections are missing.")


def run_facts(
    spec: JobSpec, state: StackState, *, species: str, pos_lo: float,
    pos_hi: float, axis_ends: tuple[str, str],
) -> list[str]:
    """Shared factual briefing, without any model-specific method advice."""
    axis = _PLANE_AXIS_LABEL.get(state.plane, "AP")
    ordered = state.in_order()
    placed = [s for s in ordered if s.position_mm is not None]
    damaged = [s.id for s in ordered if s.damaged]
    inputs = spec.inputs or {}
    noted = [s.id for s in ordered if s.id in (inputs.get("damaged") or {})]
    held = freeze.everywhere(spec.frozen)

    facts: list[str] = [
        f"- {len(state.slices)} sections, {state.plane} plane, atlas "
        f"{state.atlas} ({species}).",
        f"- Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm along the slicing "
        f"axis, measured from the origin edge of the atlas volume "
        f"({pos_lo:.2f} mm is its first section, {pos_hi:.2f} mm its last).",
        f"- {pos_lo:.2f} mm is the {axis_ends[0]} edge of the volume; "
        f"positions increase toward {axis_ends[1]}.",
        cutting_protocol_fact(state),
        stack_angles_fact(state, spec.switched_off),
    ]
    if placed:
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
    if frozen_line := freeze.statement(spec.frozen, [s.id for s in ordered]):
        facts.append(frozen_line)
    if spec.has("position") and "position" not in held:
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
        if (spec.off("interactive_transform:flip") and not spec.off("interactive_transform")
                and "orientation" not in held):
            facts.append("- Flipping sections is switched off for this run.")
        facts.append(
            "- Alignment canvas: each section is drawn at its TRUE physical size "
            "from its pixel size (read from the file, or given by the host; "
            "without one the section image's width is mapped onto the atlas "
            "plane's width and the tool payload says uncalibrated), and the "
            "atlas at its voxel size; scale 1.0 is the section's calibrated size."
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
    if cue and spec.has("transform") and not spec.off("interactive_transform:flip"):
        facts.append(f"- What marks a hemisphere, from the user: {cue}")
    facts.extend(f"- {fact.strip()}" for fact in spec.facts if str(fact).strip())

    return facts
