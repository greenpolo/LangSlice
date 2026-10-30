"""The job statement every door gives a registration agent, in one place.

The statement reports; the model reasons. Each tool is described once, by
its own description (:mod:`~langslice.tools.descriptions.render`); the
statement names the tools and describes none, except the deformable methods
when a job has more than one (:data:`DEFORMABLE_METHODS`: what each
published method is, so that none reads as the default). It gives no rules
of thumb and no warnings about failure modes. The "Method:" section per
task that is on holds facts and first principles only, never an order of
steps; ``position.playbook`` replaces the positioning lines with a
step-by-step method for models that do not find one on their own. It names
no region, no landmark and no absolute position: the same text runs
against every BrainGlobe atlas, species and plane.

:data:`STATEMENT` is the whole statement with its slots; each slot is
filled from the templates below it, chosen by the run's
:class:`StatementFacts`, its run facts by
:mod:`~langslice.tools.descriptions.run_facts`. Every door says the same
statement (:func:`job_statement`) and adds only what it must say
differently (:mod:`~langslice.tools.descriptions.door_text`):

=========  ==========================================================
door       after the statement
=========  ==========================================================
``agent``  the user's notes; the opening pictures and the status table
           follow as their own parts of the first message
``mcp``    the ``show_stack`` pages to read, the image-model line, the
           user's notes, the status table
``cli``    the opening pictures' files and how the commands behave,
           then as MCP
=========  ==========================================================

The ADK agent's system prompt is :data:`SYSTEM_PROMPT`, the same for every
job. A line that names a tool or an option is written with what it needs
(:mod:`langslice.job.settings.needs`) and rendered for the job; the user's
own words are never rendered.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import cast

from langslice.core.atlas.images import default_image
from langslice.core.pictures.opening import DEFAULT_IMAGE_LIMIT
from langslice.core.pictures.options import available_atlas_channels
from langslice.core.pictures.sizes import AUTO_RESOLUTION
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace
from langslice.job.gates import (
    INTERVAL_BREAK_MIN_RATIO,
    STRICT_INTERVAL_TOLERANCE,
)
from langslice.job.settings import frozen as freeze
from langslice.job.settings.needs import render
from langslice.job.settings.spec import JobSpec
from langslice.tools.descriptions.description import AGENT_DOOR
from langslice.tools.descriptions.door_text import (
    image_model_off_text,
    status_and_notes,
    user_notes_lines,
)
from langslice.tools.descriptions.run_facts import (
    Channels,
    atlas_facts,
    channel_facts,
    engine_fact,
    run_facts,
    section_channels,
)

#: The registration agent's system prompt: true of every job, naming no
#: tool a job can switch off; the job itself comes in the first message.
SYSTEM_PROMPT = (
    "You are an expert neuroanatomist registering one brain with LangSlice: "
    "a stack of histology sections from one brain, placed in a BrainGlobe "
    "atlas. The first message gives the job: what to do, the tools, and the "
    "facts about this stack and atlas. Work the job with the tools and "
    "finish with `submit`."
)

#: The statement, its slots filled by :func:`build_job_statement`.
STATEMENT = """\
You are an expert neuroanatomist working on a stack of {count} {plane} \
histology sections against the {atlas} ({species}) reference atlas.

Your job: {job}.{task_notes}

Run facts:
{facts}

Tools (each one's own description says what it does and returns): {tools}.

Constraints:
{constraints}{method}

Work with the tools, then call `submit`."""

#: The deformable methods a job can hold (the native fit and the in-tree
#: engine plugins), in the order the Method lists them, each with what it
#: is: the claims are the methods' own papers'.
DEFORMABLE_METHODS: dict[str, str] = {
    "fireants_syn": (
        "ANTs SyN (Avants et al. 2008), the standard symmetric diffeomorphic "
        "registration in neuroimaging, run on the GPU by FireANTs. It matches "
        "brightness (mutual information or local correlation) and optionally "
        "edges, takes masks, and `restrict_to` refines single regions."
    ),
    "stalign_lddmm": (
        "STalign (Clifton et al. 2023), LDDMM built to align single-cell "
        "spatial transcriptomics and histology to the Allen CCF. It fits a "
        "contrast map between the two images and sorts every pixel into "
        "tissue, background or artifact itself, so partial sections and "
        "debris count less."
    ),
    "spomialign": (
        "SPOmiAlign (Wang et al. 2026), built to align spatial-omics sections "
        "across modalities. It matches points with RoMa, a pretrained dense "
        "feature matcher, instead of comparing brightness, so it needs only "
        "shared tissue shape and structure, not similar staining; seconds per "
        "call. Its authors report it degrades when the two images share "
        "little tissue and in repetitive or ambiguous tissue."
    ),
    "emlddmm_stack": (
        "emlddmm (Tward et al. 2025), built to bring whole-brain series of "
        "any modality into the Allen CCF. It fits one smooth 3-D warp of the "
        "atlas to the whole stack, with each section's own mounting, and "
        "sorts out missing tissue and artifacts itself. It suits dense, "
        "evenly spaced series and gives placements that agree across the "
        "stack; it makes no single-section bends."
    ),
}

# --- "Your job:", one clause per task that is on ----------------------------

JOB_ANGLES = "give the stack its cutting angles, checked against the atlas"
JOB_POSITIONS = (
    "give every section its own position in millimetres along the slicing "
    "axis, each one inspected and checked against the atlas, damaged "
    "sections included, and report, by filename, each section after a gap "
    "where you conclude the interval between neighbouring sections is "
    "genuinely broken"
)
JOB_TRANSFORM = (
    "give every section one in-plane transform onto its atlas section"
)
JOB_ORIENTATION = "correct the orientation of any section that is {turned}"
JOB_TRANSFORM_AND_ORIENTATION = (
    "give every section one in-plane transform onto its atlas section, "
    "correcting the orientation of any section that is {turned}"
)
TURNED = "turned"
MIRRORED_OR_TURNED = "mirrored or turned"
JOB_EVERY_DEFORMATION = (
    "give every section a deformation onto the atlas, on top of its linear "
    "placement, with the section's stain as the evidence"
)
JOB_SOME_DEFORMATIONS = (
    "give a section a deformation onto the atlas where it improves on its "
    "linear placement, with the section's stain as the evidence"
)
JOB_TRACES = (
    ", or, for a section you choose to trace, the borders the image model "
    "traces on it"
)
JOB_REVIEW = "review the stack"

#: Heading of each task's user notes, in the order the tasks are listed.
TASK_NOTE_TITLES: dict[str, str] = {
    "position": "Positioning",
    "transform": "In-plane alignment",
    "nonlinear": "Nonlinear deformation",
}
TASK_NOTES = "{title} notes from the user:"

TOOLS = "`{name}`"

# --- "Constraints:" ---------------------------------------------------------

SUBMIT_POSITIONS = (
    "- `submit` is refused unless every section has a position of its own, "
    "damaged sections included."
)
STRICT_SPACING = (
    "- Every consecutive spacing must be within {tolerance:.0%} of "
    "{interval:.3f} mm, and no interval break may be reported."
)
BREAK_RULE = (
    "- A reported interval break is accepted only at a section whose "
    "interval from the section before it, in the positions you wrote, "
    "exceeds {ratio:g}x the stack's median written spacing."
)
SUBMIT_TRANSFORMS = (
    "- `submit` is refused unless every section carries a transform, damaged "
    "sections included."
)
DAMAGED_TRANSFORM = (
    "- A damaged section (one with marked regions) still needs a transform "
    "made for its surviving anatomy: set it {ways}. Inspect each overlay "
    "against the surviving internal anatomy before submitting."
)
BY_HAND = "by hand with `interactive_transform`"
BY_ELASTIX = (
    "with `elastix_affine`, which leaves the marked regions out of the fit"
)
OR_FIT = ", or fit it "
TRACE_OPTIONAL = (
    "- `trace_borders` is optional: you decide which sections, if any, to "
    "trace. A trace requires a position and a linear transform."
)
SUBMIT_DEFORMATIONS = (
    "- `submit` is refused unless every section carries a deformation "
    "applied at its current placement, damaged sections included; the user "
    "requires one on every section."
)
DATA_ONLY = (
    "- Corrections are recorded as data; the user's image files are never "
    "modified."
)
NO_LIMITS = (
    "- LangSlice sets no limit on tokens, time or tool calls. Quality is the "
    "absolute priority: this run may be evaluated on the quality of its "
    "registration alone, without regard to the time or tokens it spends."
)

# --- "Method:" --------------------------------------------------------------

METHOD = "Method:"
#: Optional coaching (``position.playbook``): the current viewing tools,
#: uncertain placements revisited, no single batch write prescribed.
PLAYBOOK = (
    "- Start from the opening stack and identify anatomical progression, "
    "possible interleaved series, missing sections and uncertain "
    "orientations. Form initial position hypotheses from the anatomy.",
    "- [[?look:positioning Use `look` in mode positioning to compare sections "
    "with candidate atlas positions and their neighbours. ]]"
    "[[?grep_atlas|grep_atlas_view Use [[#grep_atlas=`grep_atlas`||"
    "grep_atlas_view=`grep_atlas_view`]] to locate distinguishing anatomy; ]]"
    "[[?look:section|look:atlas|zoom [[?grep_atlas|grep_atlas_view u||U]]se "
    "[[#look:section|look:atlas=section or atlas views||zoom=`zoom`]] when "
    "you need more detail.]]",
    "- Write supported placements with `position_sections`, in groups that "
    "you can assess together. Revisit uncertain sections as their neighbours "
    "become clearer; the nominal interval informs the search but does not "
    "replace anatomical evidence.",
    "- Review the whole stack[[?look:positioning  in positioning mode]] after "
    "changes. Check both sides of an apparent gap before reporting an "
    "interval break. Complete the enabled alignment and deformation tasks "
    "before submitting.",
)
PLACING = (
    "- Place each section on its own evidence: compare it against candidate "
    "atlas positions before writing, and do not let the nominal interval "
    "stand in for a look.",
    "- After writing, review the whole stack against the atlas, watch for "
    "spacings that differ from their neighbours, and re-check the sections "
    "on either side of any gap before reporting an interval break.",
    "- Submit when the work is complete; address any missing requirements "
    "it returns.",
)
REVISIT_POSITIONS = (
    "- Positions can be revisited throughout registration. If [[?zoom zooms "
    "or ]]repeated looks suggest that a section's placement is wrong, return "
    "to atlas comparisons and re-place it, even after beginning linear or "
    "nonlinear alignment."
)
FITS_REFINE = (
    "- Fitting algorithms refine a placement that is already close; they do "
    "not correct a large offset."
)
INSPECT_OVERLAY = (
    "- After each automatic fit or manual adjustment, inspect the returned "
    "overlay against surviving internal anatomy. Refine each slice's "
    "alignment, damaged slices included, until no further improvement is "
    "possible with the available transforms. Keep changes only if they "
    "improve the alignment."
)
DAMAGE_LEFT_OUT = (
    "- Every fit leaves the regions marked with `mark_damage` out of the "
    "atlas, so that the regions a damaged section still has drive its fits."
)
INSPECT_FIT = (
    "- Inspect each fit's borders against the section's internal "
    "anatomy{traced}; undo a fit that does not improve the alignment."
)
TRACED = " and, where traced, its traced borders"
METHODS_CHAINED = (
    "- Several published deformable methods are available; none is the "
    "default, and each suits different sections. Each fit builds on the "
    "section's current registration, so methods can be chained."
)
DEFORMABLE_METHOD = "  - `{name}`: {what}"
FIT_REPLY = (
    "- A fit's reply names the section image, atlas image and mask it used."
)
FIT_IMAGES = (
    "- A fit compares one section image with one atlas image. It does best "
    "when both show as much internal structure as possible (layers, nuclei, "
    "fibre tracts, ventricle walls) and when the two look alike. Choose, "
    "make and look at images by that principle; there is no fixed recipe."
)


@dataclass(frozen=True)
class StatementFacts:
    """What the job statement reports of one run."""

    spec: JobSpec
    state: Stack
    #: The tools the run has, in the order the doors list them.
    tool_names: tuple[str, ...]
    species: str
    #: The lowest and highest atlas position, in millimetres.
    position_range: tuple[float, float]
    #: What the two ends of the slicing axis are in this atlas.
    axis_ends: tuple[str, str]
    channels: Channels | None
    atlas_channels: tuple[str, ...]
    nissl_image: str
    #: The largest ``look`` picture where the caller sizes pictures; None
    #: where it does not.
    max_resolution: int | None

    @classmethod
    def from_run(
        cls,
        spec: JobSpec,
        state: Stack,
        workspace: Workspace,
        *,
        tool_names: Sequence[str],
        max_resolution: int | None = None,
        auto: bool | None = None,
    ) -> StatementFacts:
        """Return the facts of a run of ``spec`` on ``state``.

        Parameters
        ----------
        spec
            The job's settings.
        state
            The stack as the run starts.
        workspace
            The run's workspace: the atlas and the section files.
        tool_names
            The tools the run has.
        max_resolution
            The largest picture the driver model takes; None: the OpenAI
            lanes' (``core.pictures.opening.DEFAULT_IMAGE_LIMIT``).
        auto
            Whether the caller sizes each picture; None: the spec's image
            resolution is ``"auto"``.

        Returns
        -------
        StatementFacts
            What the statement reports.
        """
        if auto is None:
            auto = spec.image_resolution == AUTO_RESOLUTION
        if not auto:
            max_resolution = None
        elif max_resolution is None:
            max_resolution = DEFAULT_IMAGE_LIMIT[0]
        atlas = cast("object", workspace.atlas)  # BrainGlobe's, untyped
        return cls(
            spec=spec,
            state=state,
            tool_names=tuple(tool_names),
            species=workspace.species,
            position_range=workspace.position_range,
            axis_ends=workspace.axis_ends,
            channels=section_channels(workspace, state),
            atlas_channels=available_atlas_channels(workspace),
            nissl_image=default_image(atlas),
            max_resolution=max_resolution,
        )


def job_statement(
    facts: StatementFacts,
    *,
    door: str,
    opening: Sequence[str] = (),
    notes: str = "",
    image_model_off: bool = False,
) -> str:
    """Return what ``door`` tells the agent of the job (see the module text).

    Parameters
    ----------
    facts
        The run the statement reports.
    door
        ``agent``, ``mcp`` or ``cli``.
    opening
        The door's own paragraph on its opening pictures
        (:func:`opening_for_mcp`, :func:`opening_for_cli`).
    notes
        The user's notes for the job (:func:`read_notes`).
    image_model_off
        Whether the job's image model is not connected here: the statement
        is then that of a run without one, plus the image-model line.

    Returns
    -------
    str
        The whole text.
    """
    if image_model_off:  # a copy that is not checked again: this door's view
        spec = copy.copy(facts.spec)
        spec.nonlinear = replace(spec.nonlinear, provider="none")
        facts = replace(facts, spec=spec)
    lines = [build_job_statement(facts)]
    if door == AGENT_DOOR:
        return "\n".join([*lines, *user_notes_lines(notes)])

    if opening:
        lines += ["", *opening]
    off_text = image_model_off_text(facts.spec.switched_off)
    if image_model_off and off_text:
        lines += ["", off_text]
    lines += user_notes_lines(notes)
    lines += ["", status_and_notes(facts.state)]
    return "\n".join(lines)


def build_job_statement(facts: StatementFacts) -> str:
    """Return the job statement of a run: :data:`STATEMENT`, filled.

    Parameters
    ----------
    facts
        The run the statement reports.

    Returns
    -------
    str
        The statement, the same for every door.
    """
    spec, state, names = facts.spec, facts.state, facts.tool_names
    absent = [
        name for name in ("make_image", "trace_borders") if name not in names
    ]
    shown = [
        *run_facts(
            spec,
            state,
            species=facts.species,
            position_range=facts.position_range,
            axis_ends=facts.axis_ends,
        ),
        *channel_facts(
            spec,
            facts.channels,
            facts.atlas_channels,
            max_resolution=facts.max_resolution,
            nissl_image=facts.nissl_image,
            off={*spec.switched_off, *absent},
        ),
        *atlas_facts(state.atlas, facts.atlas_channels),
    ]
    constraints = [
        *_position_constraints(spec, state),
        *_transform_constraints(spec, names),
        *_deformation_constraints(spec, names),
        DATA_ONLY,
        NO_LIMITS,
    ]
    method = _method(spec, names)
    if method:
        method = ["", METHOD, *method]
    return STATEMENT.format(
        count=len(state.slices),
        plane=state.plane,
        atlas=state.atlas,
        species=facts.species,
        job="; ".join(_job(spec, names)) or JOB_REVIEW,
        task_notes=_block(task_notes(spec)),
        facts="\n".join(shown),
        tools=", ".join(TOOLS.format(name=name) for name in names),
        constraints="\n".join(constraints),
        method=_block(method),
    )


def _writes_positions(spec: JobSpec) -> bool:
    """Return whether the positions are the agent's to write."""
    frozen_everywhere = freeze.everywhere(spec.frozen)
    return spec.has("position") and "position" not in frozen_everywhere


def _job(spec: JobSpec, names: Collection[str]) -> list[str]:
    jobs: list[str] = []
    if _writes_positions(spec):
        jobs.append(JOB_POSITIONS)
    elif spec.has("position"):
        jobs.append(JOB_ANGLES)
    if spec.has("transform"):
        jobs.append(_transform_job(spec))
    if spec.has("nonlinear"):
        deformation = JOB_SOME_DEFORMATIONS
        if spec.nonlinear.require_deformation:
            deformation = JOB_EVERY_DEFORMATION
        if "trace_borders" in names:
            deformation += JOB_TRACES
        jobs.append(deformation)
    return jobs


def _transform_job(spec: JobSpec) -> str:
    frozen_everywhere = freeze.everywhere(spec.frozen)
    turned = MIRRORED_OR_TURNED
    if spec.off("interactive_transform:flip"):
        turned = TURNED
    if "orientation" in frozen_everywhere:
        return JOB_TRANSFORM
    if "affine" in frozen_everywhere:
        return JOB_ORIENTATION.format(turned=turned)
    return JOB_TRANSFORM_AND_ORIENTATION.format(turned=turned)


def _position_constraints(spec: JobSpec, state: Stack) -> list[str]:
    if not _writes_positions(spec):
        return []
    if spec.position.strict_interval:
        interval = state.interval_mm
        spacing = STRICT_SPACING.format(
            tolerance=STRICT_INTERVAL_TOLERANCE, interval=interval
        )
    else:
        spacing = BREAK_RULE.format(ratio=INTERVAL_BREAK_MIN_RATIO)
    return [SUBMIT_POSITIONS, spacing]


def _transform_constraints(spec: JobSpec, names: Collection[str]) -> list[str]:
    frozen_everywhere = freeze.everywhere(spec.frozen)
    if not spec.has("transform") or "affine" in frozen_everywhere:
        return []
    ways = [
        way
        for name, way in (
            ("interactive_transform", BY_HAND),
            ("elastix_affine", BY_ELASTIX),
        )
        if name in names
    ]
    if not ways:
        return [SUBMIT_TRANSFORMS]
    return [
        SUBMIT_TRANSFORMS,
        DAMAGED_TRANSFORM.format(ways=OR_FIT.join(ways)),
    ]


def _deformation_constraints(
    spec: JobSpec, names: Collection[str]
) -> list[str]:
    if not spec.has("nonlinear"):
        return []
    constraints: list[str] = []
    if "trace_borders" in names:
        constraints.append(TRACE_OPTIONAL)
    if spec.nonlinear.require_deformation:
        constraints.append(SUBMIT_DEFORMATIONS)
    missing = engine_fact(names)
    if missing:
        constraints.append(missing)
    return constraints


def _method(spec: JobSpec, names: Collection[str]) -> list[str]:
    method = _positioning_method(spec)
    fits = [
        name
        for name in ("elastix_affine", *DEFORMABLE_METHODS)
        if name in names
    ]
    if spec.has("transform") and fits:
        method.append(FITS_REFINE)
    if spec.has("transform") and not spec.off("interactive_transform"):
        method.append(INSPECT_OVERLAY)
    if spec.has("nonlinear"):
        method += _deformation_method(names)
    if "elastix_affine" in names or "fireants_syn" in names:
        method += [FIT_REPLY, FIT_IMAGES]
    return method


def _positioning_method(spec: JobSpec) -> list[str]:
    if not _writes_positions(spec):
        return []
    off = spec.switched_off
    method: list[str] = list(PLACING)
    if spec.position.playbook:
        bullets = [render(bullet, off).rstrip() for bullet in PLAYBOOK]
        method = [bullet for bullet in bullets if bullet.strip(" -")]
    if spec.has("transform") or spec.has("nonlinear"):
        method.append(render(REVISIT_POSITIONS, off))
    return method


def _deformation_method(names: Collection[str]) -> list[str]:
    method: list[str] = []
    if "mark_damage" in names:
        method.append(DAMAGE_LEFT_OUT)
    traced = TRACED if "trace_borders" in names else ""
    method.append(INSPECT_FIT.format(traced=traced))
    engines = [name for name in DEFORMABLE_METHODS if name in names]
    if len(engines) > 1:
        method.append(METHODS_CHAINED)
        method += [
            DEFORMABLE_METHOD.format(name=name, what=DEFORMABLE_METHODS[name])
            for name in engines
        ]
    return method


def _block(lines: Sequence[str]) -> str:
    """Return ``lines`` each after a line break: a block that follows text."""
    return "".join(f"\n{line}" for line in lines)


def task_notes(spec: JobSpec) -> list[str]:
    """Return the user's notes for each task that is on, under its name.

    Parameters
    ----------
    spec
        The job's settings, which hold each task's ``notes``.

    Returns
    -------
    list of str
        Per task with a note, a blank line, its heading and one ``- `` line
        per non-empty line of the note, verbatim.
    """
    notes = {
        "position": spec.position.notes,
        "transform": spec.transform.notes,
        "nonlinear": spec.nonlinear.notes,
    }
    lines: list[str] = []
    for task, title in TASK_NOTE_TITLES.items():
        if not spec.has(task):
            continue
        body = [
            line.strip() for line in notes[task].splitlines() if line.strip()
        ]
        if body:
            lines += ["", TASK_NOTES.format(title=title)]
            lines += [f"- {line}" for line in body]
    return lines
