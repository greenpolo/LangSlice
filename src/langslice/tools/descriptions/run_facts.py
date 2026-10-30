"""The job statement's "Run facts:": what the agent is told about this stack.

Each fact is a template below, one bullet each, its slots filled from the
run; :func:`run_facts` chooses which stand. A fact that names a tool or an
option is written with what it needs (:mod:`langslice.job.settings.needs`)
and rendered for what the job switches off; the user's own words (the
facts, the hemisphere cue) are never rendered.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast

from langslice.core.atlas.details import atlas_description
from langslice.core.atlas.images import (
    AUGMENTED_ATLAS,
    BORROWED_NISSL,
    NISSL,
)
from langslice.core.fits.deformation import engine_available
from langslice.core.pictures.options import BORDERS
from langslice.core.pictures.sizes import MIN_RESOLUTION
from langslice.core.sections.images import opening_source, opening_words
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace
from langslice.job.settings import frozen as freeze
from langslice.job.settings.needs import render
from langslice.job.settings.spec import MAX_PARALLEL_TRANSFORMS, JobSpec

#: A section's raw channel names, or each section's when they differ.
Channels = list[str] | dict[str, list[str]]

#: The slicing axis of each plane, as the range fact names it.
PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}

# --- the stack and the atlas ------------------------------------------------

SECTIONS = "- {count} sections, {plane} plane, atlas {atlas} ({species})."
RANGE = (
    "- Valid {axis} range: {low:.2f}-{high:.2f} mm along the slicing axis, "
    "measured from the origin edge of the atlas volume ({low:.2f} mm is its "
    "first section, {high:.2f} mm its last)."
)
AXIS_ENDS = (
    "- {low:.2f} mm is the {low_end} edge of the volume; positions increase "
    "toward {high_end}."
)
PROTOCOL = (
    "- Cutting protocol: nominal section interval {interval}, section "
    "thickness {thickness}. "
)
INTERVAL = "{:.3f} mm center-to-center"
THICKNESS = "{:.3f} mm"
NOT_SUPPLIED = "not supplied"
PROTOCOL_KNOWN = (
    "These are protocol values; spacing may vary where sections are missing."
)
PROTOCOL_UNKNOWN = (
    "Infer positions and spacing from anatomy. Estimate missing protocol "
    "values where the evidence supports them, and record estimates or "
    "unresolved uncertainty in your notes. Spacing may vary where sections "
    "are missing."
)
STACK_ANGLES = (
    "- Stack-wide cutting angles: pitch {pitch:.2f} deg, yaw {yaw:.2f} deg."
)
MIXED_ANGLES = (
    "- Cutting angles: these differ between sections, as supplied; each "
    "status row carries its own cutting_angles_deg.[[?position_sections  "
    "position_sections sets one angle for every section.]]"
)
POSITIONS = "- {positioned} of {count} sections carry a position."
NO_POSITIONS = "- No section carries a position yet."
DAMAGED = "- Sections with marked damage regions: {sections}."
NOTED = (
    "- The user noted damage on these sections (damage_note in status; no "
    "regions are marked from it): {sections}."
)

# --- what this run changes --------------------------------------------------

POSITIONS_WRITTEN = (
    "- The stack's order follows the positions: writing positions renumbers "
    "the sections, so use filenames to name them."
)
POSITIONS_GIVEN = (
    "- The positions shown are given; this run does not change them."
)
TRANSFORMS_GIVEN = (
    "- Existing linear transforms are supplied and fixed for this run, "
    "orientation included."
)
ORIENTATION_GIVEN = (
    "- The orientation shown is fixed; transforms are not part of this run."
)
NO_FLIP = "- Flipping sections is switched off for this run."
CANVAS = (
    "- Alignment canvas: each section is drawn at its TRUE physical size "
    "from its pixel size (read from the file, or given by the host; without "
    "one the section image's width is mapped onto the atlas plane's width "
    "and the tool payload says uncalibrated), and the atlas at its voxel "
    "size; scale 1.0 is the section's calibrated size."
)
FRAME = (
    "- Transform frame: rotation and scales act about the pivot (the canvas "
    "centre unless the section's transform holds another), x runs right and "
    "y runs down, shifts are millimetres."
)
TRANSFORM_CAP = "- Each transform tool call takes at most {cap} {sections}."
HEMISPHERE_CUE = "- What marks a hemisphere, from the user: {cue}"
USER_FACT = "- {fact}"

# --- what the pictures and fits read ----------------------------------------

RAW_CHANNELS = (
    "the planes of each section's file as read; the names come from the "
    "file or the host and may not say which is the stain"
)
CHANNELS_SHARED = "- Raw image channels of every section ({raw}): {names}."
CHANNELS_PER_SECTION = "- Raw image channels per section ({raw}): {names}."
#: ``{opening}`` is how ``preprocessed`` was made.
SECTION_IMAGES = (
    "- Section images (status lists them[[?list_files , list_files shows "
    "their files under sections/<name>/images/]]): the raw channels of each "
    "file[[?make_image , and the images made from them with make_image]]. "
    "`preprocessed` was made when the job opened: {opening}. Every "
    "fit[[?trace_borders  and the image model read|| reads]] a section image "
    "by name, and a picture of it is exactly what they read."
)
#: The one fact about a channel's signal the agent cannot see from the atlas.
SPARSE_SIGNAL = (
    "- A channel whose signal is sparse (a tracer, a reporter) shows little "
    "of the anatomy the atlas draws."
)
ATLAS_IMAGES = "- Atlas images: {images}."
DEFAULT_NISSL = " The fits' default `{nissl}` is `{image}` on this atlas."
PICTURE_SIZE = (
    "- Picture size is yours to choose: look's resolution runs from "
    "{low} to {high} pixels."
)
ATLAS_DESCRIBED = "- The atlas as BrainGlobe describes it: {description}"
BORROWED = (
    "- `{borrowed}`: the Nissl template of {atlas} (the Blue Brain averaged "
    "Nissl, registered to the CCFv3 annotation), resampled onto this atlas's "
    "grid."
)
ENGINE_MISSING = (
    "- FireANTs (or torch) is not installed on this host, so {tools} cannot "
    "run."
)


def run_facts(
    spec: JobSpec,
    state: Stack,
    *,
    species: str,
    position_range: tuple[float, float],
    axis_ends: tuple[str, str],
) -> list[str]:
    """Return the facts about the stack, the atlas and what the run changes.

    Parameters
    ----------
    spec
        The job's settings.
    state
        The stack as the run starts.
    species
        The atlas's species, as BrainGlobe names it.
    position_range
        The lowest and highest atlas position, in millimetres.
    axis_ends
        What the low and the high end of the slicing axis are in this
        atlas (``core.space.slice_axis_ends``).

    Returns
    -------
    list of str
        One bullet per fact.
    """
    low, high = position_range
    axis = PLANE_AXIS_LABEL.get(state.plane, "AP")
    facts = [
        SECTIONS.format(
            count=len(state.slices),
            plane=state.plane,
            atlas=state.atlas,
            species=species,
        ),
        RANGE.format(axis=axis, low=low, high=high),
        AXIS_ENDS.format(low=low, low_end=axis_ends[0], high_end=axis_ends[1]),
        _protocol_fact(state),
        _angles_fact(state, spec.switched_off),
        *_section_facts(spec, state),
        *_change_facts(spec),
    ]
    if spec.has("transform"):
        facts += _transform_facts(spec)
    facts += [
        USER_FACT.format(fact=fact.strip())
        for fact in spec.facts
        if fact.strip()
    ]
    return facts


def _protocol_fact(state: Stack) -> str:
    interval = NOT_SUPPLIED
    if state.interval_mm is not None:
        interval = INTERVAL.format(state.interval_mm)
    thickness = NOT_SUPPLIED
    if state.thickness_mm is not None:
        thickness = THICKNESS.format(state.thickness_mm)
    known = state.interval_mm is not None and state.thickness_mm is not None
    tail = PROTOCOL_KNOWN if known else PROTOCOL_UNKNOWN
    return PROTOCOL.format(interval=interval, thickness=thickness) + tail


def _angles_fact(state: Stack, off: Collection[str]) -> str:
    # each section its own angles: a registration supplied per section
    if state.mixed_angles:
        return render(MIXED_ANGLES, off)
    pitch, yaw = state.stack_angles
    return STACK_ANGLES.format(pitch=pitch, yaw=yaw)


def _section_facts(spec: JobSpec, state: Stack) -> list[str]:
    ordered = state.in_order()
    positioned = [s for s in ordered if s.position_mm is not None]
    facts = [NO_POSITIONS]
    if positioned:
        count = len(state.slices)
        facts = [POSITIONS.format(positioned=len(positioned), count=count)]
    damaged = [s.id for s in ordered if s.damaged]
    if damaged:
        facts.append(DAMAGED.format(sections=", ".join(damaged)))
    # the spec checks it: {filename: note}
    notes = cast("Mapping[str, str]", spec.inputs.get("damaged") or {})
    noted = [s.id for s in ordered if s.id in notes]
    if noted:
        facts.append(NOTED.format(sections=", ".join(noted)))
    frozen = freeze.statement(spec.frozen, [s.id for s in ordered])
    if frozen:
        facts.append(frozen)
    return facts


def _change_facts(spec: JobSpec) -> list[str]:
    frozen_everywhere = freeze.everywhere(spec.frozen)
    facts = [POSITIONS_GIVEN]
    if spec.has("position") and "position" not in frozen_everywhere:
        facts = [POSITIONS_WRITTEN]
    if spec.has("transform"):
        return facts
    if spec.has("nonlinear"):
        return [*facts, TRANSFORMS_GIVEN]
    return [*facts, ORIENTATION_GIVEN]


def _transform_facts(spec: JobSpec) -> list[str]:
    frozen_everywhere = freeze.everywhere(spec.frozen)
    flips = not spec.off("interactive_transform:flip")
    by_hand = not spec.off("interactive_transform")
    facts: list[str] = []
    if not flips and by_hand and "orientation" not in frozen_everywhere:
        facts.append(NO_FLIP)
    facts += [CANVAS, FRAME]
    cap = spec.transform.max_parallel
    if cap < MAX_PARALLEL_TRANSFORMS:
        sections = "section" if cap == 1 else "sections"
        facts.append(TRANSFORM_CAP.format(cap=cap, sections=sections))
    # mirroring belongs to in-plane alignment: never in a positioning-only run
    cue = spec.transform.hemisphere_cue.strip()
    if cue and flips:
        facts.append(HEMISPHERE_CUE.format(cue=cue))
    return facts


def channel_facts(
    spec: JobSpec,
    channels: Channels | None,
    atlas_channels: tuple[str, ...] | None,
    *,
    max_resolution: int | None,
    nissl_image: str | None,
    off: Collection[str],
) -> list[str]:
    """Return the facts on what the pictures and fits read.

    Parameters
    ----------
    spec
        The job's settings, which say how ``preprocessed`` was made.
    channels
        The raw channel names (:func:`section_channels`).
    atlas_channels
        The atlas's images and ``borders``.
    max_resolution
        The largest ``look`` picture where the agent sizes pictures; None
        where it does not.
    nissl_image
        What the fits' default ``nissl`` image is on this atlas.
    off
        What the job switches off and the run lacks.

    Returns
    -------
    list of str
        The raw channels, the section images, the atlas images and the
        picture size, one bullet each.
    """
    facts: list[str] = []
    if isinstance(channels, list) and channels:
        names = ", ".join(channels)
        facts.append(CHANNELS_SHARED.format(raw=RAW_CHANNELS, names=names))
    elif isinstance(channels, dict) and channels:
        names = "; ".join(
            f"{name}: {', '.join(listed)}" for name, listed in channels.items()
        )
        facts.append(
            CHANNELS_PER_SECTION.format(raw=RAW_CHANNELS, names=names)
        )

    opening = opening_words(opening_source(spec.image_settings()))
    if opening:
        facts.append(render(SECTION_IMAGES.format(opening=opening), off))
        facts.append(SPARSE_SIGNAL)
    images = [
        name for name in atlas_channels or () if name not in (BORDERS, NISSL)
    ]
    if images:
        line = ATLAS_IMAGES.format(images=", ".join(images))
        if nissl_image:
            line += DEFAULT_NISSL.format(nissl=NISSL, image=nissl_image)
        facts.append(line)
    if max_resolution:
        size = PICTURE_SIZE.format(low=MIN_RESOLUTION, high=max_resolution)
        facts.append(size)
    return facts


def atlas_facts(
    atlas_name: str, atlas_channels: tuple[str, ...] | None
) -> list[str]:
    """Return what the atlas and its borrowed Nissl image are.

    Parameters
    ----------
    atlas_name
        The BrainGlobe atlas.
    atlas_channels
        The atlas's images and ``borders``.

    Returns
    -------
    list of str
        BrainGlobe's own description of the atlas
        (:mod:`langslice.core.atlas.details`) and, where the atlas is offered
        it, what the borrowed Nissl is.
    """
    facts: list[str] = []
    described = atlas_description(atlas_name)
    if described:
        facts.append(ATLAS_DESCRIBED.format(description=described))
    if atlas_channels and BORROWED_NISSL in atlas_channels:
        facts.append(
            BORROWED.format(borrowed=BORROWED_NISSL, atlas=AUGMENTED_ATLAS)
        )
    return facts


def engine_fact(tool_names: Collection[str]) -> str | None:
    """Return the line said when the deformable engine cannot run here.

    Parameters
    ----------
    tool_names
        The run's tools.

    Returns
    -------
    str or None
        The line naming the tools that need FireANTs; None when it runs, or
        when no tool of the run needs it.
    """
    needing = [
        name
        for name in ("fireants_syn", "trace_borders")
        if name in tool_names
    ]
    if not needing or engine_available():
        return None
    return ENGINE_MISSING.format(tools=" and ".join(needing))


def section_channels(workspace: Workspace, state: Stack) -> Channels | None:
    """Return the raw channel names of the stack's sections.

    Parameters
    ----------
    workspace
        The run's workspace, which reads the section files.
    state
        The stack whose sections' files are read.

    Returns
    -------
    list of str, dict of str to list of str, or None
        One list when every section shares it, else each section's by
        filename; None when no file reads.
    """
    named: dict[str, list[str]] = {}
    for section in state.in_order():
        names = _channel_names(workspace, section.id)
        if names is not None:
            named[section.id] = names
    distinct = {tuple(names) for names in named.values()}
    if len(distinct) == 1:
        return list(next(iter(distinct))) or None
    return named or None


def _channel_names(workspace: Workspace, section_id: str) -> list[str] | None:
    try:
        return list(workspace.section_channels(section_id)[0])
    except Exception:  # noqa: BLE001 - unreadable: the opening says so
        return None
