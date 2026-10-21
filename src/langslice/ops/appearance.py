"""How sections look: the derived channels every fit and the image model read
(``preprocessed`` among them, :mod:`langslice.core.appearance`), and the raw
channels' display properties (:mod:`langslice.core.channels`). :func:`set_channel`
writes either."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core import appearance as looks
from langslice.core import channels
from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from PIL import Image

    from langslice.core.appearance import Look
    from langslice.core.display import DisplayOptions
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


@dataclass(frozen=True)
class AppearanceSet:
    """What :func:`set_appearance` wrote and what is now in force."""

    #: The sections written, or None for the stack setting.
    section_ids: list[str] | None
    #: ``{"stack": settings}`` or ``{"sections": {id: settings}}`` (None is
    #: the default appearance).
    in_force: dict[str, Any] = field(default_factory=dict)


def planned_settings(
    state: StackState, section_ids: Sequence[str] | None, settings: Look, section_id: str,
    name: str = looks.PREPROCESSED,
) -> Look:
    """Derived channel *name*'s settings *section_id* would have after that write.

    Nothing is written: a door draws the AFTER picture of a write it has not
    made yet, and makes it only when every picture succeeded.
    """
    scratch = copy.copy(state)
    scratch.appearance = copy.deepcopy(state.appearance)
    looks.set_derived(scratch, name, None if section_ids is None else list(section_ids), settings)
    return looks.derived_settings(scratch, name, section_id)


def set_appearance(
    job: Job, section_ids: Sequence[str] | None, settings: Look,
    name: str = looks.PREPROCESSED,
) -> AppearanceSet:
    """Write derived channel *name*'s *settings* (validated,
    :func:`langslice.core.appearance.validate_recipe`; None = back to the
    default, or deleted for a name other than ``preprocessed``), stack-wide
    (*section_ids* None) or for those sections. One undo step.
    """
    ids = None if section_ids is None else list(section_ids)
    before = job.snapshot()
    looks.set_derived(job.state, name, ids, settings)
    job.commit(before)
    state = job.state
    in_force = ({"sections": {section: looks.derived_settings(state, name, section)
                              for section in ids}}
                if ids else {"stack": looks.derived_settings(state, name, "")})
    return AppearanceSet(section_ids=ids, in_force=in_force)


@dataclass(frozen=True)
class BeforeAfter:
    """One section's appearance before and after a :func:`preprocess` call:
    the settings and the whole-image picture of each (no caption: the door
    labels them)."""

    record: SliceState
    before_settings: Look
    before: Image.Image
    after_settings: Look
    after: Image.Image


@dataclass(frozen=True)
class Preprocessed:
    """What :func:`preprocess` wrote, and the pictures it drew."""

    written: AppearanceSet
    pictures: list[BeforeAfter] = field(default_factory=list)


def preprocess(
    job: Job,
    workspace: Workspace,
    section_ids: Sequence[str] | None,
    settings: Look,
    *,
    shown: Sequence[SliceState] = (),
    options: DisplayOptions | None = None,
    name: str = looks.PREPROCESSED,
) -> Preprocessed:
    """Set the appearance (:func:`set_appearance`), picturing it first.

    For each section in *shown*, with *options*: derived channel *name*
    BEFORE the call (drawn first, from the settings as they
    stand) and AFTER (from the settings the write will leave,
    :func:`planned_settings`), each the whole image
    (:func:`langslice.core.display.framed_section`). The write happens only
    once every picture is drawn: a picture that fails refuses the call
    (``RENDER_FAILED``), nothing written.
    """
    from langslice.core.display import framed_section

    state = job.state
    ids = None if section_ids is None else list(section_ids)
    pairs: list[BeforeAfter] = []
    if options is not None:
        try:
            earlier = [
                (record, looks.derived_settings(state, name, record.id),
                 framed_section(workspace, state, record, options,
                                look=looks.derived_settings(state, name, record.id)))
                for record in shown
            ]
            for record, was, before in earlier:
                now = planned_settings(state, ids, settings, record.id, name)
                pairs.append(BeforeAfter(
                    record=record, before_settings=was, before=before, after_settings=now,
                    after=framed_section(workspace, state, record, options, look=now),
                ))
        except Exception as exc:
            raise Refused("RENDER_FAILED", message=str(exc)) from exc
    written = set_appearance(job, ids, settings, name)
    return Preprocessed(written=written, pictures=pairs)


# --- the raw channels' display properties ------------------------------------


@dataclass(frozen=True)
class ChannelPropertiesSet:
    """What :func:`set_raw_channel` wrote and what is now in force."""

    channel: str
    #: The properties now in force (None: none, the default display).
    properties: dict[str, Any] | None
    #: The sections that have the channel.
    sections: list[str]
    #: The channel's file intensities over those sections
    #: (:func:`langslice.core.channels.channel_summary`): the sample type and
    #: its range, and the 1st and 99.5th percentiles.
    intensities: dict[str, Any] | None
    #: Whether this call changed anything (an unchanged call takes no undo step).
    changed: bool


def set_raw_channel(
    job: Job,
    workspace: Workspace,
    channel: str,
    *,
    contrast_limits: Sequence[float] | None = None,
    gamma: float | None = None,
    colormap: str | None = None,
    reset: bool = False,
) -> ChannelPropertiesSet:
    """Set how the raw channel *channel* is displayed, stack-wide.

    Display only (:mod:`langslice.core.channels`): the pictures that draw raw
    channels change, nothing a fit or the image model reads does, and the
    user's images are never edited. An argument left None keeps the
    channel's current value; *reset* clears them all. *contrast_limits* are
    file intensities. One undo step, none when nothing changed.
    ``UNKNOWN_CHANNEL`` when no section has the channel, ``BAD_ARGS`` for a
    bad value.
    """
    state = job.state
    name = str(channel)
    names_by_section = {record.id: workspace.section_channels(record.id)[0]
                        for record in state.in_order()}
    having = [section for section, names in names_by_section.items() if name in names]
    if not having:
        known = sorted({value for names in names_by_section.values() for value in names})
        raise Refused("UNKNOWN_CHANNEL", channel=name, channels=known)
    current = channels.channel_properties(state, name) or channels.ChannelProperties()
    target: channels.ChannelProperties | None = None
    if not reset:
        try:
            target = channels.validate_properties(
                contrast_limits=(list(contrast_limits) if contrast_limits is not None
                                 else current.contrast_limits),
                gamma=gamma if gamma is not None else current.gamma,
                colormap=colormap if colormap is not None else current.colormap,
            )
        except ValueError as exc:
            raise Refused("BAD_ARGS", message=str(exc)) from exc
    was = channels.channel_properties(state, name)
    now = None if target is None or target.is_default else target
    changed = was != now
    if changed:
        before = job.snapshot()
        channels.set_properties(state, name, now)
        job.commit(before)
    return ChannelPropertiesSet(
        channel=name, properties=None if now is None else now.to_dict(), sections=having,
        intensities=channels.channel_summary(workspace, having, name), changed=changed,
    )


# --- set_channel: one write for raw and derived channels ----------------------


@dataclass(frozen=True)
class ChannelSet:
    """What :func:`set_channel` wrote and what is now in force."""

    name: str
    #: ``raw`` (display properties) or ``derived`` (a recipe).
    kind: str
    #: Derived: ``{"stack": recipe}`` or ``{"sections": {id: recipe}}`` as in force
    #: (a recipe None: the default, or none for a deleted name). Raw: empty.
    in_force: dict[str, Any] = field(default_factory=dict)
    #: Raw: the display properties now in force (None: the default display).
    properties: dict[str, Any] | None = None
    #: Raw: the sections that have the channel, and its file intensities there.
    sections: list[str] = field(default_factory=list)
    intensities: dict[str, Any] | None = None
    #: Every channel name now, raw then derived (the stack's union).
    channels: list[str] = field(default_factory=list)
    #: A derived channel removed by ``reset``.
    deleted: bool = False
    #: Whether this call changed anything (an unchanged raw call takes no undo step).
    changed: bool = True
    #: Derived: the before/after pictures (:func:`preprocess`).
    pictures: list[BeforeAfter] = field(default_factory=list)


def _all_channel_names(workspace: Workspace, state: StackState) -> list[str]:
    raw: list[str] = []
    for record in state.in_order():
        for name in workspace.section_channels(record.id)[0]:
            if name not in raw:
                raw.append(name)
    return [*raw, *(name for name in looks.derived_names(state) if name not in raw)]


def set_channel(
    job: Job,
    workspace: Workspace,
    name: str,
    sections: Sequence[str] | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    contrast_limits: Sequence[float] | None = None,
    gamma: float | None = None,
    colormap: str | None = None,
    invert: bool = False,
    reset: bool = False,
    shown: Sequence[SliceState] = (),
    options: DisplayOptions | None = None,
) -> ChannelSet:
    """Write channel *name*, raw or derived. One undo step (none for an unchanged raw call).

    A **raw** name (a channel some section's file has) takes display properties only
    (*contrast_limits* in file intensities, *gamma*, *colormap*; an argument left
    None keeps its value; *reset* clears them,
    :func:`langslice.core.channels`); a recipe argument or *sections* is refused
    (``BAD_ARGS``). ``preprocessed`` or any other name writes a derived channel's
    recipe (:func:`langslice.core.appearance.validate_recipe`): *weights*
    ``{raw name: weight}`` (empty: automatic), CLAHE, *n4*, *denoise*, *contrast_limits*
    and *gamma* (fractions of the blended image, 0 to 1), *invert*; stack-wide
    (*sections* empty) or as the override of those sections. *reset* returns
    ``preprocessed`` to LangSlice's default recipe and deletes another name (with
    *sections*, only their overrides). Each section in *shown* is pictured BEFORE
    and AFTER a derived write (:func:`preprocess`), drawn with *options*.

    Refused: ``BAD_ARGS`` (a recipe argument on a raw name, a colormap on a derived
    one, a name that is empty, long, or a raw channel's, an invalid value, weights
    naming no channel of a section or leaving it nothing), ``UNKNOWN_SLICE_IDS``,
    ``RENDER_FAILED``.
    """
    state = job.state
    channel = str(name).strip()
    if not channel or len(channel) > looks.MAX_DERIVED_NAME or channel.lower() == "all":
        raise Refused("BAD_ARGS", message=f"a channel name is 1 to {looks.MAX_DERIVED_NAME} "
                      "characters and not 'all'")
    ids = [str(v) for v in sections or ()] or None
    raw_by_section = {record.id: list(workspace.section_channels(record.id)[0])
                      for record in state.in_order()}
    if any(channel in names for names in raw_by_section.values()):
        extras = [word for word, used in (
            ("sections", ids is not None), ("weights", bool(weights)),
            ("clahe_clip", clahe_clip != looks.DEFAULT_CLAHE_CLIP),
            ("clahe_tiles", clahe_tiles != looks.DEFAULT_CLAHE_TILES), ("n4", n4),
            ("denoise", denoise), ("invert", invert)) if used]
        if extras:
            raise Refused("BAD_ARGS", message=f"{channel} is a raw channel: it takes "
                          "contrast_limits, gamma, colormap and reset only (a recipe is for "
                          f"preprocessed or a new name); not {', '.join(extras)}.")
        done = set_raw_channel(
            job, workspace, channel, contrast_limits=list(contrast_limits or []) or None,
            gamma=gamma or None, colormap=colormap or None, reset=reset)
        return ChannelSet(
            name=channel, kind="raw", properties=done.properties, sections=done.sections,
            intensities=done.intensities, channels=_all_channel_names(workspace, state),
            changed=done.changed)
    if colormap:
        raise Refused("BAD_ARGS", message="colormap is for a raw channel; a derived channel "
                      "is gray.")
    if ids is not None:
        unknown = [section for section in ids if state.by_id(section) is None]
        if unknown:
            raise unknown_sections(state, unknown)
    scope = [r for r in (state.in_order() if ids is None else [state.by_id(i) for i in ids])
             if r is not None]
    recipe: dict[str, Any] | None = None
    if not reset:
        try:
            recipe = looks.validate_recipe(
                weights=dict(weights) if weights else None, clahe_clip=clahe_clip,
                clahe_tiles=clahe_tiles, n4=n4, denoise=denoise,
                contrast_limits=list(contrast_limits) if contrast_limits else None,
                gamma=gamma or None, invert=invert)
            for record in scope:
                looks.weights_for(recipe, raw_by_section[record.id])
        except ValueError as exc:
            raise Refused("BAD_ARGS", message=str(exc)) from exc
        named = recipe["weights"] or {}
        known = {value for names in raw_by_section.values() for value in names}
        absent = sorted(set(named) - known)
        if absent:
            raise Refused("BAD_ARGS", message=f"weights name no channel: {', '.join(absent)}; "
                          f"the raw channels: {', '.join(sorted(known))}.")
    pictured = preprocess(job, workspace, ids, recipe, shown=shown, options=options,
                          name=channel)
    now = job.state
    deleted = recipe is None and channel not in looks.derived_names(now)
    in_force = ({"sections": {section: looks.derived_settings(now, channel, section)
                              for section in ids}}
                if ids else {"stack": looks.derived_settings(now, channel, "")})
    return ChannelSet(
        name=channel, kind="derived", in_force=in_force,
        channels=_all_channel_names(workspace, now), deleted=deleted,
        pictures=pictured.pictures)
