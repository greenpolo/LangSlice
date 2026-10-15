"""How sections look: the preprocessed channel every fit and the image model
read (:mod:`langslice.core.appearance`), and the raw channels' display
properties (:mod:`langslice.core.channels`)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
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
) -> Look:
    """The preprocessed-channel settings *section_id* would have after that write.

    Nothing is written: a door draws the AFTER picture of a write it has not
    made yet, and makes it only when every picture succeeded.
    """
    scratch = copy.copy(state)
    scratch.appearance = copy.deepcopy(state.appearance)
    looks.set_settings(scratch, None if section_ids is None else list(section_ids), settings)
    return looks.preprocessed_settings(scratch, section_id)


def set_appearance(
    job: Job, section_ids: Sequence[str] | None, settings: Look,
) -> AppearanceSet:
    """Write the preprocessed channel's *settings* (validated,
    :func:`langslice.core.appearance.validate_settings`; None = back to the
    default), stack-wide (*section_ids* None) or for those sections. One
    undo step.
    """
    ids = None if section_ids is None else list(section_ids)
    before = job.snapshot()
    looks.set_settings(job.state, ids, settings)
    job.commit(before)
    state = job.state
    in_force = ({"sections": {section: looks.preprocessed_settings(state, section)
                              for section in ids}}
                if ids else {"stack": looks.preprocessed_settings(state, "")})
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
) -> Preprocessed:
    """Set the appearance (:func:`set_appearance`), picturing it first.

    For each section in *shown*, with *options*: its preprocessed channel
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
                (record, looks.preprocessed_settings(state, record.id),
                 framed_section(workspace, state, record, options,
                                look=looks.preprocessed_settings(state, record.id)))
                for record in shown
            ]
            for record, was, before in earlier:
                now = planned_settings(state, ids, settings, record.id)
                pairs.append(BeforeAfter(
                    record=record, before_settings=was, before=before, after_settings=now,
                    after=framed_section(workspace, state, record, options, look=now),
                ))
        except Exception as exc:
            raise Refused("RENDER_FAILED", message=str(exc)) from exc
    written = set_appearance(job, ids, settings)
    return Preprocessed(written=written, pictures=pairs)


# --- the preprocessed channel ------------------------------------------------


def set_preprocessed(
    job: Job,
    workspace: Workspace,
    section_ids: Sequence[str] | None,
    *,
    channel_weights: Sequence[float] | None = None,
    clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    reset: bool = False,
    shown: Sequence[SliceState] = (),
    options: DisplayOptions | None = None,
) -> Preprocessed:
    """Set the preprocessed channel's recipe, stack-wide (*section_ids* None)
    or as an override for those sections; *reset* returns them to the
    default (a section reset removes its override, so it follows the stack).
    One undo step.

    The recipe is checked first (:func:`langslice.core.appearance.validate_settings`,
    ``BAD_ARGS``): *channel_weights* is one weight per raw channel, empty for
    automatic weights (``CHANNEL_COUNT_MISMATCH`` for a section with another
    channel count), and every
    id must be a section (``UNKNOWN_SLICE_IDS``). With *options*, each section
    in *shown* is pictured BEFORE and AFTER (:func:`preprocess`).
    """
    state = job.state
    ids = None if section_ids is None else [str(name) for name in section_ids]
    if ids is not None:
        unknown = [name for name in ids if state.by_id(name) is None]
        if unknown or not ids:
            raise unknown_sections(state, unknown)
    scope = state.in_order() if ids is None else [state.by_id(name) for name in ids]
    settings: dict[str, Any] | None = None
    if not reset:
        try:
            settings = looks.validate_settings(
                channel_weights=list(channel_weights or []) or None,
                clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=n4, denoise=denoise,
            )
        except ValueError as exc:
            raise Refused("BAD_ARGS", message=str(exc)) from exc
        weights = settings["channel_weights"]
        if weights is not None:
            mismatched = {
                record.id: list(names) for record in scope if record is not None
                and len(names := workspace.section_channels(record.id)[0]) != len(weights)
            }
            if mismatched:
                raise Refused("CHANNEL_COUNT_MISMATCH", weights=len(weights),
                              channels=mismatched)
    return preprocess(job, workspace, ids, settings, shown=shown, options=options)


# --- the raw channels' display properties ------------------------------------


@dataclass(frozen=True)
class ChannelPropertiesSet:
    """What :func:`set_channel_properties` wrote and what is now in force."""

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


def set_channel_properties(
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
