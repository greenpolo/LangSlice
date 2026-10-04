"""How sections look: the ``view`` and ``fit`` appearances (:mod:`langslice.core.appearance`)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core import appearance as looks
from langslice.ops.refusal import Refused

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

    targets: list[str]
    #: The sections written, or None for the stack setting.
    section_ids: list[str] | None
    #: Per target: ``{"stack": settings}`` or ``{"sections": {id: settings}}``
    #: (None is the default appearance).
    in_force: dict[str, Any] = field(default_factory=dict)


def planned_settings(
    state: StackState, target: str, section_ids: Sequence[str] | None, settings: Look,
    section_id: str,
) -> Look:
    """The settings *section_id* would have for *target* after that write.

    Nothing is written: a door draws the AFTER picture of a write it has not
    made yet, and makes it only when every picture succeeded.
    """
    scratch = copy.copy(state)
    scratch.appearance = copy.deepcopy(state.appearance)
    looks.set_settings(scratch, target, None if section_ids is None else list(section_ids),
                       settings)
    return looks.section_settings(scratch, target, section_id)


def set_appearance(
    job: Job, targets: Sequence[str], section_ids: Sequence[str] | None, settings: Look,
) -> AppearanceSet:
    """Write *settings* (validated, :func:`langslice.core.appearance.validate_settings`;
    None = back to the default) for each target, stack-wide (*section_ids*
    None) or for those sections. One undo step.
    """
    ids = None if section_ids is None else list(section_ids)
    before = job.snapshot()
    for name in targets:
        looks.set_settings(job.state, name, ids, settings)
    job.commit(before)
    state = job.state
    in_force = {
        name: (
            {"sections": {section: looks.section_settings(state, name, section)
                          for section in ids}}
            if ids else {"stack": looks.section_settings(state, name, "")}
        )
        for name in targets
    }
    return AppearanceSet(targets=list(targets), section_ids=ids, in_force=in_force)


@dataclass(frozen=True)
class BeforeAfter:
    """One section's appearance before and after a :func:`preprocess` call:
    the settings and the tissue-framed picture of each (no caption: the door
    labels them)."""

    record: SliceState
    before_settings: Look
    before: Image.Image
    after_settings: Look
    after: Image.Image


@dataclass(frozen=True)
class Preprocessed:
    """What :func:`preprocess` wrote, and the pictures of the target it drew."""

    written: AppearanceSet
    #: The target pictured (the first of the call's; "both" writes one
    #: setting to both).
    target: str
    pictures: list[BeforeAfter] = field(default_factory=list)


def preprocess(
    job: Job,
    workspace: Workspace,
    targets: Sequence[str],
    section_ids: Sequence[str] | None,
    settings: Look,
    *,
    shown: Sequence[SliceState] = (),
    options: DisplayOptions | None = None,
) -> Preprocessed:
    """Set the appearance (:func:`set_appearance`), picturing it first.

    For each section in *shown*, with *options*: the first target's
    appearance BEFORE the call (drawn first, from the settings as they
    stand) and AFTER (from the settings the write will leave,
    :func:`planned_settings`), each tissue-framed
    (:func:`langslice.core.display.framed_section`). The write happens only
    once every picture is drawn: a picture that fails refuses the call
    (``RENDER_FAILED``), nothing written.
    """
    from langslice.core.display import framed_section

    state = job.state
    pictured = targets[0]
    ids = None if section_ids is None else list(section_ids)
    pairs: list[BeforeAfter] = []
    if options is not None:
        try:
            earlier = [
                (record, looks.section_settings(state, pictured, record.id),
                 framed_section(workspace, state, record, options,
                                look=looks.section_settings(state, pictured, record.id)))
                for record in shown
            ]
            for record, was, before in earlier:
                now = planned_settings(state, pictured, ids, settings, record.id)
                pairs.append(BeforeAfter(
                    record=record, before_settings=was, before=before, after_settings=now,
                    after=framed_section(workspace, state, record, options, look=now),
                ))
        except Exception as exc:
            raise Refused("RENDER_FAILED", message=str(exc)) from exc
    written = set_appearance(job, targets, ids, settings)
    return Preprocessed(written=written, target=pictured, pictures=pairs)
