"""How sections look: the ``view`` and ``fit`` appearances (:mod:`langslice.linear.appearance`)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.linear import appearance as looks

if TYPE_CHECKING:
    from langslice.linear.appearance import Look
    from langslice.linear.job import Job
    from langslice.linear.state import StackState


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
    """Write *settings* (validated, :func:`langslice.linear.appearance.validate_settings`;
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
