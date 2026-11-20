"""The flags of the ``langslice job`` commands that take them.

``init``, ``launch`` and ``wait`` each take the flags of their group in
:mod:`langslice.doors.commands.flags`, added to a parser that raises
:class:`Refusal` (the envelope answers it) instead of exiting; ``schema``
lists the same flags as data (:func:`flag_rows_of`).
"""

from __future__ import annotations

import argparse
import typing
from collections.abc import Mapping, Sequence
from typing import Any

from langslice.core.refusal import RefusalCode
from langslice.doors.commands.flags import FlagGroup, add_flags, flag_rows
from langslice.doors.commands.job.envelope import Envelope
from langslice.doors.commands.job.launch import ROLES

FLAG_GROUPS: dict[str, tuple[FlagGroup, ...]] = {
    "init": ("job", "resume", "init"),
    "launch": ("launch",),
    "wait": ("wait",),
}


class Refusal(Exception):
    """Carry a command's own refusal out of a call, its envelope made.

    The envelope is made where the refusal is raised and
    :func:`langslice.doors.commands.job.calls.execute` answers it as it stands.
    """

    def __init__(self, envelope: Envelope) -> None:
        super().__init__(envelope.error)
        self.envelope = envelope


class JobParser(argparse.ArgumentParser):
    """argparse that raises instead of exiting (the envelope reports it)."""

    def error(self, message: str) -> typing.NoReturn:
        raise Refusal(Envelope.failure(RefusalCode.BAD_ARGUMENT, message))


def choices_of(command: str) -> Mapping[str, Sequence[str]]:
    """The values of the flags of *command* that the command knows and the
    spec does not (``--viewer``'s viewers belong to the pictures)."""
    from langslice.core.pictures.opening import VIEWER_LIMITS

    if command == "init":
        return {"--viewer": sorted(VIEWER_LIMITS)}
    if command == "launch":
        return {"--role": list(ROLES)}
    return {}


def flag_parser(command: str) -> JobParser:
    """The parser of the flags of *command* (``init``, ``launch`` or ``wait``)."""
    parser = JobParser(prog=f"langslice job FOLDER {command}", add_help=False)
    for group in FLAG_GROUPS[command]:
        add_flags(parser, group, choices=choices_of(command))
    return parser


def flag_rows_of(command: str) -> list[dict[str, Any]]:
    """The flags of *command* as data, the ones :func:`flag_parser` adds."""
    return flag_rows(FLAG_GROUPS[command], choices=choices_of(command))
