"""The job flags: how the commands that open a stack take each setting.

The settings, their flags' spellings, help text and defaults are the job
spec's (:class:`langslice.job.settings.spec.JobSpec`, walked by
:func:`langslice.job.settings.walk.leaves`: a flag is derived from a field's
path, its help is the field's description, its default the field's); this
module holds what only the commands have: the flags that are no field of the
spec (:data:`COMMAND_ONLY`), the small converters that read a flag's text
(:func:`load_json_arg`, :func:`names`) and the arguments' way back into a
spec (:func:`values_from_args`). :func:`add_flags` adds one group of the
flags to a command's parser.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langslice.job.settings.spec import JobSpec
from langslice.job.settings.walk import Cli, FlagGroup, Kind, Leaf, leaves

SWITCHES = ("store_true", "store_false")

APPENDS: tuple[Kind, ...] = ("repeat", "repeat_csv", "pairs")


@dataclass(frozen=True, kw_only=True)
class Flag:
    """One flag as the commands add it."""

    flag: str
    aliases: tuple[str, ...] = ()
    dest: str
    help: str
    group: FlagGroup = "job"
    metavar: str | None = None
    kind: Kind = "value"
    number: type | None = None
    switch: bool = False
    default: Any = None
    choices: tuple[str, ...] | None = None
    path: str | None = None
    deferred: bool = False

    @property
    def action(self) -> str:
        """argparse's action: a switch, a repeated flag, or the plain value."""
        if self.switch:
            return "store_false" if self.default is True else "store_true"
        if self.kind in APPENDS:
            return "append"
        return "store"


COMMAND_ONLY: tuple[Flag, ...] = (
    Flag(
        flag="--registration",
        dest="registration",
        metavar="FILE",
        help="A linear registration made elsewhere, imported as every "
        "section's position, cutting angles, orientation and in-plane "
        "transform: QuickNII or VisuAlign JSON/XML, DeepSlice CSV/JSON/XML, "
        "or a LangSlice registration.json. Not with --positions, "
        "--transforms, --orientation, --pitch/--yaw, --section-angles or "
        "--transform-angles. VisuAlign markers are not imported",
    ),
    Flag(
        flag="--pitch",
        dest="pitch",
        number=float,
        help="Host-given cutting pitch, degrees",
    ),
    Flag(
        flag="--yaw",
        dest="yaw",
        number=float,
        help="Host-given cutting yaw, degrees",
    ),
    Flag(
        flag="--section-angles",
        dest="section_angles",
        metavar="JSON",
        kind="json",
        help="Host-given cutting angles per section (a registration made "
        "elsewhere): JSON file or inline mapping of filenames to "
        '{"pitch": deg, "yaw": deg}; not with --pitch/--yaw',
    ),
    Flag(
        flag="--notes",
        dest="notes",
        group="init",
        metavar="TEXT",
        help="The user's notes for this job (job.json), which every door "
        "gives the registration agent",
    ),
    Flag(
        flag="--viewer",
        dest="viewer",
        group="init",
        help="Which model reads the pictures (job.json; default claude): it "
        "sets their largest size",
    ),
)


def load_json_arg(value: str | None, flag: str = "JSON argument") -> object | None:
    """A JSON argument, given inline or as a path to a JSON file.

    An inline object or list (``{...}``, ``[...]``) is parsed as JSON whatever
    its length (never tried as a file name: a real ``--transforms`` map is
    longer than any). Anything else is a path when that file exists, else
    inline JSON. Neither is a ``ValueError`` naming *flag*.
    """
    if not value:
        return None
    path: Path | None = None
    if not value.lstrip().startswith(("{", "[")):
        try:
            candidate = Path(value).expanduser()
            path = candidate if candidate.is_file() else None
        except OSError:
            path = None
    if path is not None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{flag}: could not read JSON from {path}: {exc}") from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag}: neither an existing JSON file nor inline JSON ({exc})") from exc


def read_text_arg(value: str | None, flag: str) -> str | None:
    """The text of the UTF-8 file *value* names (None without one); a file that
    cannot be read is a ``ValueError`` naming *flag*."""
    if not value:
        return None
    path = Path(value).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{flag}: could not read {path}: {exc}") from exc


def names(items: Sequence[str]) -> list[str]:
    """Split a repeatable ``NAME[,NAME...]`` flag into its names, in the order
    given, without blanks."""
    return [name.strip() for item in items for name in item.split(",") if name.strip()]


def _flag_of(leaf: Leaf, cli: Cli) -> Flag:
    switch = leaf.is_switch
    return Flag(
        flag=leaf.flag,
        aliases=cli.aliases,
        dest=leaf.dest,
        help=leaf.description,
        group=cli.group,
        metavar=cli.metavar,
        kind=cli.kind,
        number=leaf.number,
        switch=switch,
        default=leaf.default,
        choices=leaf.choices,
        path=leaf.path,
        deferred=cli.deferred,
    )


def all_flags(group: FlagGroup) -> list[Flag]:
    """The flags of one group: the spec's settings in declaration order, then
    the flags that are no setting."""
    own = [
        _flag_of(leaf, leaf.cli)
        for leaf in leaves(JobSpec)
        if leaf.cli is not None and leaf.cli.group == group
    ]
    return [*own, *(flag for flag in COMMAND_ONLY if flag.group == group)]


def flag_names() -> dict[str, str]:
    """Each setting's flag by its path (``position.thickness_um``,
    ``inputs.pixel_size_um``), and each command-only flag by its destination
    (``notes``), for the window's prompt generator."""
    spelled = {leaf.path: leaf.flag for leaf in leaves(JobSpec) if leaf.cli is not None}
    spelled.update({flag.dest: flag.flag for flag in COMMAND_ONLY})
    return spelled


def add_flags(
    parser: argparse.ArgumentParser,
    group: FlagGroup,
    *,
    choices: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Add one group's flags to a command's parser.

    Parameters
    ----------
    parser
        The command's parser.
    group
        Which commands' flags (:data:`FlagGroup`).
    choices
        The values of a flag that the command knows and the spec does not,
        by spelling (``--viewer``'s viewers belong to the pictures).
    """
    known = choices or {}
    for flag in all_flags(group):
        values = _values(known.get(flag.flag), flag)
        _add(parser, flag.flag, flag, values, hidden=False)
        for alias in flag.aliases:
            _add(parser, alias, flag, values, hidden=True)


def flag_rows(
    groups: Sequence[FlagGroup],
    *,
    choices: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return the flags of some groups as data, from the walk :func:`add_flags` reads.

    Parameters
    ----------
    groups
        Which commands' flags, in the order the command adds them.
    choices
        As :func:`add_flags`.

    Returns
    -------
    list of dict
        Per flag its ``flag``, ``aliases`` (none: an old spelling is hidden),
        ``help``, ``default``, ``choices``, ``takes_value`` and ``metavar``.
    """
    known = choices or {}
    rows: list[dict[str, Any]] = []
    for group in groups:
        for flag in all_flags(group):
            switch = flag.switch
            values = (
                None if switch or flag.action == "append" else _values(known.get(flag.flag), flag)
            )
            rows.append(
                {
                    "flag": flag.flag,
                    "aliases": [],
                    "help": flag.help,
                    "default": [] if flag.action == "append" else _default(flag),
                    "choices": values,
                    "takes_value": not switch,
                    "metavar": None if switch else flag.metavar,
                }
            )
    return rows


def values_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The settings the parsed flags give, as :class:`JobSpec` takes them.

    A setting whose flag was not parsed, or not given and with no value of its
    own, is left out, so the spec's default stands. The plugin flags are the
    plugins' (:func:`langslice.doors.commands.linear.with_tool_plugins`), and
    ``--pitch``, ``--yaw`` and ``--section-angles`` make ``inputs.angles``.

    Parameters
    ----------
    args
        The parsed job flags.

    Returns
    -------
    dict
        Top-level settings, a task's section as its own dict, the answers
        under ``inputs``.

    Raises
    ------
    ValueError
        For a JSON or file argument that cannot be read, or angle flags that
        cannot be combined.
    """
    values: dict[str, Any] = {}
    for group in ("job", "session", "resume"):
        for flag in all_flags(group):
            if flag.path is None or not hasattr(args, flag.dest):
                continue
            value = _read(flag, getattr(args, flag.dest))
            if value is not None:
                _put(values, flag.path, value)
    angles = _angles(args)
    if angles is not None:
        _put(values, "inputs.angles", angles)
    return values


def _read(flag: Flag, given: Any) -> Any:
    """One flag's value in the form the spec takes, None when there is none."""
    if given is None or flag.kind == "pairs":
        return None
    if flag.kind == "json":
        return load_json_arg(given, flag.flag)
    if flag.kind == "file":
        return read_text_arg(given, flag.flag)
    if flag.kind == "csv":
        return names([given])
    if flag.kind == "repeat_csv":
        return names(given) or None
    if flag.kind == "repeat":
        return list(given) or None
    return given


def _put(values: dict[str, Any], path: str, value: Any) -> None:
    """Set *value* at the dotted *path* of *values*, a section made as needed."""
    section, _dot, key = path.rpartition(".")
    held = values
    for part in section.split(".") if section else ():
        held = held.setdefault(part, {})
    held[key] = value


def _angles(args: argparse.Namespace) -> dict[str, Any] | None:
    """``inputs.angles`` from ``--pitch`` / ``--yaw``, or ``--section-angles``."""
    pitch = getattr(args, "pitch", None)
    yaw = getattr(args, "yaw", None)
    angles: dict[str, Any] | None = None
    if pitch is not None or yaw is not None:
        angles = {"pitch": pitch or 0.0, "yaw": yaw or 0.0}
    per_section = load_json_arg(getattr(args, "section_angles", None), "--section-angles")
    if per_section is None:
        return angles
    if angles is not None:
        raise ValueError(
            "--section-angles gives each section its own plane; it cannot "
            "be combined with the stack-wide --pitch/--yaw"
        )
    if not isinstance(per_section, dict) or not all(
        isinstance(value, dict) for value in per_section.values()
    ):
        raise ValueError('--section-angles must map filenames to {"pitch": deg, "yaw": deg}')
    return per_section


def _default(flag: Flag) -> Any:
    """The default argparse is given: the setting's own, none for a flag the
    command works the default out of (``tasks``, the provider), a JSON or file
    flag, or a flag that is no setting."""
    if flag.path is None or flag.deferred or flag.kind in ("json", "file", "csv"):
        return None
    return flag.default


def _values(listed: Sequence[str] | None, flag: Flag) -> list[str] | None:
    """Return the values argparse accepts for a flag; None: any.

    *listed* is what the command knows; else the setting's own fixed list.
    """
    chosen = listed if listed is not None else flag.choices
    return None if chosen is None else list(chosen)


def _add(
    parser: argparse.ArgumentParser,
    spelling: str,
    flag: Flag,
    values: list[str] | None,
    *,
    hidden: bool,
) -> argparse.Action:
    """Add one spelling of a flag to *parser*, as :func:`add_flags` does.

    A hidden spelling (an old name of the flag) has no default and no help.
    Returns the parser's action for it.
    """
    text = argparse.SUPPRESS if hidden else flag.help
    default = argparse.SUPPRESS if hidden else _default(flag)
    if flag.switch:
        return parser.add_argument(
            spelling, dest=flag.dest, action=flag.action, default=default, help=text
        )
    if flag.action == "append":
        return parser.add_argument(
            spelling,
            dest=flag.dest,
            action="append",
            default=default if hidden else [],
            metavar=flag.metavar,
            help=text,
        )
    return parser.add_argument(
        spelling,
        dest=flag.dest,
        action="store",
        default=default,
        type=flag.number or str,
        choices=values,
        metavar=flag.metavar,
        help=text,
    )
