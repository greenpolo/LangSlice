"""The job flags: how the commands that open a stack take each setting.

The settings, their flags' spellings and order and the job's defaults are
the settings table's (:data:`langslice.job.settings.schema.SETTINGS`); this
module holds what only the commands have: each flag's ``--help`` text, how
argparse reads it and which commands add it (:data:`FLAGS`).
:func:`add_flags` adds one group of them to a command's parser, in the
table's order.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from langslice.job.settings.schema import (
    CLI_TASKS,
    REGISTRATION_TASKS,
    SETTINGS,
    Setting,
)
from langslice.job.settings.tool_table import valid_names

#: Which commands add a flag: ``job`` every command that opens a stack
#: (``linear run``, ``init``, ``mcp``), ``plugins`` and ``session`` only
#: ``linear run``, ``resume`` ``linear run`` and ``init``, ``init`` only
#: ``init``.
FlagGroup = Literal["job", "plugins", "session", "resume", "init"]

#: The actions of a flag that takes no value: it switches the setting.
SWITCHES = ("store_true", "store_false")

#: The help of ``--plane``.
PLANE_HELP = (
    "Slicing plane (normal axis). Position is interpreted along this axis "
    "(AP for coronal, ML for sagittal, DV for horizontal)."
)


@dataclass(frozen=True, kw_only=True)
class Flag:
    """How the commands take one flag of a setting."""

    #: Its ``--help`` text.
    help: str
    #: Which commands add it.
    group: FlagGroup = "job"
    #: The attribute it is parsed into; None: the one argparse makes of the
    #: spelling (the setting's name, ``.`` written ``_``).
    dest: str | None = None
    #: argparse's action; None: the value is stored. An ``append`` flag
    #: starts from an empty list.
    action: str | None = None
    #: What the text is read as (``str``: as it is).
    type: Callable[[str], object] = str
    #: The value's name in the help.
    metavar: str | None = None
    #: Without the flag the value is the setting's default; False: None,
    #: and the command makes the value (``--tasks``) or the flag carries
    #: JSON text (``--frozen``).
    table_default: bool = True
    #: argparse refuses a value not among the setting's choices.
    checked: bool = False


#: Every job flag, by spelling.
FLAGS: dict[str, Flag] = {
    "--tasks": Flag(
        table_default=False,
        help="Comma-separated subset of position,transform,nonlinear: "
        "position places and orders the sections, transform aligns them "
        "in-plane, nonlinear deforms each linearly aligned section "
        "(fireants_syn, and trace_borders with an image model). A task left "
        "out takes its answer from the flags that supply it (--positions, "
        f"--transforms ...). Default: {CLI_TASKS}; with --registration, "
        f"{REGISTRATION_TASKS} (the imported linear registration kept as it "
        "is)",
    ),
    "--atlas": Flag(help="BrainGlobe atlas name"),
    "--plane": Flag(checked=True, help=PLANE_HELP),
    "--nonlinear-provider": Flag(
        metavar="PROVIDER",
        help="Image provider for the nonlinear task's trace_borders "
        "(openai-oauth, openai-api, gemini-api), or none: the nonlinear task "
        "then fits deformations to the stain alone, with no image model",
    ),
    "--nonlinear-image-model": Flag(
        metavar="MODEL",
        help="Image model for the nonlinear tool; default is the provider's "
        "image model",
    ),
    "--nonlinear-image-prompt": Flag(
        metavar="FILE",
        help="A UTF-8 text file holding the base prompt trace_borders sends "
        "the image model, in place of LangSlice's own "
        "(NonlinearSpec.image_prompt); {plane} in it becomes the section "
        "plane. It must describe the provider's attachment order, and its "
        "traces are recorded as an untested profile. The agent's "
        "per-section edit starts from it. Default: LangSlice's prompt for "
        "the provider",
    ),
    "--pixel-size-um": Flag(
        metavar="UM|JSON",
        help="Micrometres per pixel of the section images: one number for "
        "every file, or a JSON file or inline mapping of filenames to "
        "micrometres per pixel. Default (and for a file the mapping does not "
        "name): read from each file's TIFF/OME tags",
    ),
    "--preprocess": Flag(
        checked=True,
        help="How each section's preprocessed image is made: auto blends the "
        "channels with local contrast enhancement, none takes the file as "
        "read, in gray. It is the image the agent sees first, the one the "
        "image model is shown, and the one every fit reads unless the agent "
        "names another section image. The image files are never modified",
    ),
    "--image-resolution": Flag(
        checked=True,
        help="Size of the pictures the agent is shown; auto lets it choose "
        "per call",
    ),
    "--disabled": Flag(
        action="append",
        metavar="NAME[,NAME...]",
        help="Tools and tool options the agent does not get, on top of what "
        "the tasks give (repeatable): a tool (zoom), or tool:option "
        "(look:positioning, interactive_transform:flip, which keeps the "
        "agent from mirroring sections). Valid names: "
        + ", ".join(valid_names()),
    ),
    "--transform-hemisphere-cue": Flag(
        metavar="TEXT",
        help="What marks a hemisphere in these sections (a notch, an "
        "injection...)",
    ),
    "--position-thickness-um": Flag(
        type=int,
        metavar="UM",
        help="Known section thickness in microns; omit if unknown",
    ),
    "--position-interval-um": Flag(
        type=int,
        metavar="UM",
        help="Known section interval in microns; omit to infer spacing",
    ),
    "--position-strict-interval": Flag(
        action="store_true",
        help="Sections must sit exactly one interval apart",
    ),
    "--transform-angles": Flag(
        action="store_true", help="Let the agent set the cutting angles"
    ),
    "--pitch": Flag(type=float, help="Host-given cutting pitch, degrees"),
    "--yaw": Flag(type=float, help="Host-given cutting yaw, degrees"),
    "--section-angles": Flag(
        metavar="JSON",
        help="Host-given cutting angles per section (a registration made "
        "elsewhere): JSON file or inline mapping of filenames to "
        '{"pitch": deg, "yaw": deg}; not with --pitch/--yaw',
    ),
    "--facts": Flag(
        action="append",
        metavar="TEXT",
        help="A fact about this stack, passed to the agent verbatim "
        "(repeatable)",
    ),
    "--positions": Flag(
        metavar="JSON",
        help="Host-supplied positions: a JSON file path or inline JSON "
        "mapping filename -> mm",
    ),
    "--registration": Flag(
        metavar="FILE",
        help="A linear registration made elsewhere, imported as every "
        "section's position, cutting angles, orientation and in-plane "
        "transform: QuickNII or VisuAlign JSON/XML, DeepSlice CSV/JSON/XML, "
        "or a LangSlice registration.json. Not with --positions, "
        "--transforms, --orientation, --pitch/--yaw, --section-angles or "
        "--transform-angles. VisuAlign markers are not imported",
    ),
    "--order": Flag(
        metavar="JSON",
        help="Host-supplied order: a JSON file path or inline JSON list of "
        "filenames",
    ),
    "--transforms": Flag(
        metavar="JSON",
        help="Host-supplied calibrated linear transforms: JSON file or "
        "inline mapping of filenames to saved transform records",
    ),
    "--orientation": Flag(
        metavar="JSON",
        help="Host-supplied orientation: JSON file or inline mapping of "
        'filenames to {"flip": true|false, "rotation_deg": 0|90|180|270}',
    ),
    "--frozen": Flag(
        metavar="JSON",
        table_default=False,
        help="What the agent may not change: JSON file or inline mapping of a "
        'filename (or "*" for every section) to a list of properties, any of '
        "position, angles, orientation, affine, deformation; a frozen "
        "property keeps the value the supplied inputs give it",
    ),
    "--damaged": Flag(
        metavar="JSON",
        help="Notes on damaged tissue: JSON file or inline mapping of "
        "filenames to notes; notes alone do not exclude atlas regions from "
        "fits",
    ),
    "--out": Flag(
        help="Results JSON path. Default: <job folder>/exports/"
        "linear_results.json",
    ),
    "--job-dir": Flag(
        metavar="PATH",
        help="Put the job folder (checkpoint, undo history, pictures, "
        "results) here instead of <image_folder>/langslice. A folder holding "
        "the job of another image folder is refused",
    ),
    "--position-playbook": Flag(
        action="store_true",
        help="Add optional positioning guidance: compare candidate planes "
        "and regions, write supported placements, revisit uncertainty and "
        "review the stack",
    ),
    "--position-notes": Flag(
        metavar="TEXT",
        help="The user's notes for the position task, shown to the agent "
        "with the task (one line each; PositionSpec.notes)",
    ),
    "--transform-notes": Flag(
        metavar="TEXT",
        help="The user's notes for the transform task, shown to the agent "
        "with the task (one line each; TransformSpec.notes)",
    ),
    "--nonlinear-notes": Flag(
        metavar="TEXT",
        help="The user's notes for the nonlinear task, shown to the agent "
        "with the task (one line each; NonlinearSpec.notes)",
    ),
    "--plugins": Flag(
        group="plugins",
        action="append",
        metavar="MODULE:FACTORY",
        help="Add the tools of a tool plugin to the agent's toolbox "
        "(repeatable): a Python module and the function in it that returns "
        "the tools. Importing it runs its code; the job records which plugin "
        "ran (docs/developers/plugins.md)",
    ),
    "--plugin-config": Flag(
        group="plugins",
        action="append",
        metavar="MODULE:FACTORY=JSON",
        help="Options for a plugin's factory (repeatable): the plugin's "
        "reference, '=', then a JSON object or the path of a JSON file",
    ),
    "--model": Flag(group="session", help="Model name for the agent session"),
    "--reasoning": Flag(
        group="session",
        checked=True,
        help="Reasoning effort for models that expose one. Default: the "
        "provider's own",
    ),
    "--no-debrief": Flag(
        group="session",
        dest="debrief",
        action="store_false",
        help="Skip the post-submit debrief question (what tools the agent "
        "missed)",
    ),
    "--force-view": Flag(
        group="session",
        action="store_true",
        help="The change and fitting tools always return their picture: "
        "their view argument is not offered (JobSpec.force_view)",
    ),
    "--nonlinear-require-deformation": Flag(
        group="session",
        action="store_true",
        help="Nonlinear task: submit is refused until every section carries "
        "a deformation (NonlinearSpec.require_deformation); otherwise a "
        "deformation is the agent's choice",
    ),
    "--no-resume": Flag(
        group="resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoint and start over",
    ),
    "--notes": Flag(
        group="init",
        metavar="TEXT",
        help="The user's notes for this job (job.json), which every door "
        "gives the registration agent",
    ),
    "--viewer": Flag(
        group="init",
        help="Which model reads the pictures (job.json; default claude): it "
        "sets their largest size",
    ),
}


def group_flags(group: FlagGroup) -> Iterator[tuple[str, Flag, Setting]]:
    """Yield one group's flags in the table's order.

    Parameters
    ----------
    group
        Which commands' flags (:data:`FlagGroup`).

    Yields
    ------
    spelling : str
        The flag as it is typed.
    flag : Flag
        How the commands take it.
    row : Setting
        The setting it gives.
    """
    for row in SETTINGS:
        for spelling in row.flags:
            if FLAGS[spelling].group == group:
                yield spelling, FLAGS[spelling], row


def add_flags(
    parser: argparse.ArgumentParser,
    group: FlagGroup,
    *,
    choices: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Add one group's flags to a command's parser, in the table's order.

    Parameters
    ----------
    parser
        The command's parser.
    group
        Which commands' flags (:data:`FlagGroup`).
    choices
        The values of a flag that the command knows and the table does not,
        by spelling (``--viewer``'s viewers belong to the pictures).
    """
    known = choices or {}
    for spelling, flag, row in group_flags(group):
        listed = known.get(spelling)
        if listed is None and flag.checked:
            listed = row.choices
        values = None if listed is None else list(listed)
        dest = flag.dest or spelling.removeprefix("--").replace("-", "_")
        default = row.default if flag.table_default else None
        _add(parser, spelling, flag, dest, default, values)
        if spelling == row.flags[0]:
            for alias in row.aliases:  # read, never listed, never a default
                _add(parser, alias, flag, dest, argparse.SUPPRESS, values)


def _add(
    parser: argparse.ArgumentParser,
    spelling: str,
    flag: Flag,
    dest: str,
    default: object,
    values: list[str] | None,
) -> None:
    """Add one spelling of a flag to *parser*, as :func:`add_flags` does.

    A spelling without a default (``argparse.SUPPRESS``) has no help either.
    """
    hidden = default is argparse.SUPPRESS
    text = argparse.SUPPRESS if hidden else flag.help
    if flag.action in SWITCHES:
        parser.add_argument(
            spelling, dest=dest, action=flag.action, default=default, help=text
        )
    elif flag.action == "append":  # each parse appends to its own list
        empty: object = default if hidden else []
        parser.add_argument(
            spelling,
            dest=dest,
            action=flag.action,
            default=empty,
            metavar=flag.metavar,
            help=text,
        )
    else:
        parser.add_argument(
            spelling,
            dest=dest,
            default=default,
            type=flag.type,
            choices=values,
            metavar=flag.metavar,
            help=text,
        )
