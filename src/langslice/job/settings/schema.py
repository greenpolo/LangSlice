"""Every job setting in one table: its name, type, default and spellings.

A job's settings are given through several doors (the flags of ``langslice
linear run``, of the agent CLI's ``init`` and of ``langslice mcp``, the
library's job creation) and kept in the job
folder's ``job.json``. :data:`SETTINGS` lists each setting once
(:class:`Setting`): its name, its type and the job's default, its
command-line flags, the library's keyword for it, its ``job.json`` key and
the doors that may set it; where a door's default is not the job's own the
row says what it is.

Every spelling follows the setting's name. A setting of a task's section is
named by section and key (``position.thickness_um``), a supplied answer by
``inputs.`` and its key. Its ``job.json`` key is ``spec.`` and the name; its
flag is the name with ``inputs.`` dropped and ``.`` and ``_`` written ``-``
(``--position-thickness-um``, ``--positions``), and ``--no-`` and the name
for a switch that turns an on setting off (``--no-debrief``); its library
keyword is the name, a section's key given in that section's
mapping (``position={"thickness_um": 40}``). The cutting angles alone are
given by other flags: ``--pitch`` and ``--yaw``, or ``--section-angles``.
The flags a setting was spelled with before are its ``aliases``: still
read, listed nowhere.

The filled form is :class:`~langslice.job.settings.spec.JobSpec`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

DEFAULT_TASKS: tuple[str, ...] = ("position", "transform")
ALL_TASKS: tuple[str, ...] = (*DEFAULT_TASKS, "nonlinear")

PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")

IMAGE_RESOLUTIONS: tuple[str, ...] = ("low", "medium", "high", "auto")

PREPROCESSING: tuple[str, ...] = ("auto", "none")

REASONING: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

MAX_PARALLEL_TRANSFORMS = 4

DEFAULT_ATLAS = "allen_mouse_25um"

CLI_TASKS = ",".join(DEFAULT_TASKS)
REGISTRATION_TASKS = "nonlinear"

ANSWER_PREFIX = "inputs."

Door = Literal["run", "init", "mcp", "library"]

COMMANDS: tuple[Door, ...] = ("run", "init", "mcp")
EVERY_DOOR: tuple[Door, ...] = (*COMMANDS, "library")
RUN_DOORS: tuple[Door, ...] = ("run", "library")
HOST_DOORS: tuple[Door, ...] = ("library",)

_CLI_TASKS_DEFAULT = f"{CLI_TASKS} ({REGISTRATION_TASKS} with --registration)"


@dataclass(frozen=True, kw_only=True)
class Setting:
    """One setting of a job, and how each door spells it."""

    name: str
    type: str
    default: object
    doors: tuple[Door, ...]
    json_key: str | None
    keyword: str | None = None
    flags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    choices: tuple[str, ...] | None = None
    door_defaults: Mapping[Door, str] = field(default_factory=dict[Door, str])


SETTINGS: tuple[Setting, ...] = (
    Setting(
        name="tasks",
        type="list[str]",
        default=list(DEFAULT_TASKS),
        doors=EVERY_DOOR,
        json_key="spec.tasks",
        keyword="tasks",
        flags=("--tasks",),
        choices=ALL_TASKS,
        door_defaults={
            "run": _CLI_TASKS_DEFAULT,
            "init": _CLI_TASKS_DEFAULT,
            "mcp": _CLI_TASKS_DEFAULT,
            "library": "nonlinear, plus transform unless every section has a "
            "supplied transform (nonlinear alone with registration=)",
        },
    ),
    Setting(
        name="atlas",
        type="str",
        default=DEFAULT_ATLAS,
        doors=EVERY_DOOR,
        json_key="spec.atlas",
        keyword="atlas",
        flags=("--atlas",),
    ),
    Setting(
        name="plane",
        type="str",
        default="coronal",
        doors=EVERY_DOOR,
        json_key="spec.plane",
        keyword="plane",
        flags=("--plane",),
        choices=PLANES,
    ),
    Setting(
        name="nonlinear.provider",
        type="str",
        default="none",
        doors=EVERY_DOOR,
        json_key="spec.nonlinear.provider",
        keyword="nonlinear",
        flags=("--nonlinear-provider",),
        aliases=("--image-provider",),
        door_defaults={
            "library": "none, or image_model='s provider",
            "run": "none; with --enabled trace_borders, the saved connection",
            "init": "none; with --enabled trace_borders, the saved connection",
            "mcp": "none; with --enabled trace_borders, the saved connection",
        },
    ),
    Setting(
        name="nonlinear.image_model",
        type="str | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.nonlinear.image_model",
        keyword="nonlinear",
        flags=("--nonlinear-image-model",),
        aliases=("--image-model",),
    ),
    Setting(
        name="nonlinear.image_prompt",
        type="str | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.nonlinear.image_prompt",
        keyword="nonlinear",
        flags=("--nonlinear-image-prompt",),
        aliases=("--image-prompt",),
    ),
    Setting(
        name="inputs.pixel_size_um",
        type="float | dict[str, float] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.pixel_size_um",
        keyword="pixel_size_um",
        flags=("--pixel-size-um",),
    ),
    Setting(
        name="preprocess",
        type="str",
        default="auto",
        doors=EVERY_DOOR,
        json_key="spec.preprocess",
        keyword="preprocess",
        flags=("--preprocess",),
        choices=PREPROCESSING,
    ),
    Setting(
        name="image_resolution",
        type="str",
        default="low",
        doors=EVERY_DOOR,
        json_key="spec.image_resolution",
        keyword="image_resolution",
        flags=("--image-resolution",),
        choices=IMAGE_RESOLUTIONS,
    ),
    Setting(
        name="disabled",
        type="list[str]",
        default=[],
        doors=EVERY_DOOR,
        json_key="spec.disabled",
        keyword="disabled",
        flags=("--disabled",),
        aliases=("--disable",),
    ),
    Setting(
        name="enabled",
        type="list[str]",
        default=[],
        doors=EVERY_DOOR,
        json_key="spec.enabled",
        keyword="enabled",
        flags=("--enabled",),
        aliases=("--enable",),
    ),
    Setting(
        name="transform.hemisphere_cue",
        type="str",
        default="",
        doors=EVERY_DOOR,
        json_key="spec.transform.hemisphere_cue",
        keyword="transform",
        flags=("--transform-hemisphere-cue",),
        aliases=("--hemisphere-cue",),
    ),
    Setting(
        name="position.thickness_um",
        type="int | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.position.thickness_um",
        keyword="position",
        flags=("--position-thickness-um",),
        aliases=("--thickness",),
    ),
    Setting(
        name="position.interval_um",
        type="int | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.position.interval_um",
        keyword="position",
        flags=("--position-interval-um",),
        aliases=("--interval",),
    ),
    Setting(
        name="position.start_mm",
        type="float | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.position.start_mm",
        keyword="position",
        flags=("--position-start-mm",),
        aliases=("--start-mm",),
    ),
    Setting(
        name="position.strict_interval",
        type="bool",
        default=False,
        doors=EVERY_DOOR,
        json_key="spec.position.strict_interval",
        keyword="position",
        flags=("--position-strict-interval",),
        aliases=("--strict-interval",),
    ),
    Setting(
        name="transform.angles",
        type="bool",
        default=False,
        doors=EVERY_DOOR,
        json_key="spec.transform.angles",
        keyword="transform",
        flags=("--transform-angles",),
        aliases=("--angles",),
    ),
    Setting(
        name="inputs.angles",
        type="{pitch, yaw} | dict[str, {pitch, yaw}] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.angles",
        keyword="angles",
        flags=("--pitch", "--yaw", "--section-angles"),
    ),
    Setting(
        name="facts",
        type="list[str]",
        default=[],
        doors=EVERY_DOOR,
        json_key="spec.facts",
        keyword="facts",
        flags=("--facts",),
        aliases=("--fact",),
    ),
    Setting(
        name="inputs.positions",
        type="dict[str, float] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.positions",
        keyword="positions",
        flags=("--positions",),
    ),
    Setting(
        name="registration",
        type="path | None",
        default=None,
        doors=EVERY_DOOR,
        json_key=None,
        keyword="registration",
        flags=("--registration",),
    ),
    Setting(
        name="inputs.order",
        type="list[str] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.order",
        keyword="inputs",
        flags=("--order",),
    ),
    Setting(
        name="inputs.transforms",
        type="dict[str, transform] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.transforms",
        keyword="transforms",
        flags=("--transforms",),
    ),
    Setting(
        name="inputs.orientation",
        type="dict[str, {flip, rotation_deg}] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.orientation",
        keyword="orientation",
        flags=("--orientation",),
    ),
    Setting(
        name="frozen",
        type="dict[str, list[str]]",
        default={},
        doors=EVERY_DOOR,
        json_key="spec.frozen",
        keyword="frozen",
        flags=("--frozen",),
    ),
    Setting(
        name="inputs.damaged",
        type="dict[str, str] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.damaged",
        keyword="inputs",
        flags=("--damaged",),
    ),
    Setting(
        name="inputs.damaged_regions",
        type="dict[str, list[str]] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.damaged_regions",
        keyword="inputs",
        flags=("--damaged-regions",),
    ),
    Setting(
        name="out",
        type="str | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.out",
        keyword="out",
        flags=("--out",),
    ),
    Setting(
        name="job_dir",
        type="str | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.job_dir",
        keyword="job_dir",
        flags=("--job-dir",),
    ),
    Setting(
        name="position.playbook",
        type="bool",
        default=False,
        doors=EVERY_DOOR,
        json_key="spec.position.playbook",
        keyword="position",
        flags=("--position-playbook",),
        aliases=("--playbook",),
    ),
    Setting(
        name="position.notes",
        type="str",
        default="",
        doors=EVERY_DOOR,
        json_key="spec.position.notes",
        keyword="position",
        flags=("--position-notes",),
    ),
    Setting(
        name="transform.notes",
        type="str",
        default="",
        doors=EVERY_DOOR,
        json_key="spec.transform.notes",
        keyword="transform",
        flags=("--transform-notes",),
    ),
    Setting(
        name="nonlinear.notes",
        type="str",
        default="",
        doors=EVERY_DOOR,
        json_key="spec.nonlinear.notes",
        keyword="nonlinear",
        flags=("--nonlinear-notes",),
    ),
    Setting(
        name="plugins",
        type="list[str]",
        default=[],
        doors=("run", "library"),
        json_key="spec.plugins",
        keyword="plugins",
        flags=("--plugins",),
        aliases=("--tool-plugin",),
    ),
    Setting(
        name="plugin_config",
        type="dict[str, Any]",
        default={},
        doors=("run", "library"),
        json_key="spec.plugin_config",
        keyword="plugin_config",
        flags=("--plugin-config",),
        aliases=("--tool-plugin-config",),
    ),
    Setting(
        name="model",
        type="str | None",
        default=None,
        doors=RUN_DOORS,
        json_key="spec.model",
        keyword="model",
        flags=("--model",),
        door_defaults={
            "run": "the model the LangSlice window's Settings saved, else the "
            "provider's",
        },
    ),
    Setting(
        name="reasoning",
        type="str | None",
        default=None,
        doors=RUN_DOORS,
        json_key="spec.reasoning",
        keyword="reasoning",
        flags=("--reasoning",),
        choices=REASONING,
    ),
    Setting(
        name="debrief",
        type="bool",
        default=True,
        doors=RUN_DOORS,
        json_key="spec.debrief",
        keyword="debrief",
        flags=("--no-debrief",),
        door_defaults={"library": "False"},
    ),
    Setting(
        name="force_view",
        type="bool",
        default=False,
        doors=RUN_DOORS,
        json_key="spec.force_view",
        keyword="force_view",
        flags=("--force-view",),
    ),
    Setting(
        name="nonlinear.require_deformation",
        type="bool",
        default=False,
        doors=RUN_DOORS,
        json_key="spec.nonlinear.require_deformation",
        keyword="nonlinear",
        flags=("--nonlinear-require-deformation",),
        aliases=("--require-deformation",),
    ),
    Setting(
        name="resume",
        type="bool",
        default=True,
        doors=("run", "init", "library"),
        json_key="spec.resume",
        keyword="resume",
        flags=("--no-resume",),
        aliases=("--fresh",),
        door_defaults={
            "mcp": "True: a folder's own job keeps its saved settings",
        },
    ),
    Setting(
        name="notes",
        type="str | None",
        default=None,
        doors=("init",),
        json_key="notes",
        flags=("--notes",),
    ),
    Setting(
        name="viewer",
        type="str | None",
        default=None,
        doors=("init",),
        json_key="viewer",
        flags=("--viewer",),
        door_defaults={"init": "claude, when job.json names none"},
    ),
    Setting(
        name="image_folder",
        type="str",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.image_folder",
        keyword="images",
        door_defaults={
            "run": "the command's folder",
            "init": "the command's folder",
            "mcp": "the command's folder, else start_job's",
        },
    ),
    Setting(
        name="host_preprocessing",
        type="dict[str, Any] | None",
        default=None,
        doors=HOST_DOORS,
        json_key="spec.host_preprocessing",
        keyword="host_preprocessing",
    ),
    Setting(
        name="transform.max_parallel",
        type="int",
        default=MAX_PARALLEL_TRANSFORMS,
        doors=HOST_DOORS,
        json_key="spec.transform.max_parallel",
        keyword="transform",
    ),
    Setting(
        name="inputs.channel_names",
        type="list[str] | None",
        default=None,
        doors=HOST_DOORS,
        json_key="spec.inputs.channel_names",
        keyword="inputs",
    ),
)


def setting(name: str) -> Setting:
    """Return the row of one setting.

    Parameters
    ----------
    name
        The setting's name (:attr:`Setting.name`).

    Returns
    -------
    Setting
        Its row of :data:`SETTINGS`.
    """
    return next(row for row in SETTINGS if row.name == name)


def spec_keys() -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    """Return the keys the settings object of ``job.json`` takes.

    Returns
    -------
    top : frozenset of str
        The keys of the object itself (``atlas``, ``position``).
    sections : dict
        Each task section's own keys (``position`` -> ``thickness_um``...).
    """
    top: set[str] = set()
    sections: dict[str, set[str]] = {}
    for row in SETTINGS:
        if row.json_key is None or not row.json_key.startswith("spec."):
            continue
        name, _dot, key = row.json_key.removeprefix("spec.").partition(".")
        top.add(name)
        if key and not row.name.startswith(ANSWER_PREFIX):
            sections.setdefault(name, set()).add(key)
    frozen = {name: frozenset(keys) for name, keys in sections.items()}
    return frozenset(top), frozen


def flag_spellings() -> dict[str, str]:
    """Return each setting's first command-line flag, by setting name.

    Returns
    -------
    dict
        Setting name -> the flag as typed, for the settings that have one.
    """
    return {row.name: row.flags[0] for row in SETTINGS if row.flags}
