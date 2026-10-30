"""Every job setting in one table: its name, type, default and spellings.

A job's settings are given through several doors (the flags of ``langslice
linear run``, of the agent CLI's ``init`` and of ``langslice mcp``, the
library's job creation, the Fiji worker's request) and kept in the job
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

#: The tasks a job runs without being told, in the order every list gives.
DEFAULT_TASKS: tuple[str, ...] = ("position", "transform")
#: Every task, in the order they are listed everywhere else.
ALL_TASKS: tuple[str, ...] = (*DEFAULT_TASKS, "nonlinear")

#: The section planes.
PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")

#: How large the pictures the agent is shown are drawn: a long edge for the
#: opening images and one for every later picture per level
#: (:data:`langslice.core.pictures.sizes.PICTURE_EDGES`); "auto" lets the
#: agent pass ``resolution`` per call. Nothing a fit computes or a transform
#: stores depends on it, and the image model's inputs do not change.
IMAGE_RESOLUTIONS: tuple[str, ...] = ("low", "medium", "high", "auto")

#: How each section's ``preprocessed`` image is made, the one the agent sees
#: first, the image model is shown and every fit reads by default: the
#: adaptive preprocess, or the file as is.
PREPROCESSING: tuple[str, ...] = ("auto", "none")

#: The reasoning efforts a model may be run at.
REASONING: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: Most sections one transform-tool call may take
#: (``TransformSpec.max_parallel``).
MAX_PARALLEL_TRANSFORMS = 4

#: The atlas a job uses without being told.
DEFAULT_ATLAS = "allen_mouse_25um"

#: The tasks a command that opens a stack runs without ``--tasks``.
CLI_TASKS = ",".join(DEFAULT_TASKS)
#: ... and with ``--registration``: only the nonlinear step, on top of the
#: imported linear registration.
REGISTRATION_TASKS = "nonlinear"

#: The start of a supplied answer's name (``inputs.positions``).
ANSWER_PREFIX = "inputs."

#: A way a setting is given: ``run`` (``langslice linear run``), ``init``
#: (``langslice-job FOLDER init``), ``mcp`` (``langslice mcp``), ``library``
#: (``create_job``), ``abba`` (the Fiji worker's request).
Door = Literal["run", "init", "mcp", "library", "abba"]

#: The commands that take the job flags.
COMMANDS: tuple[Door, ...] = ("run", "init", "mcp")
#: The doors that take the job flags' settings, and the library and the
#: Fiji worker beside them.
EVERY_DOOR: tuple[Door, ...] = (*COMMANDS, "library", "abba")
#: The doors of a script or a coding agent: the commands and the library.
CODE_DOORS: tuple[Door, ...] = (*COMMANDS, "library")
#: The built-in agent's run and the doors that build one from code.
RUN_DOORS: tuple[Door, ...] = ("run", "library", "abba")
#: The doors of a host: the library and the Fiji worker.
HOST_DOORS: tuple[Door, ...] = ("library", "abba")

#: How each of the commands words its default of the tasks.
_CLI_TASKS_DEFAULT = f"{CLI_TASKS} ({REGISTRATION_TASKS} with --registration)"


@dataclass(frozen=True, kw_only=True)
class Setting:
    """One setting of a job, and how each door spells it."""

    #: The setting's name: its ``JobSpec`` field (``atlas``), a section's
    #: key (``position.thickness_um``) or a supplied answer
    #: (``inputs.positions``).
    name: str
    #: Its type, as ``JobSpec`` declares it or the supplied answer takes it.
    type: str
    #: The job's own default: what a job made without the setting holds
    #: (None for the image folder, which every job is given).
    default: object
    #: The doors that may set it.
    doors: tuple[Door, ...]
    #: Its key in ``job.json``, dotted through the objects; None: the job
    #: keeps it nowhere.
    json_key: str | None
    #: ``create_job``'s keyword for it; a section's key is given in that
    #: section's mapping (``position={"thickness_um": 40}``), a supplied
    #: answer without a keyword of its own in ``inputs``.
    keyword: str | None = None
    #: Its command-line flags, in the order the commands list them.
    flags: tuple[str, ...] = ()
    #: The flags it was spelled with before, read as its first flag and
    #: listed in no help.
    aliases: tuple[str, ...] = ()
    #: The values it takes, when they are a fixed list.
    choices: tuple[str, ...] | None = None
    #: A door's default where it is not :attr:`default`, in that door's
    #: terms.
    door_defaults: Mapping[Door, str] = field(default_factory=dict[Door, str])


#: Every setting, in the order the commands list their flags.
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
            "abba": "the host's choice, at least one",
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
        door_defaults={"abba": f"{DEFAULT_ATLAS}, the one atlas it takes"},
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
        door_defaults={"abba": "coronal, the one plane it takes"},
    ),
    Setting(
        name="nonlinear.provider",
        type="str",
        default="openai-oauth",
        doors=EVERY_DOOR,
        json_key="spec.nonlinear.provider",
        keyword="nonlinear",
        flags=("--nonlinear-provider",),
        aliases=("--image-provider",),
        door_defaults={"library": "none, or image_model='s provider"},
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
        doors=CODE_DOORS,
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
        door_defaults={"abba": "required: the snapshots' pixel size"},
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
        door_defaults={"abba": "True: the stack's angles are ABBA's to set"},
    ),
    Setting(
        name="inputs.angles",
        type="{pitch, yaw} | dict[str, {pitch, yaw}] | None",
        default=None,
        doors=EVERY_DOOR,
        json_key="spec.inputs.angles",
        keyword="angles",
        flags=("--pitch", "--yaw", "--section-angles"),
        door_defaults={"abba": "required: ABBA's angles for the stack"},
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
        door_defaults={"abba": "required: ABBA's positions"},
    ),
    Setting(
        name="registration",
        type="path | None",
        default=None,
        doors=CODE_DOORS,
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
        door_defaults={"abba": "the order of the positions"},
    ),
    Setting(
        name="inputs.transforms",
        type="dict[str, transform] | None",
        default=None,
        doors=CODE_DOORS,
        json_key="spec.inputs.transforms",
        keyword="transforms",
        flags=("--transforms",),
    ),
    Setting(
        name="inputs.orientation",
        type="dict[str, {flip, rotation_deg}] | None",
        default=None,
        doors=CODE_DOORS,
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
            "abba": "False: each request is a new run",
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
        door_defaults={"abba": "the request's preprocessing, else auto"},
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
