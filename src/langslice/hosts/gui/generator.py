"""The prompt generator's data: what the form offers, read from the code.

The prompt generator tab (``static/generator.js``) is a form that runs in the
page: it turns a person's choices into one message to paste into their agent
app, a short request and the job's ``langslice-job FOLDER init`` command,
which the main agent's ``register-brain`` skill runs. This module gives the
page what the choices are, so nothing in the form is written by hand twice:

- the tools and their options from
  :func:`langslice.job.settings.tool_table.catalog` (the scripting verbs left
  out: a script's, never the agent's to lose), each with its people's label
  (:mod:`langslice.job.settings.labels`), so a renamed or removed tool can
  never linger in the form;
- the tasks, the default ones and the tools that write each task's answer
  (``TASK_WRITERS``), so the page can say when a choice of tools cannot
  work, as the spec would;
- the atlases BrainGlobe describes
  (:func:`langslice.core.atlas.details.descriptions`), the planes and the
  default atlas;
- the flag each setting the form fills is spelled with, from the settings
  table (:mod:`langslice.job.settings.schema`), so the command it writes is
  the one ``init`` reads;
- the image models: every provider name a job spec accepts (the names the
  job spec checks ``--nonlinear-provider`` against), each with the models
  LangSlice's providers list for it;
- a job's own settings, to start the form from: a job prepared in ABBA, or
  any job folder with a ``job.json``.

Nothing here is model-facing: the labels are for people, and the message the
page makes is the person's to send.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from langslice.core.atlas.details import descriptions
from langslice.core.connections import CANONICAL_PROVIDERS
from langslice.core.jsonable import JsonObject
from langslice.core.stack.json_reads import (
    integer_at,
    number_at,
    object_at,
    optional_at,
    text_at,
    text_map_at,
    texts_at,
)
from langslice.doors.credentials import image_model_connected
from langslice.hosts.gui.settings import LABELS as CONNECTION_LABELS
from langslice.job.layout import JobLayout, held_image_folder, read_job_file
from langslice.job.migrations import without_removed_settings
from langslice.job.settings import labels as tool_labels
from langslice.job.settings import tool_table
from langslice.job.settings.schema import (
    ALL_TASKS,
    DEFAULT_ATLAS,
    DEFAULT_TASKS,
    PLANES,
    flag_spellings,
    setting,
)
from langslice.job.settings.spec import JobSpec, NonlinearSpec
from langslice.providers import connections as image_registry
from langslice.providers.openai_api import OPENAI_IMAGE_MODELS

#: The atlas species the list is grouped by, in order: (group, words in a name).
SPECIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mouse", ("mouse",)),
    ("Rat", ("_rat_",)),
)


#: The properties the form's "what you already know" can keep on every
#: section (:data:`langslice.job.settings.frozen.PROPERTIES`), each with its
#: task.
KEEP: dict[str, str] = {"angles": "position", "orientation": "transform"}

#: The image models LangSlice lists for each provider that has some.
IMAGE_MODELS: dict[str, tuple[str, ...]] = {
    "openai-oauth": image_registry.OPENAI_OAUTH_IMAGE_MODELS,
    "openai-api": OPENAI_IMAGE_MODELS,
    "gemini-api": image_registry.GEMINI_IMAGE_MODELS,
}

#: The image-model provider ``none`` as the form names it.
NO_IMAGE_MODEL = "None (no image model)"

#: ``host.kind`` of a job prepared in ABBA.
ABBA_HOST = "abba"


def species(name: str) -> str:
    """The group *name* is listed under: Mouse, Rat or Other."""
    for group, words in SPECIES:
        if any(word in name for word in words):
            return group
    return "Other"


def task_sets() -> list[list[str]]:
    """Every non-empty choice of tasks, each in :data:`ALL_TASKS` order."""
    return [list(chosen) for size in range(1, len(ALL_TASKS) + 1)
            for chosen in combinations(ALL_TASKS, size)]


def tools() -> list[dict[str, Any]]:
    """The form's tools, in table order: the catalog's (scripting verbs left
    out), each with its label, its options' names and labels, whether it
    calls the image model, and ``runs_in``: the task choices (comma-joined)
    whose runs have it (``ToolDef.when``)."""
    specs = [JobSpec(image_folder=".", tasks=tasks) for tasks in task_sets()]
    rows = []
    for entry in tool_table.catalog():
        held = tool_table.TOOLS.get(entry["name"])
        if held is not None and held.scripting:
            continue
        options = [f"{entry['name']}{tool_table.OPTION_SEPARATOR}{option}"
                   for option in entry["options"]]
        rows.append({
            "name": entry["name"], "group": entry["group"],
            "label": tool_labels.label(entry["name"]),
            "disableable": entry["disableable"], "needs": entry["needs"],
            "image_model": held is not None and held.image_model,
            "options": [{"name": name, "label": tool_labels.label(name)} for name in options],
            "runs_in": [",".join(spec.tasks) for spec in specs
                        if held is None or held.when(spec)],
        })
    return rows


def frozen_off(*properties: str) -> list[str]:
    """The tools and options a freeze of *properties* on every section
    switches off in a run of every task
    (:func:`langslice.job.settings.tool_table.off_names`)."""
    return JobSpec(image_folder=".", tasks=list(ALL_TASKS),
                   frozen={"*": list(properties)}).switched_off


@dataclass(frozen=True)
class ImageModelChoice:
    """One image-model provider the form offers, as the page reads it."""

    #: The provider's name, as ``--nonlinear-provider`` takes it.
    provider: str
    #: The name people see.
    label: str
    #: The job's own default provider.
    default: bool
    #: The models LangSlice lists for it (``--nonlinear-image-model``).
    models: list[str]
    #: The model it uses when none is named (None: none LangSlice names).
    default_model: str | None
    #: This computer has its key or sign-in (always, for no image model).
    connected: bool


def _calls_no_model(provider: str) -> bool:
    """Tell whether *provider* runs the warp with no image model at all."""
    return not NonlinearSpec(provider=provider).uses_image_model


def _provider_order(provider: str) -> tuple[bool, bool]:
    """Sort a provider in the form: the job's default first, no model last."""
    default = setting("nonlinear.provider").default
    return provider != default, _calls_no_model(provider)


def image_models() -> list[ImageModelChoice]:
    """List the image-model providers the form offers.

    Every provider name a job spec accepts by name (``custom`` is the
    library's alone), the job's default first and the one with no image
    model last, each with the models LangSlice lists for it. Whether a
    provider is connected is the offline presence check of
    :func:`image_model_connected`.

    Returns
    -------
    list of ImageModelChoice
        The providers, in the form's order.
    """
    default = setting("nonlinear.provider").default
    rows: list[ImageModelChoice] = []
    for name in sorted(CANONICAL_PROVIDERS, key=_provider_order):
        no_model = _calls_no_model(name)
        label = CONNECTION_LABELS.get(name, (name, ""))[0]
        if no_model:
            label = NO_IMAGE_MODEL
        rows.append(ImageModelChoice(
            provider=name,
            label=label,
            default=name == default,
            models=list(IMAGE_MODELS.get(name, ())),
            default_model=image_registry.default_image_model(name),
            connected=no_model or image_model_connected(name),
        ))
    return rows


def payload() -> dict[str, Any]:
    """Everything the generator page needs, as plain data (``GET generator``)."""
    atlases = [{"name": name, "about": about, "species": species(name)}
               for name, about in sorted(descriptions().items())]
    if not any(atlas["name"] == DEFAULT_ATLAS for atlas in atlases):
        atlases.insert(0, {"name": DEFAULT_ATLAS, "about": "", "species": species(DEFAULT_ATLAS)})
    groups = tool_table.GROUPS
    writers = tool_table.TASK_WRITERS
    return {
        "tasks": [{"name": task, "label": tool_labels.TASK_LABELS[task],
                   "about": tool_labels.TASK_ABOUT[task], "default": task in DEFAULT_TASKS}
                  for task in ALL_TASKS],
        "groups": [{"name": group, "label": tool_labels.GROUP_LABELS[group][0],
                    "task": tool_labels.GROUP_LABELS[group][1]} for group in groups],
        "tools": tools(),
        "task_writers": {task: list(names) for task, names in writers.items()},
        "keep": KEEP,
        "frozen_off": {prop: frozen_off(prop) for prop in KEEP},
        "atlases": atlases,
        "species": [group for group, _ in SPECIES] + ["Other"],
        "default_atlas": DEFAULT_ATLAS,
        "planes": list(PLANES),
        "flags": flag_spellings(),
        "image_models": [asdict(choice) for choice in image_models()],
    }



@dataclass(frozen=True)
class AbbaBlock:
    """What the form shows of a job prepared in ABBA (its ``host.abba``)."""

    #: ABBA's name for the atlas.
    atlas: str
    #: Each snapshot file's ABBA slice name.
    sections: dict[str, str]
    #: The snapshots ABBA had already registered.
    registered: list[str]
    #: The registered snapshots with a warp of their own.
    existing_warp: list[str]
    #: The snapshots' micrometres per pixel.
    pixel_size_um: float
    #: ABBA's section thickness and interval (0: ABBA had none).
    thickness_um: int
    interval_um: int


@dataclass(frozen=True)
class Prefill:
    """The form as a job's ``job.json`` holds it (``GET generator?job=n``).

    The text fields are as the form's inputs hold them: a number as text,
    blank for none.
    """

    #: The image folder, absolute.
    folder: str
    atlas: str
    plane: str
    pixel_size: str
    thickness: str
    interval: str
    tasks: list[str]
    #: The job's ``disabled``.
    off: list[str]
    #: Each :data:`KEEP` property: frozen on every section.
    keep: dict[str, bool]
    #: ``file: note`` lines.
    damaged: str
    notes: str
    facts: list[str]
    #: The per-section freezes (``"*"`` is :attr:`keep`'s).
    frozen: dict[str, list[str]]
    #: The job was prepared in ABBA: what ABBA fixed.
    abba: AbbaBlock | None


@dataclass(frozen=True)
class _SavedJob:
    """What :func:`prefill` reads of a ``job.json``."""

    spec: JobSpec
    notes: str
    abba: AbbaBlock | None


def prefill(folder: Path) -> Prefill | None:
    """Read the form a job folder's ``job.json`` holds.

    Only ``job.json`` is read: its ``spec`` (through
    :meth:`JobSpec.from_dict`, the image folder resolved from the job
    folder), its ``notes`` and, for a job prepared in ABBA, its
    ``host.abba``. A job prepared in ABBA has ABBA's thickness and
    interval where its spec has none.

    Parameters
    ----------
    folder
        The job folder.

    Returns
    -------
    Prefill or None
        The form; None when the folder has no ``job.json`` this version
        reads.
    """
    try:
        saved = _read_saved(folder)
    except (OSError, ValueError, TypeError):
        return None
    if saved is None:
        return None

    spec, abba = saved.spec, saved.abba
    thickness = spec.position.thickness_um
    interval = spec.position.interval_um
    if abba is not None:
        thickness = thickness or abba.thickness_um
        interval = interval or abba.interval_um
    pixel_size = ""
    if isinstance(spec.inputs.get("pixel_size_um"), int | float):
        pixel_size = _text(number_at(spec.inputs, "pixel_size_um"))
    damaged = optional_at(spec.inputs, "damaged", text_map_at) or {}
    everywhere = spec.frozen.get("*", [])
    frozen = {name: list(props) for name, props in spec.frozen.items()}
    frozen.pop("*", None)
    return Prefill(
        folder=spec.image_folder,
        atlas=spec.atlas,
        plane=spec.plane,
        pixel_size=pixel_size,
        thickness=_text(thickness),
        interval=_text(interval),
        tasks=list(spec.tasks),
        off=list(spec.disabled),
        keep={prop: prop in everywhere for prop in KEEP},
        damaged="".join(f"{name}: {note}\n" for name, note in damaged.items()),
        notes=saved.notes,
        facts=list(spec.facts),
        frozen=frozen,
        abba=abba,
    )


def _read_saved(folder: Path) -> _SavedJob | None:
    """Read a job folder's ``job.json``: None without one, ``ValueError``
    for one this version cannot read.
    """
    record = read_job_file(JobLayout(folder))
    if record is None or "spec" not in record:
        return None

    settings = dict(object_at(record, "spec"))
    images = held_image_folder(folder, record)
    settings["image_folder"] = str(images or "")
    kept, _removed = without_removed_settings(settings)
    host = optional_at(record, "host", object_at)
    abba = None
    if host is not None:
        abba = abba_block(host)
    return _SavedJob(
        spec=JobSpec.from_dict(dict(kept)),
        notes=optional_at(record, "notes", text_at) or "",
        abba=abba,
    )


def abba_block(host: JsonObject) -> AbbaBlock | None:
    """Read what the form shows of a job prepared in ABBA.

    Parameters
    ----------
    host
        The ``host`` record of a ``job.json``.

    Returns
    -------
    AbbaBlock or None
        Its ``abba`` block when ``host.kind`` is ``abba``; None for any
        other job. ``ValueError`` for a block without the fields
        ``abba.prepare`` writes.
    """
    if host.get("kind") != ABBA_HOST:
        return None
    block = object_at(host, "abba")
    return AbbaBlock(
        atlas=text_at(block, "atlas"),
        sections=text_map_at(block, "sections"),
        registered=list(texts_at(block, "registered")),
        existing_warp=list(texts_at(block, "existing_warp")),
        pixel_size_um=number_at(block, "pixel_size_um"),
        thickness_um=integer_at(block, "thickness_um"),
        interval_um=integer_at(block, "interval_um"),
    )


def _text(value: float | None) -> str:
    """Write a number as the form's text field holds it (none, 0: blank)."""
    if not value:
        return ""
    return f"{value:g}"
