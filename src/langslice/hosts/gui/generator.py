"""The prompt generator's data: what the form offers, read from the code.

The prompt generator tab (``static/generator.js``) is a form that runs in the
page: it turns a person's choices into one message to paste into their agent
app, a short request and the job's ``langslice-job FOLDER init`` command,
which the main agent's ``register-brain`` skill runs. This module gives the
page what the choices are, so nothing in the form is written by hand twice:

- the tools and their options from :func:`langslice.core.toolset.catalog`
  (the scripting verbs left out: a script's, never the agent's to lose), each
  with its people's label (:mod:`langslice.core.tool_labels`), so a renamed or
  removed tool can never linger in the form;
- the tasks, :data:`~langslice.core.spec.DEFAULT_TASKS` and the tools that
  write each task's answer (:data:`~langslice.core.toolset.TASK_WRITERS`), so
  the page can say when a choice of tools cannot work, as the spec would;
- the atlases BrainGlobe describes (:func:`langslice.core.atlas.details.descriptions`),
  the planes, and :class:`~langslice.core.spec.JobSpec`'s default atlas.

Nothing here is model-facing: the labels are for people, and the message the
page makes is the person's to send.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from langslice.core import tool_labels, toolset
from langslice.core.atlas.details import descriptions
from langslice.core.spec import ALL_TASKS, DEFAULT_TASKS, PLANES, JobSpec

#: The atlas species the list is grouped by, in order: (group, words in a name).
SPECIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mouse", ("mouse",)),
    ("Rat", ("_rat_",)),
)


#: The properties the form's "what you already know" can keep on every
#: section (:data:`langslice.core.frozen.PROPERTIES`), each with its task.
KEEP: dict[str, str] = {"angles": "position", "orientation": "transform"}


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
    out), each with its label, its options' names and labels, and ``runs_in``:
    the task choices (comma-joined) whose runs have it (``ToolDef.when``)."""
    specs = [JobSpec(image_folder=".", tasks=tasks) for tasks in task_sets()]
    rows = []
    for entry in toolset.catalog():
        held = toolset.TOOLS.get(entry["name"])
        if held is not None and held.scripting:
            continue
        options = [f"{entry['name']}{toolset.OPTION_SEPARATOR}{option}"
                   for option in entry["options"]]
        rows.append({
            "name": entry["name"], "group": entry["group"],
            "label": tool_labels.label(entry["name"]),
            "disableable": entry["disableable"], "needs": entry["needs"],
            "options": [{"name": name, "label": tool_labels.label(name)} for name in options],
            "runs_in": [",".join(spec.tasks) for spec in specs
                        if held is None or held.when(spec)],
        })
    return rows


def frozen_off(*properties: str) -> list[str]:
    """The tools and options a freeze of *properties* on every section
    switches off in a run of every task (:func:`langslice.core.toolset.off_names`)."""
    return JobSpec(image_folder=".", tasks=list(ALL_TASKS),
                   frozen={"*": list(properties)}).switched_off


def payload() -> dict[str, Any]:
    """Everything the generator page needs, as plain data (``GET generator``)."""
    defaults = JobSpec(image_folder=".")
    atlases = [{"name": name, "about": about, "species": species(name)}
               for name, about in sorted(descriptions().items())]
    if not any(atlas["name"] == defaults.atlas for atlas in atlases):
        atlases.insert(0, {"name": defaults.atlas, "about": "", "species": species(defaults.atlas)})
    return {
        "tasks": [{"name": task, "label": tool_labels.TASK_LABELS[task],
                   "about": tool_labels.TASK_ABOUT[task], "default": task in DEFAULT_TASKS}
                  for task in ALL_TASKS],
        "groups": [{"name": group, "label": tool_labels.GROUP_LABELS[group][0],
                    "task": tool_labels.GROUP_LABELS[group][1]} for group in toolset.GROUPS],
        "tools": tools(),
        "task_writers": {task: list(names) for task, names in toolset.TASK_WRITERS.items()},
        "keep": KEEP,
        "frozen_off": {prop: frozen_off(prop) for prop in KEEP},
        "atlases": atlases,
        "species": [group for group, _ in SPECIES] + ["Other"],
        "default_atlas": defaults.atlas,
        "planes": list(PLANES),
    }
