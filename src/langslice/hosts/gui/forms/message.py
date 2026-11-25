"""The prompt generator's message, made from the person's form.

The prompt generator tab (``web/src/panels/prompt/``) holds the form; the page posts
it to ``generator/message`` on every change and shows what comes back
(:func:`evaluate`): the message to paste into an agent app (a short request
and the job's ``langslice job FOLDER init`` command, which the main agent's
``register-brain`` skill runs), the problems that stop it, the warnings, and
how each tool of the form stands in this run. The page computes none of it.

The tool choices are checked twice: first in the people's words, by the
tasks' writers and the tools' needs as :mod:`.generator` lists them with
their labels, then by the job spec itself
(:class:`~langslice.job.settings.spec.JobSpec`), which ``init`` builds from the
same values, so the form can never offer a command the spec refuses. Each
flag is spelled as the spec's settings spell it
(:func:`langslice.doors.commands.flags.flag_names`).

Nothing here writes a file, and nothing is model-facing: the message is the
person's to send.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import Any
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.alias_generators import to_camel

from langslice.hosts.gui.forms import generator
from langslice.job.settings.spec import ALL_TASKS, DEFAULT_ATLAS, JobSpec, NonlinearSpec

ORDERED = "The file names sort in the order the sections were cut."
SAFE_WORD = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")
DOUBLE_QUOTE_EXPANDS = re.compile(r"[\"$`\\!]")
WHOLE_NUMBER = re.compile(r"^[1-9][0-9]*$")
DECIMAL = re.compile(r"^([0-9]+\.?[0-9]*|\.[0-9]+)$")
REQUEST = (
    "Please register my brain sections with LangSlice, following its register-brain "
    "skill: create the job with the init command below, then launch the registration "
    "and check the result."
)
WARP_ALONE = (
    "Warping alone starts from sections already placed and aligned: say in the notes "
    "where that registration is (a QuickNII, VisuAlign, DeepSlice or LangSlice file)."
)
OTHER_SHELL = "The command is quoted for bash; if your shell is another, quote it for yours."
NO_IMAGE_MODEL = "none"


class Form(BaseModel):
    """The prompt generator's form as the page holds it (camelCase keys)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    folder: str = ""
    atlas: str = ""
    plane: str = ""
    thickness: str = ""
    interval: str = ""
    pixel_size: str = ""
    tasks: list[str] = []
    off: list[str] = []
    on: list[str] = []
    ordered: bool = False
    damaged: str = ""
    keep: dict[str, bool] = {}
    notes: str = ""
    image_provider: str = ""
    image_model: str = ""
    abba: dict[str, Any] | None = None
    keep_registrations: bool = True
    frozen: dict[str, Any] = {}
    integrations: list[str] = []


@dataclass(frozen=True)
class ToolStand:
    """How one tool of the form stands in this run, for its switch.

    ``present``: the run has it (its task is chosen, and an image-model tool
    only with an image model); ``held``: a kept property leaves it nothing to
    do; ``on``: its switch is on (an essential tool's always is).
    """

    present: bool
    held: bool
    on: bool


@dataclass(frozen=True)
class Made:
    """What the page shows for a form (``POST generator/message``)."""

    text: str
    problems: list[str]
    warnings: list[str]
    tools: dict[str, ToolStand] = field(default_factory=dict)
    image_model_applies: bool = False


@cache
def _choices() -> dict[str, Any]:
    """The form's choices (:func:`.generator.payload`) without the parts that
    change while the window is open (the connected image models)."""
    data = generator.payload()
    return {
        key: data[key]
        for key in ("tasks", "tools", "task_writers", "keep", "frozen_off", "flags", "integrations")
    }


def _blank(value: object) -> bool:
    return str(value if value is not None else "").strip() == ""


def shell_quote(value: str) -> str:
    """A word for a POSIX shell: as it is when it needs no quoting, else in
    single quotes, or in double quotes when it holds a single quote and
    nothing a double-quoted word expands."""
    if SAFE_WORD.match(value):
        return value
    if "'" not in value:
        return f"'{value}'"
    if not DOUBLE_QUOTE_EXPANDS.search(value):
        return f'"{value}"'
    return "'" + value.replace("'", "'\"'\"'") + "'"


def clean_path(raw: str) -> str:
    """A pasted folder path without the quotes Windows' "Copy as path" adds,
    or a ``file://`` address's prefix."""
    path = raw.strip()
    if len(path) >= 2 and path[0] in "\"'" and path[-1] == path[0]:
        path = path[1:-1].strip()
    if path.startswith("file://"):
        path = unquote(path[len("file://") :])
    return path


def damaged_map(text: str) -> dict[str, str]:
    """The damaged sections the form lists, one ``file: what is wrong`` a line."""
    out: dict[str, str] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        name, _, note = line.partition(":")
        if name.strip():
            out[name.strip()] = note.strip() or "damaged"
    return out


def _join_words(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class _Run:
    """The form read against the choices: which tasks and tools the run has."""

    def __init__(self, form: Form) -> None:
        self.form = form
        self.data = _choices()
        self.tools: list[dict[str, Any]] = self.data["tools"]
        names = [task["name"] for task in self.data["tasks"]]
        self.chosen = [name for name in names if name in form.tasks]
        defaults = [task["name"] for task in self.data["tasks"] if task["default"]]
        self.run_tasks = self.chosen or defaults
        key = ",".join(self.run_tasks)
        no_model = form.image_provider == NO_IMAGE_MODEL
        self.present = {
            tool["name"]
            for tool in self.tools
            if key in tool["runs_in"] and not (no_model and tool["image_model"])
        }
        keep: dict[str, str] = self.data["keep"]
        self.kept = [prop for prop in keep if form.keep.get(prop) and keep[prop] in self.chosen]
        self.kept_off = {
            name for prop in self.kept for name in self.data["frozen_off"].get(prop, [])
        }

    def tool_off(self, tool: dict[str, Any]) -> bool:
        """Whether the form has *tool* off: one that starts off unless ``on``
        names it, any other when ``off`` does."""
        if tool["default_off"]:
            return tool["name"] not in self.form.on
        return tool["name"] in self.form.off

    def starts_off(self) -> list[str]:
        return [tool["name"] for tool in self.tools if tool["default_off"] and self.tool_off(tool)]

    def on_names(self) -> list[str]:
        """The names ``--enabled`` takes: tools of this run that start off and
        the form switches on."""
        return [
            tool["name"]
            for tool in self.tools
            if tool["default_off"]
            and tool["name"] in self.present
            and tool["name"] not in self.kept_off
            and not self.tool_off(tool)
        ]

    def off_names(self) -> list[str]:
        """The names ``--disabled`` takes: tools of this run switched off,
        those a kept property already switches off left out."""
        return [
            tool["name"]
            for tool in self.tools
            if tool["name"] in self.present
            and tool["name"] not in self.kept_off
            and tool["disableable"]
            and self.tool_off(tool)
            and not tool["default_off"]
        ]

    def label_of(self, name: str) -> str:
        return next((tool["label"] for tool in self.tools if tool["name"] == name), name)

    def image_model_applies(self) -> bool:
        """Whether the image-model choice applies: the run warps, and a tool
        that calls the image model is not switched off."""
        if "nonlinear" not in self.run_tasks:
            return False
        key = ",".join(self.run_tasks)
        return any(
            tool["image_model"]
            and key in tool["runs_in"]
            and not (tool["disableable"] and self.tool_off(tool))
            and tool["name"] not in self.kept_off
            for tool in self.tools
        )

    def image_model_settings(self) -> list[tuple[str, str]]:
        provider, model = self.form.image_provider.strip(), self.form.image_model.strip()
        if not provider or not self.image_model_applies():
            return []
        out = [("nonlinear.provider", provider)]
        if model and provider != NO_IMAGE_MODEL:
            out.append(("nonlinear.image_model", model))
        return out

    def frozen(self) -> dict[str, Any] | None:
        """The job's ``--frozen``, or None for none: the kept properties on
        every section, and an ABBA job's per-section freezes while kept;
        unticked, ``{}`` frees them."""
        saved = self.form.frozen
        freed = bool(self.form.abba) and not self.form.keep_registrations and bool(saved)
        sections = {} if freed else saved
        if self.kept:
            return {"*": self.kept, **sections}
        return {} if freed else None

    def problems(self) -> list[str]:
        form, out = self.form, []
        if _blank(clean_path(form.folder)):
            out.append("Enter the folder that holds the section images.")
        if not _blank(form.thickness) and not WHOLE_NUMBER.match(form.thickness.strip()):
            out.append("Section thickness is a whole number of micrometres (µm).")
        if not _blank(form.interval) and not WHOLE_NUMBER.match(form.interval.strip()):
            out.append("Section interval is a whole number of micrometres (µm).")
        size = form.pixel_size.strip()
        if not form.abba and size and not (DECIMAL.match(size) and float(size) > 0):
            out.append("Pixel size is a number of micrometres per pixel, above 0.")
        off = {*self.off_names(), *self.starts_off()}
        for task in self.run_tasks:
            writers: list[str] = self.data["task_writers"].get(task, [])
            if writers and not any(name in self.present and name not in off for name in writers):
                label = next(t["label"] for t in self.data["tasks"] if t["name"] == task)
                tools = ", ".join(f'"{self.label_of(name)}"' for name in writers)
                out.append(f'"{label}" needs one of these tools on: {tools}.')
        needing: dict[tuple[str, ...], list[str]] = {}
        for tool in self.tools:
            if tool["name"] not in self.present or tool["name"] in off:
                continue
            for group in tool["needs"]:
                if all(name in off for name in group):
                    needing.setdefault(tuple(group), []).append(tool["label"])
        for group, users in needing.items():
            needed = " or ".join(f'"{self.label_of(name)}"' for name in group)
            many = len(users) > 1
            quoted = _join_words([f'"{label}"' for label in users])
            out.append(
                f"{quoted} "
                f"{'need' if many else 'needs'} {needed}: turn it back on, or turn "
                f"{'those' if many else 'that'} off too."
            )
        return out

    def left_out(self) -> list[str]:
        form, fixed, out = self.form, bool(self.form.abba), []
        if not self.chosen:
            out.append("what to do (place, align, warp)")
        if not fixed and _blank(form.atlas):
            out.append("the atlas")
        if not fixed and _blank(form.plane):
            out.append("the cutting plane")
        if _blank(form.thickness):
            out.append("the section thickness")
        if _blank(form.interval):
            out.append("the section interval")
        if not fixed and _blank(form.pixel_size):
            out.append("the pixel size (when the image files do not record it)")
        return out

    def settings(self) -> list[tuple[str, str]]:
        """The command's settings in order, each a setting path and its value."""
        form, fixed, out = self.form, bool(self.form.abba), []
        if self.chosen:
            out.append(("tasks", ",".join(self.chosen)))
        if not fixed and not _blank(form.atlas):
            out.append(("atlas", form.atlas.strip()))
        if not fixed and not _blank(form.plane):
            out.append(("plane", form.plane.strip()))
        if not _blank(form.thickness):
            out.append(("position.thickness_um", form.thickness.strip()))
        if not _blank(form.interval):
            out.append(("position.interval_um", form.interval.strip()))
        if not fixed and not _blank(form.pixel_size):
            out.append(("inputs.pixel_size_um", form.pixel_size.strip()))
        if off := self.off_names():
            out.append(("disabled", ",".join(off)))
        if on := self.on_names():
            out.append(("enabled", ",".join(on)))
        out.extend(self.image_model_settings())
        frozen = self.frozen()
        if frozen is not None:
            out.append(("frozen", json.dumps(frozen, separators=(",", ":"))))
        if damaged := damaged_map(form.damaged):
            out.append(
                ("inputs.damaged", json.dumps(damaged, separators=(",", ":"), ensure_ascii=False))
            )
        if form.ordered:
            out.append(("facts", ORDERED))
        if not _blank(form.notes):
            out.append(("notes", form.notes.strip()))
        return out

    def spec_refusal(self) -> str | None:
        """Why the job spec refuses this run's tool choices; None when it takes them."""
        image = dict(self.image_model_settings())
        nonlinear: dict[str, Any] = {}
        if "nonlinear.provider" in image:
            nonlinear["provider"] = image["nonlinear.provider"]
        if "nonlinear.image_model" in image:
            nonlinear["image_model"] = image["nonlinear.image_model"]
        try:
            JobSpec(
                image_folder=".",
                tasks=self.run_tasks,
                disabled=self.off_names(),
                enabled=self.on_names(),
                frozen={"*": self.kept} if self.kept else {},
                nonlinear=NonlinearSpec(**nonlinear),
            )
        except ValidationError as exc:
            return "; ".join(
                str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()
            )
        return None

    def command(self) -> tuple[str, bool]:
        """The command, quoted for bash, and whether any word needed quotes."""
        flags: dict[str, str] = self.data["flags"]
        words = [[shell_quote(clean_path(self.form.folder)), "init"]]
        words.extend([flags[path], shell_quote(value)] for path, value in self.settings())
        quoted = any(word[:1] in ("'", '"') for line in words for word in line)
        return "langslice job " + " \\\n  ".join(" ".join(line) for line in words), quoted

    def integration_words(self) -> list[str]:
        if self.form.abba:
            return []
        chosen = set(self.form.integrations)
        return [item["words"] for item in self.data["integrations"] if item["name"] in chosen]

    def stands(self) -> dict[str, ToolStand]:
        return {
            tool["name"]: ToolStand(
                present=tool["name"] in self.present,
                held=tool["name"] in self.kept_off,
                on=not tool["disableable"]
                or (tool["name"] not in self.kept_off and not self.tool_off(tool)),
            )
            for tool in self.tools
        }


def start_form(saved: generator.Prefill | None) -> dict[str, Any]:
    """The form the page starts with (``GET generator``'s ``form``).

    Without a job, the default atlas and the default tasks. From a job's
    ``job.json`` (:func:`.generator.prefill`), its settings; a job prepared in
    ABBA keeps its registered sections unless the person unticks it.
    """
    if saved is None:
        defaults = JobSpec(image_folder=".").tasks
        return Form(atlas=DEFAULT_ATLAS, tasks=list(defaults)).model_dump(by_alias=True)
    form = Form(
        folder=saved.folder,
        atlas=saved.atlas,
        plane=saved.plane,
        thickness=saved.thickness,
        interval=saved.interval,
        pixel_size=saved.pixel_size,
        tasks=[task for task in ALL_TASKS if task in saved.tasks],
        off=list(saved.off),
        on=list(saved.on),
        ordered=ORDERED in saved.facts,
        damaged=saved.damaged,
        keep=dict(saved.keep),
        notes=saved.notes,
        abba=asdict(saved.abba) if saved.abba is not None else None,
        frozen=dict(saved.frozen),
    )
    return form.model_dump(by_alias=True)


def evaluate(form: Form) -> Made:
    """Make the message for *form*, with its problems, warnings and tools.

    Parameters
    ----------
    form
        The form as the page holds it.

    Returns
    -------
    Made
        ``text`` is empty while a problem stands.
    """
    run = _Run(form)
    stops = run.problems()
    if not stops and (reason := run.spec_refusal()) is not None:
        stops = [reason]
    notes = [WARP_ALONE] if run.chosen == ["nonlinear"] else []
    tools, applies = run.stands(), run.image_model_applies()
    if stops:
        return Made("", stops, notes, tools, applies)
    command, quoted = run.command()
    lines = [REQUEST]
    if missing := run.left_out():
        lines.append(
            f"The command leaves out {_join_words(missing)}: choose "
            f"{'these' if len(missing) > 1 else 'it'} yourself, or ask me when the images "
            "do not tell you."
        )
    if programs := run.integration_words():
        lines.append(
            f"When the registration is done, prepare the results for {_join_words(programs)}."
        )
    if quoted:
        lines.append(OTHER_SHELL)
    text = " ".join(lines) + "\n\n```bash\n" + command + "\n```\n"
    return Made(text, stops, notes, tools, applies)


def answer(request: object) -> dict[str, Any]:
    """``POST generator/message``: the form in, :func:`evaluate`'s answer out.

    Raises
    ------
    ValueError
        For a body that is no form.
    """
    try:
        form = Form.model_validate(request)
    except ValidationError as exc:
        raise ValueError(str(exc)) from None
    return asdict(evaluate(form))
