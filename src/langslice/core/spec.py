"""The job spec: one filled form per ``langslice linear`` run.

A host (CLI, ABBA connector, engine service) fills a :class:`JobSpec` and hands it
to :func:`langslice.agent.engine.run`. Every checkbox a user sees maps to a field
here; nothing else is user-facing. ``tasks`` switches whole capabilities on and
off — a task that is OFF takes its answer from ``inputs`` instead of from the
agent, and its tools are not built at all. ``disabled`` takes single tools and
tool options off on top of what the tasks give.

See ``docs/registration.md`` for the design this mirrors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from langslice.core import frozen as freeze
from langslice.core import toolset

#: Every task, in the order they are listed everywhere else.
DEFAULT_TASKS: tuple[str, ...] = ("position", "transform")
ALL_TASKS: tuple[str, ...] = (*DEFAULT_TASKS, "nonlinear")

PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")

#: How large the pictures the agent is shown are drawn: a long edge for the
#: opening images and one for every later picture per level
#: (:data:`langslice.core.sizes.PICTURE_EDGES`); "auto" lets the agent pass
#: ``resolution`` per call. Nothing a fit computes or a transform stores
#: depends on it, and the image model's inputs do not change.
IMAGE_RESOLUTIONS: tuple[str, ...] = ("low", "medium", "high", "auto")

#: Most sections one transform-tool call may take (``TransformSpec.max_parallel``).
MAX_PARALLEL_TRANSFORMS = 4

#: The keys :attr:`JobSpec.inputs` takes; any other is refused (a misspelled
#: key would otherwise drop the supplied answer without a word).
INPUT_KEYS: tuple[str, ...] = (
    "order", "positions", "angles", "orientation", "transforms", "damaged",
    "pixel_size_um", "channel_names",
)

#: The two parts of a supplied cutting angle (``inputs.angles``).
ANGLE_KEYS: tuple[str, ...] = ("pitch", "yaw")


def _degrees(value: Any, where: str) -> float:
    if isinstance(value, (bool, dict, list, tuple)) or value is None:
        raise ValueError(f"{where} must be a number of degrees; got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a number of degrees; got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite; got {value!r}")
    return number


def _angle_pair(value: dict[str, Any], where: str) -> tuple[float, float]:
    strange = sorted(str(key) for key in value if key not in ANGLE_KEYS)
    if strange:
        raise ValueError(f"{where} takes only pitch and yaw; got {strange}")
    pitch, yaw = (_degrees(value.get(key, 0.0), f"{where}.{key}") for key in ANGLE_KEYS)
    return pitch, yaw


def supplied_angles(
    value: Any,
) -> tuple[tuple[float, float] | None, dict[str, tuple[float, float]]]:
    """``inputs.angles`` read: ``(stack, per_section)``.

    Two forms. Stack-wide, ``{"pitch": deg, "yaw": deg}`` (the CLI's
    ``--pitch``/``--yaw``): every section gets that plane, returned as
    ``((pitch, yaw), {})``. Per section, ``{filename: {"pitch": deg,
    "yaw": deg}, ...}`` (a registration made elsewhere, each section's own
    plane): returned as ``(None, {filename: (pitch, yaw)})``; a section it
    does not name keeps the flat plane. A missing part is 0. The form is
    read from the values (numbers or objects); a mapping that mixes the two
    is refused, as are other keys in an angle, and non-finite or
    non-numeric degrees (``ValueError``). Empty or None: ``(None, {})``.
    """
    if value is None or value == {}:
        return None, {}
    if not isinstance(value, dict):
        raise ValueError("inputs.angles must be {pitch, yaw} for the whole stack, or "
                         "{filename: {pitch, yaw}} per section")
    nested = [key for key, item in value.items() if isinstance(item, dict)]
    if not nested:
        return _angle_pair(value, "inputs.angles"), {}
    if len(nested) != len(value):
        flat = sorted(str(key) for key in value if key not in nested)
        raise ValueError(
            f"inputs.angles mixes the stack-wide form ({flat}) with per-section "
            "entries; give either {pitch, yaw} for the whole stack or "
            "{filename: {pitch, yaw}} for each section")
    return None, {str(name): _angle_pair(item, f"inputs.angles[{name!r}]")
                  for name, item in value.items()}


def _micrometres(value: Any, where: str) -> float:
    try:
        um = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be micrometres per pixel; got {value!r}") from None
    if not math.isfinite(um) or um <= 0:
        raise ValueError(f"{where} must be a positive, finite µm per pixel; got {value!r}")
    return um


def supplied_pixel_sizes(value: Any) -> tuple[float | None, dict[str, float]]:
    """``inputs.pixel_size_um`` read: ``(stack, per_section)``.

    Two forms. One number (the CLI's ``--pixel-size-um``): every file has that
    µm per pixel, returned as ``(um, {})``. Per file, ``{filename: um, ...}``
    (files of one stack scanned or exported at different sizes): returned as
    ``(None, {filename: um})``; a file it does not name falls back to its own
    tags. Non-numeric, non-finite or non-positive sizes are refused
    (``ValueError``). None or empty: ``(None, {})``.
    """
    if value is None or value == {}:
        return None, {}
    if isinstance(value, dict):
        return None, {str(name): _micrometres(um, f"inputs.pixel_size_um[{name!r}]")
                      for name, um in value.items()}
    return _micrometres(value, "inputs.pixel_size_um"), {}


@dataclass
class PositionSpec:
    """Knobs of the ``position`` task; the cutting protocol rides along."""

    #: None means not supplied; the agent infers what the anatomy supports.
    thickness_um: int | None = None
    interval_um: int | None = None
    #: Sections must sit exactly one interval apart (gated at submit).
    strict_interval: bool = False
    #: Optional positioning coaching: compare candidate planes and regions,
    #: write supported placements, revisit uncertainty and review the stack.
    #: Guidance only, off by default.
    playbook: bool = False
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("thickness_um", "interval_um"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value <= 0):
                raise ValueError(f"{name} must be a positive integer in micrometres or null")
        if self.strict_interval and self.interval_um is None:
            raise ValueError("strict_interval requires a supplied interval_um")


@dataclass
class TransformSpec:
    """Knobs of the ``transform`` task."""

    #: User text describing what marks a hemisphere (a notch, an injection...),
    #: shown where the agent may mirror sections (``interactive_transform``'s
    #: ``flip``, an option of the tool that ``disabled`` can take off).
    hemisphere_cue: str = ""
    #: The agent may set the stack-wide cutting angles.
    angles: bool = False
    #: Most sections one transform-tool call (``elastix_affine``,
    #: ``interactive_transform``) may take, 1..4. Both tools take at most four
    #: (the default); below 4 both refuse a call naming more sections.
    max_parallel: int = MAX_PARALLEL_TRANSFORMS
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""

    def __post_init__(self) -> None:
        value = self.max_parallel
        if (not isinstance(value, int) or isinstance(value, bool)
                or not 1 <= value <= MAX_PARALLEL_TRANSFORMS):
            raise ValueError(
                f"transform.max_parallel must be an integer from 1 to "
                f"{MAX_PARALLEL_TRANSFORMS}; got {value!r}"
            )


@dataclass
class NonlinearSpec:
    """Knobs of the ``nonlinear`` task: a deformation per section on top of
    its linear placement, fitted to the stain or, for a section the agent
    chooses to trace, to the borders the image model draws.

    ``provider`` is the image model's access method (``providers/registry``);
    ``"none"`` runs the task without an image model: no ``trace_borders``,
    no traced section images. No section ever needs a trace. ``"custom"``
    (:data:`langslice.core.provider_names.CUSTOM_PROVIDER`): an image model
    the caller hands the library itself (``langslice.open_job(...,
    image_model=...)``); a door that is handed none offers no
    ``trace_borders``.
    """

    provider: str = "openai-oauth"
    image_model: str | None = None
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""
    #: The host requires a deformation on every section: ``submit`` is refused
    #: until each section carries one at its current placement. Off, a
    #: deformation is the agent's choice, section by section.
    require_deformation: bool = False
    #: The base prompt ``trace_borders`` sends the image model for this job,
    #: in place of LangSlice's own for the provider
    #: (:func:`langslice.core.nonlinear.prompts.border_correction_tool_prompt`);
    #: ``{plane}`` in it becomes the section plane. It must describe the
    #: provider's attachment order (the OpenAI providers: Image 1 the clean
    #: photograph, Image 2 the placed borders; the others the other way
    #: round). The agent's per-section edit starts from it, and its traces
    #: are recorded as an untested profile (``custom``). None: LangSlice's own.
    image_prompt: str | None = None

    def __post_init__(self) -> None:
        from langslice.core.provider_names import (
            CANONICAL_PROVIDERS,
            CUSTOM_PROVIDER,
            canonical_provider,
        )

        accepted = (*CANONICAL_PROVIDERS, CUSTOM_PROVIDER)
        if canonical_provider(str(self.provider or "")) not in accepted:
            raise ValueError(
                f"nonlinear.provider must be one of {accepted}; got {self.provider!r}"
            )
        if self.image_prompt is not None:
            if not isinstance(self.image_prompt, str) or not self.image_prompt.strip():
                raise ValueError("nonlinear.image_prompt must be non-empty text or null")
            if not self.uses_image_model:
                raise ValueError("nonlinear.image_prompt needs an image model; the "
                                 "nonlinear provider is 'none'")
            self.image_prompt = self.image_prompt.strip()

    @property
    def uses_image_model(self) -> bool:
        """Whether the image model is part of this run (provider is not ``none``)."""
        from langslice.core.provider_names import canonical_provider

        return canonical_provider(str(self.provider or "")) != "none"


@dataclass
class JobSpec:
    """One whole ``langslice linear`` run, host-agnostic."""

    image_folder: str
    atlas: str = "allen_mouse_25um"
    plane: str = "coronal"
    model: str | None = None
    #: Reasoning effort for models that expose one (low|medium|high|xhigh|
    #: max). None leaves the provider's own default alone.
    reasoning: str | None = None
    out: str | None = None
    #: Where the job folder goes (``--job-dir``). None: next to the images,
    #: ``<images>/langslice`` (:func:`langslice.job.layout.locate_job_folder`).
    job_dir: str | None = None
    #: Display-side preprocessing for everything the agent looks at:
    #: "auto" runs :func:`langslice.core.image_prep.adaptive_preprocess`, "none"
    #: shows the raw section. Never written back to the user's files.
    preprocess: str = "auto"
    #: A host's channel-blend settings (``preprocess.preview``'s
    #: ``PreprocessingSettings``: auto, or custom weights and CLAHE) for
    #: snapshots exported one page per channel. Set, every section's DEFAULT
    #: appearance is :func:`langslice.core.image_prep.host_preprocess` over its raw
    #: pages; the pages themselves stay readable as channels. None: the file
    #: is shown through ``preprocess`` as above.
    host_preprocessing: dict[str, Any] | None = None
    #: Size of every picture the agent sees: "low", "medium", "high" or
    #: "auto" (:data:`IMAGE_RESOLUTIONS`). Display only: fits, working frames
    #: and stored transforms are unchanged, and the image model's inputs are
    #: untouched.
    image_resolution: str = "low"
    #: The change tools always return their picture: their ``view`` argument
    #: is not offered, so the agent cannot switch the picture off.
    force_view: bool = False
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    #: Tools and tool options switched off on top of what the tasks give:
    #: ``"zoom"`` (a tool, whole) or ``"look:positioning"`` (``tool:option``).
    #: The valid names are :func:`langslice.core.toolset.valid_names`; a name
    #: that is none, a tool the run cannot do without (``submit``) or a list
    #: that cannot work is refused when the spec is made
    #: (:func:`langslice.core.toolset.refusal`). A host's ``inputs["damaged"]``
    #: notes are shown whatever is disabled.
    disabled: list[str] = field(default_factory=list)
    #: What the agent may not change, per section: ``{"<filename>":
    #: ["orientation", "affine"], "*": ["angles"]}`` (``"*"``: every section;
    #: the properties are :data:`langslice.core.frozen.PROPERTIES`). A frozen
    #: property keeps the value ``inputs`` supplies for it, or its "none"
    #: (:mod:`langslice.core.frozen`); the ops refuse a write that would
    #: change it (``FROZEN``), a tool that writes only properties frozen on
    #: every section is not offered (:attr:`switched_off`) and a task whose
    #: properties are all frozen on every section is off (:meth:`has`).
    frozen: dict[str, list[str]] = field(default_factory=dict)
    position: PositionSpec = field(default_factory=PositionSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    nonlinear: NonlinearSpec = field(default_factory=NonlinearSpec)
    #: Free-form user facts, one line each, passed to the agent verbatim.
    facts: list[str] = field(default_factory=list)
    #: Host-supplied answers for tasks that are OFF, plus the optional
    #: calibration override:
    #: ``{"positions": {filename: mm}, "order": [filename, ...],
    #: "angles": {"pitch": deg, "yaw": deg}, "pixel_size_um": float | {filename: float},
    #: "orientation": {filename: {"flip": bool, "rotation_deg": 0|90|180|270}},
    #: "transforms": {filename: transform_dict},
    #: "damaged": {filename: note}}``.
    #: A supplied transform may be mirrored (negative determinant, as a
    #: host's own alignment carries a flip); it is kept as supplied. A
    #: supplied ``orientation`` is the section's flip and quarter turn (a
    #: supplied transform describes the section after it), kept as supplied.
    #: ``angles`` is the whole stack's plane, or per section
    #: ``{filename: {"pitch": deg, "yaw": deg}}`` (a registration made
    #: elsewhere keeps each section's own plane; :func:`supplied_angles`).
    #: ``channel_names`` (one name per page of a host's multi-page snapshot)
    #: names its channels. Any other key is refused (:data:`INPUT_KEYS`).
    #: ``damaged`` supplies notes, not region exclusions. What is supplied
    #: is a starting point: the agent may change it unless ``frozen`` says
    #: it may not.
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Resume from the folder checkpoint when one exists.
    resume: bool = True
    #: After submit, ask the agent (same context) what tools it missed and
    #: record the answer on the state. One extra model call.
    debrief: bool = True
    #: Tool plugins the job loads, each ``module:factory``
    #: (:mod:`langslice.core.plugins`, ``docs/developers/plugins.md``): their tools join
    #: the toolbox, and are named in ``disabled`` like the native ones. A
    #: plugin that cannot be loaded, or whose tools clash with another's, is
    #: refused when the spec is made. Code runs on load: only the library and
    #: the CLI build specs with plugins (the MCP server and the ABBA worker
    #: refuse them).
    plugins: list[str] = field(default_factory=list)
    #: Options a plugin's factory is called with, by plugin reference (a JSON
    #: object each; the generic MCP adapter's server and allow-list are one).
    #: A plugin without an entry is called without arguments.
    plugin_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_resolution not in IMAGE_RESOLUTIONS:
            raise ValueError(
                f"Unsupported image_resolution {self.image_resolution!r}; "
                f"expected one of {IMAGE_RESOLUTIONS}"
            )
        if self.host_preprocessing is not None and not isinstance(self.host_preprocessing, dict):
            raise ValueError("host_preprocessing must be a settings object or null")
        if self.plane not in PLANES:
            raise ValueError(f"Unsupported plane {self.plane!r}; expected one of {PLANES}")
        unknown = [task for task in self.tasks if task not in ALL_TASKS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}; expected any of {list(ALL_TASKS)}")
        if self.inputs is not None:
            if not isinstance(self.inputs, dict):
                raise ValueError(f"inputs must be a mapping of {list(INPUT_KEYS)}")
            strange = sorted(str(key) for key in self.inputs if key not in INPUT_KEYS)
            if strange:
                raise ValueError(
                    f"Unknown inputs key(s) {strange}; inputs takes only {list(INPUT_KEYS)}"
                )
            supplied_angles(self.inputs.get("angles"))
            supplied_pixel_sizes(self.inputs.get("pixel_size_um"))
        if not isinstance(self.disabled, (list, tuple)) or not all(
                isinstance(name, str) for name in self.disabled):
            raise ValueError("disabled must be a list of tool and option names")
        self.disabled = list(dict.fromkeys(name.strip() for name in self.disabled))
        if not isinstance(self.plugins, (list, tuple)) or not all(
                isinstance(ref, str) and ref.strip() for ref in self.plugins):
            raise ValueError("plugins must be a list of module:factory references")
        self.plugins = list(dict.fromkeys(ref.strip() for ref in self.plugins))
        if not isinstance(self.plugin_config, dict) or not all(
                isinstance(ref, str) and isinstance(item, dict)
                for ref, item in self.plugin_config.items()):
            raise ValueError("plugin_config must map a plugin reference to an options object")
        stray = sorted(set(self.plugin_config) - set(self.plugins))
        if stray:
            raise ValueError(f"plugin_config names {stray}, which are no plugin of this job")
        if self.plugins:
            toolset.tools_for(self)  # loads them: refused here when they cannot be
        self.frozen = freeze.clean(self.frozen)
        supplied = (self.inputs or {}).get("positions") or {}
        unsupplied = sorted(name for name, held in self.frozen.items()
                            if name != freeze.EVERY and "position" in held
                            and name not in supplied)
        if unsupplied:
            raise ValueError(
                f"frozen: the position of {unsupplied} is frozen, and a position has no "
                "default: supply it in inputs.positions")
        reason = toolset.refusal(self)
        if reason is not None:
            raise ValueError(reason)

    # --- views -----------------------------------------------------------

    def has(self, task: str) -> bool:
        """Whether *task* is on for this run: in ``tasks``, and not every one
        of its properties frozen on every section (``frozen["*"]``): a task
        with nothing left to write is off."""
        return task in self.tasks and not (
            set(freeze.TASK_PROPERTIES[task]) <= freeze.everywhere(self.frozen))

    @property
    def active_tasks(self) -> list[str]:
        """The tasks that are on (:meth:`has`), in :data:`ALL_TASKS` order."""
        return [task for task in ALL_TASKS if self.has(task)]

    @property
    def frozen_everywhere(self) -> frozenset[str]:
        """The properties no section may change: ``frozen["*"]`` and the
        properties of every task that is off (the host supplied them). The
        cutting angles stay free in a run whose Linear task may set them
        (``transform.angles``)."""
        return freeze.everywhere(self.frozen) | freeze.derived(
            self.active_tasks, angles_free=bool(self.has("transform") and self.transform.angles))

    def frozen_for(self, section_id: str) -> frozenset[str]:
        """The properties frozen on one section: its own, ``frozen["*"]``
        and what the tasks that are off supply."""
        return freeze.for_section(self.frozen, section_id) | self.frozen_everywhere

    def off(self, name: str) -> bool:
        """Whether the tool or ``tool:option`` *name* is off: disabled, or
        taken off by what is frozen (a tool's options are off with it)."""
        return toolset.is_off(self.switched_off, name)

    @property
    def switched_off(self) -> list[str]:
        """Every name that is off for this run: ``disabled`` and, after it,
        the tools and options that can change nothing because everything
        they write is frozen on every section
        (:func:`langslice.core.toolset.frozen_off`). What every door and
        every text renders by."""
        return toolset.off_names(self)

    @property
    def thickness_mm(self) -> float | None:
        value = self.position.thickness_um
        return value / 1000.0 if value is not None else None

    @property
    def interval_mm(self) -> float | None:
        value = self.position.interval_um
        return value / 1000.0 if value is not None else None

    # --- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("job_dir") is None:  # the default stays out of saved specs
            data.pop("job_dir", None)
        if not data["nonlinear"].get("require_deformation"):  # likewise
            data["nonlinear"].pop("require_deformation", None)
        if data["nonlinear"].get("image_prompt") is None:  # likewise
            data["nonlinear"].pop("image_prompt", None)
        if not data.get("force_view"):  # likewise
            data.pop("force_view", None)
        if not data.get("disabled"):  # likewise
            data.pop("disabled", None)
        if not data.get("frozen"):  # likewise
            data.pop("frozen", None)
        if not data.get("plugins"):  # likewise
            data.pop("plugins", None)
        if not data.get("plugin_config"):  # likewise
            data.pop("plugin_config", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Rebuild from :meth:`to_dict` output; ``ValueError`` naming any
        unknown key (a nested one as ``section.key``)."""
        nested = {
            "position": PositionSpec,
            "transform": TransformSpec,
            "nonlinear": NonlinearSpec,
        }
        unknown = [name for name in data if name not in cls.__dataclass_fields__]
        unknown += [f"{name}.{key}" for name, sub in nested.items()
                    for key in (data.get(name) or {}) if key not in sub.__dataclass_fields__]
        if unknown:
            raise ValueError(f"Unknown job setting(s) {sorted(unknown)}")
        kwargs: dict[str, Any] = {
            name: nested[name](**(value or {})) if name in nested else value
            for name, value in data.items()}
        return cls(**kwargs)
