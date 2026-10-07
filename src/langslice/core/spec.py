"""The job spec: one filled form per ``langslice linear`` run.

A host (CLI, ABBA connector, engine service) fills a :class:`JobSpec` and hands it
to :func:`langslice.agent.engine.run`. Every checkbox a user sees maps to a field
here; nothing else is user-facing. ``tasks`` switches whole capabilities on and
off — a task that is OFF takes its answer from ``inputs`` instead of from the
agent, and its tools are not built at all.

See ``docs/linear_design.md`` for the design this mirrors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

#: Every task, in the order they are listed everywhere else.
DEFAULT_TASKS: tuple[str, ...] = ("reorder", "position", "transform")
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
    "order", "positions", "angles", "orientation", "transforms", "damaged", "locked",
    "keep_warp", "nonlinear_skip", "pixel_size_um", "channel_names",
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


#: How much a job folder keeps (:attr:`JobSpec.output_level`): "full" every
#: file a job writes; "lean" the results only (the state, ``job.json``,
#: ``registration.json``, each section's maps, the deformation records and
#: image-model traces they are made from, the exports). Lean writes no
#: pictures (``views/``, ``views.jsonl``), no undo history on disk
#: (``history/``; undo still works within the process that wrote the step),
#: no ``logs/`` and no reference card (``AGENTS.md``/``CLAUDE.md``).
OUTPUT_LEVELS: tuple[str, ...] = ("full", "lean")


#: ``NonlinearSpec.engine`` values: a fixed deformable-fit engine, or the
#: agent's choice per call.
DEFORMABLE_ENGINES: tuple[str, ...] = ("ants", "elastix", "either")


@dataclass
class PositionSpec:
    """Knobs of the ``position`` task; the cutting protocol rides along."""

    thickness_um: int = 50
    interval_um: int = 200
    #: Sections must sit exactly one interval apart (gated at submit).
    strict_interval: bool = False
    #: Build the ``search_position`` tool (the oblique fitter).
    bayesian: bool = False
    #: Look-before-you-write gates: ``set_positions`` is refused for a section
    #: not compared at any position since its last write, and ``submit`` until
    #: ``view_stack`` has run after the last write. The refusals name what is
    #: missing.
    gated: bool = False
    #: A suggested method written into the job statement: hypothesise order
    #: and every position from the opening images, confirm each section at that
    #: position four per call, write, re-check the doubtful, review, submit.
    #: Coaching text, off by default.
    playbook: bool = False
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""


@dataclass
class TransformSpec:
    """Knobs of the ``transform`` task."""

    #: The agent may mirror sections left-right (``orient_slices``). A mirror
    #: is part of the in-plane alignment: the sign of the affine.
    flip: bool = True
    #: User text describing what marks a hemisphere (a notch, an injection...).
    hemisphere_cue: str = ""
    #: Offer direct visual adjustment.
    interactive: bool = True
    #: Offer the automatic affine fit (``fit_affine``: Elastix by default,
    #: silhouette as an option).
    automatic: bool = True
    #: The agent may set the stack-wide cutting angles.
    angles: bool = False
    #: Most sections one transform-tool call (``fit_affine``,
    #: ``adjust_transforms``) may take, 1..4. At 4, the default, the tools
    #: keep their own limits (``adjust_transforms`` four, ``fit_affine`` any
    #: number); below 4 both refuse a call naming more sections.
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
    #: The deformable-fit engine (`fit_deformable`): "ants" or "elastix" fixes
    #: it for the run; "either" (default) lets the agent choose per call.
    engine: str = "either"
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""
    #: The host requires a deformation on every section: ``submit`` refuses
    #: ``left_linear`` (sections left without one, each with its reason).
    require_deformation: bool = False

    def __post_init__(self) -> None:
        from langslice.core.provider_names import (
            CANONICAL_PROVIDERS,
            CUSTOM_PROVIDER,
            canonical_provider,
        )

        if self.engine not in DEFORMABLE_ENGINES:
            raise ValueError(
                f"nonlinear.engine must be one of {DEFORMABLE_ENGINES}; got {self.engine!r}"
            )
        accepted = (*CANONICAL_PROVIDERS, CUSTOM_PROVIDER)
        if canonical_provider(str(self.provider or "")) not in accepted:
            raise ValueError(
                f"nonlinear.provider must be one of {accepted}; got {self.provider!r}"
            )

    @property
    def uses_image_model(self) -> bool:
        """Whether the image model is part of this run (provider is not ``none``)."""
        from langslice.core.provider_names import canonical_provider

        return canonical_provider(str(self.provider or "")) != "none"


#: Optional per-request context safeguard, disabled unless a host sets it.
#: Cumulative input measures repeated processing, not context-window size.
DEFAULT_MAX_INPUT_TOKENS: int | None = None
#: Share of the provider's usage window one run may spend, when the provider
#: reports one (the OAuth lane's x-codex headers). Cached tokens are ~0.13x
#: there, so this, not the raw input count, is the cost.
DEFAULT_MAX_QUOTA_PERCENT = 25


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
    #: What the job folder keeps: "full" (default) or "lean", the results
    #: only (:data:`OUTPUT_LEVELS`). Left out of saved specs when "full".
    output_level: str = "full"
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
    #: Build the ``preprocess`` tool: the agent may set channel weights, CLAHE,
    #: N4 and denoising for what it views and what a fit reads. Off, both stay
    #: the default appearance above.
    agent_preprocessing: bool = False
    #: Size of every picture the agent sees: "low", "medium", "high" or
    #: "auto" (:data:`IMAGE_RESOLUTIONS`). Display only: fits, working frames
    #: and stored transforms are unchanged, and the image model's inputs are
    #: untouched.
    image_resolution: str = "low"
    #: Build ``mark_damaged``: the agent may flag damaged sections. Off, only
    #: the host's ``inputs["damaged"]`` flags exist. Either way the agent can
    #: never clear a flag the host set.
    agent_damage: bool = True
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    position: PositionSpec = field(default_factory=PositionSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    nonlinear: NonlinearSpec = field(default_factory=NonlinearSpec)
    #: Free-form user facts, one line each, passed to the agent verbatim.
    facts: list[str] = field(default_factory=list)
    #: Host-supplied answers for tasks that are OFF, plus the optional
    #: calibration override:
    #: ``{"positions": {filename: mm}, "order": [filename, ...],
    #: "angles": {"pitch": deg, "yaw": deg}, "pixel_size_um": float,
    #: "orientation": {filename: {"flip": bool, "rotation_deg": 0|90|180|270}},
    #: "transforms": {filename: transform_dict},
    #: "damaged": {filename: note}, "locked": [filename, ...]}``.
    #: A supplied transform may be mirrored (negative determinant, as a
    #: host's own alignment carries a flip); it is kept as supplied. A
    #: supplied ``orientation`` is the section's flip and quarter turn (the
    #: ``orient_slices`` data; a supplied transform describes the section
    #: after it), kept as supplied.
    #: ``angles`` is the whole stack's plane, or per section
    #: ``{filename: {"pitch": deg, "yaw": deg}}`` (a registration made
    #: elsewhere keeps each section's own plane; :func:`supplied_angles`).
    #: ``channel_names`` (one name per page of a host's multi-page snapshot)
    #: names its channels. Any other key is refused (:data:`INPUT_KEYS`).
    #: ``damaged`` flags cannot be cleared by the agent. ``locked`` sections
    #: were aligned in-plane by the user: the agent cannot change their flip,
    #: rotation or transform (a ``"host"`` identity transform unless
    #: ``transforms`` supplies one), but their positions still move.
    #: ``keep_warp`` sections (a list of filenames) carry the user's own
    #: deformation in the host: ``fit_deformable`` refuses them
    #: (``KEEPS_HOST_WARP``). ``nonlinear_skip`` sections (a list of
    #: filenames) the user left out of Nonlinear (no linear registration and
    #: not to be aligned first): ``fit_deformable`` and ``trace_borders``
    #: refuse them (``NONLINEAR_SKIPPED``).
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Resume from the folder checkpoint when one exists.
    resume: bool = True
    #: After submit, ask the agent (same context) what tools it missed and
    #: record the answer on the state. One extra model call.
    debrief: bool = True
    #: Stop after an observed request exceeds this input count (cached tokens
    #: included), with one grace call to submit. None disables the safeguard.
    #: This is checked after usage arrives, not a preflight context guarantee.
    max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS
    #: Hard stop on the usage-window share one run may spend, measured from
    #: the window's reading on the first call; ignored when the provider
    #: reports no quota.
    max_quota_percent: int = DEFAULT_MAX_QUOTA_PERCENT

    def __post_init__(self) -> None:
        if self.image_resolution not in IMAGE_RESOLUTIONS:
            raise ValueError(
                f"Unsupported image_resolution {self.image_resolution!r}; "
                f"expected one of {IMAGE_RESOLUTIONS}"
            )
        if not isinstance(self.agent_damage, bool):
            raise ValueError(f"agent_damage must be true or false; got {self.agent_damage!r}")
        if not isinstance(self.agent_preprocessing, bool):
            raise ValueError(
                f"agent_preprocessing must be true or false; got {self.agent_preprocessing!r}"
            )
        if self.host_preprocessing is not None and not isinstance(self.host_preprocessing, dict):
            raise ValueError("host_preprocessing must be a settings object or null")
        if self.output_level not in OUTPUT_LEVELS:
            raise ValueError(
                f"output_level must be one of {OUTPUT_LEVELS}; got {self.output_level!r}"
            )
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

    # --- views -----------------------------------------------------------

    @property
    def lean(self) -> bool:
        """Whether the job folder keeps the results only (``output_level`` "lean")."""
        return self.output_level == "lean"

    def has(self, task: str) -> bool:
        """Whether *task* is switched on for this run."""
        return task in self.tasks

    @property
    def thickness_mm(self) -> float:
        return self.position.thickness_um / 1000.0

    @property
    def interval_mm(self) -> float:
        return self.position.interval_um / 1000.0

    # --- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("job_dir") is None:  # the default stays out of saved specs
            data.pop("job_dir", None)
        if data.get("output_level") == "full":  # likewise
            data.pop("output_level", None)
        if not data["nonlinear"].get("require_deformation"):  # likewise
            data["nonlinear"].pop("require_deformation", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        nested = {
            "position": PositionSpec,
            "transform": TransformSpec,
            "nonlinear": NonlinearSpec,
        }
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in nested:
                sub = nested[name]
                fields = sub.__dataclass_fields__
                kwargs[name] = sub(**{k: v for k, v in (value or {}).items() if k in fields})
            elif name in cls.__dataclass_fields__:
                kwargs[name] = value
        return cls(**kwargs)
