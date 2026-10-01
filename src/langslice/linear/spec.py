"""The job spec: one filled form per ``langslice linear`` run.

A host (CLI, ABBA plugin, engine service) fills a :class:`JobSpec` and hands it
to :func:`langslice.linear.run`. Every checkbox a user sees maps to a field
here; nothing else is user-facing. ``tasks`` switches whole capabilities on and
off — a task that is OFF takes its answer from ``inputs`` instead of from the
agent, and its tools are not built at all.

See ``docs/linear_design.md`` for the design this mirrors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Every task, in the order they are listed everywhere else.
DEFAULT_TASKS: tuple[str, ...] = ("reorder", "position", "transform")
ALL_TASKS: tuple[str, ...] = (*DEFAULT_TASKS, "nonlinear")

PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")

#: How large the pictures the agent is shown are drawn. "low" is the
#: calibrated size (the atlas's own resolution); the larger settings draw the
#: same pictures bigger (:data:`langslice.linear.render.IMAGE_RESOLUTION_SCALE`).
#: Nothing a fit computes or a transform stores depends on it.
IMAGE_RESOLUTIONS: tuple[str, ...] = ("low", "medium", "high")

#: Most sections one transform-tool call may take (``TransformSpec.max_parallel``).
MAX_PARALLEL_TRANSFORMS = 4

#: ``NonlinearSpec.engine`` values: a fixed deformable-fit engine, or the
#: agent's choice per call.
DEFORMABLE_ENGINES: tuple[str, ...] = ("ants", "elastix", "either")


@dataclass
class ReorderSpec:
    """Knobs of the ``reorder`` task (the order only).

    ``flip`` and ``hemisphere_cue`` moved to :class:`TransformSpec` on
    2026-09-29: a mirror is the sign of the in-plane affine, so it belongs to
    the ``transform`` task. The two fields remain here only as aliases for
    hosts that still fill them: :class:`JobSpec` folds them into
    ``transform`` (flip off in either place switches it off; a cue given
    only here is used) and then mirrors the effective values back for
    readers, and :meth:`JobSpec.to_dict` writes them under ``transform``
    only.
    """

    #: Deprecated alias of ``TransformSpec.flip``.
    flip: bool = True
    #: Deprecated alias of ``TransformSpec.hemisphere_cue``.
    hemisphere_cue: str = ""


@dataclass
class PositionSpec:
    """Knobs of the ``position`` task; the cutting protocol rides along."""

    thickness_um: int = 50
    interval_um: int = 200
    #: Sections must sit exactly one interval apart (gated at submit).
    strict_interval: bool = False
    #: Build the ``run_deepslice`` tool (coronal mouse/rat only).
    deepslice: bool = False
    #: Build the ``search_position`` tool (the oblique fitter).
    bayesian: bool = False
    #: Look-before-you-write gates (2026-09-09, for the cheaper models):
    #: ``set_positions`` is refused for a section not compared at any
    #: position since its last write, and ``submit`` until
    #: ``view_stack`` has run after the last write. Data-only refusals that
    #: name what is missing; Astra passes them without noticing, Luna wrote
    #: 36 uncompared positions in one call without them.
    gated: bool = False
    #: Astra's own run-8 method, written into the job statement for the
    #: cheaper models (2026-09-09): hypothesise order and every position from
    #: the opening images, confirm each section at that position four per
    #: call, write, re-check the doubtful, review, submit. Coaching text, so
    #: off for Astra and off by default.
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
    #: Offer automatic silhouette / optional Elastix fitting.
    automatic: bool = True
    #: The agent may set the stack-wide cutting angles.
    angles: bool = False
    #: ``fit_affine`` may use the Elastix intensity affine.
    elastix: bool = False
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
    """Image-model border correction after a supplied linear placement."""

    provider: str = "openai-oauth"
    image_model: str | None = None
    #: The deformable-fit engine (`fit_deformable`): "ants" or "elastix" fixes
    #: it for the run; "either" (default) lets the agent choose per call.
    engine: str = "either"
    #: The user's own notes for this task, shown to the agent with the task.
    notes: str = ""

    def __post_init__(self) -> None:
        if self.engine not in DEFORMABLE_ENGINES:
            raise ValueError(
                f"nonlinear.engine must be one of {DEFORMABLE_ENGINES}; got {self.engine!r}"
            )


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
    #: Display-side preprocessing for everything the agent looks at:
    #: "auto" runs :func:`langslice.image_prep.adaptive_preprocess`, "none"
    #: shows the raw section. Never written back to the user's files.
    preprocess: str = "auto"
    #: A host's channel-blend settings (``preprocess.preview``'s
    #: ``PreprocessingSettings``: auto, or custom weights and CLAHE) for
    #: snapshots exported one page per channel. Set, every section's DEFAULT
    #: appearance is :func:`langslice.image_prep.host_preprocess` over its raw
    #: pages; the pages themselves stay readable as channels. None: the file
    #: is shown through ``preprocess`` as above.
    host_preprocessing: dict[str, Any] | None = None
    #: Build the ``preprocess`` tool: the agent may set channel weights, CLAHE,
    #: N4 and denoising for what it views and what a fit reads. Off, both stay
    #: the default appearance above.
    agent_preprocessing: bool = False
    #: Size of every picture the agent sees: "low" (the calibrated size),
    #: "medium" or "high". Display only: fits, working frames and stored
    #: transforms are unchanged, and the image model's inputs are untouched.
    image_resolution: str = "low"
    #: Build ``mark_damaged``: the agent may flag damaged sections. Off, only
    #: the host's ``inputs["damaged"]`` flags exist. Either way the agent can
    #: never clear a flag the host set.
    agent_damage: bool = True
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    reorder: ReorderSpec = field(default_factory=ReorderSpec)
    position: PositionSpec = field(default_factory=PositionSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    nonlinear: NonlinearSpec = field(default_factory=NonlinearSpec)
    #: Free-form user facts, one line each, passed to the agent verbatim.
    facts: list[str] = field(default_factory=list)
    #: Host-supplied answers for tasks that are OFF, plus the optional
    #: calibration override:
    #: ``{"positions": {filename: mm}, "order": [filename, ...],
    #: "angles": {"pitch": deg, "yaw": deg}, "pixel_size_um": float,
    #: "transforms": {filename: transform_dict},
    #: "damaged": {filename: note}, "locked": [filename, ...]}``.
    #: A supplied transform may be mirrored (negative determinant, as a
    #: host's own alignment carries a flip); it is kept as supplied.
    #: ``damaged`` flags cannot be cleared by the agent. ``locked`` sections
    #: were aligned in-plane by the user: the agent cannot change their flip,
    #: rotation or transform (a ``"host"`` identity transform unless
    #: ``transforms`` supplies one), but their positions still move.
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Resume from the folder checkpoint when one exists.
    resume: bool = True
    #: After submit, ask the agent (same context) what tools it missed and
    #: record the answer on the state. One extra model call.
    debrief: bool = True
    #: Compatibility field: only the established working-set policy is supported.
    image_retention: str = "legacy"
    #: Stop after an observed request exceeds this input count (cached tokens
    #: included), with one grace call to submit. None disables the safeguard.
    #: This is checked after usage arrives, not a preflight context guarantee.
    max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS
    #: Hard stop on the usage-window share one run may spend, measured from
    #: the window's reading on the first call; ignored when the provider
    #: reports no quota.
    max_quota_percent: int = DEFAULT_MAX_QUOTA_PERCENT

    def __post_init__(self) -> None:
        if self.image_retention != "legacy":
            raise ValueError("image_retention must be legacy; completion retirement was removed")
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
        if self.plane not in PLANES:
            raise ValueError(f"Unsupported plane {self.plane!r}; expected one of {PLANES}")
        unknown = [task for task in self.tasks if task not in ALL_TASKS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}; expected any of {list(ALL_TASKS)}")
        # Flip moved from reorder to transform (2026-09-29); a host that still
        # sets the old fields keeps working, and old readers see the values
        # that apply.
        self.transform.flip = bool(self.transform.flip and self.reorder.flip)
        self.transform.hemisphere_cue = str(
            self.transform.hemisphere_cue or self.reorder.hemisphere_cue or ""
        )
        self.reorder.flip = self.transform.flip
        self.reorder.hemisphere_cue = self.transform.hemisphere_cue

    # --- views -----------------------------------------------------------

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
        # The flip lives under ``transform``; the reorder aliases are not
        # written, so a reloaded spec has one source for it.
        data["reorder"].pop("flip", None)
        data["reorder"].pop("hemisphere_cue", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        nested = {
            "reorder": ReorderSpec,
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
