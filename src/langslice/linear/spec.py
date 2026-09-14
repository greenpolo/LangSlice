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
ALL_TASKS: tuple[str, ...] = ("reorder", "position", "transform")

PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")


@dataclass
class ReorderSpec:
    """Knobs of the ``reorder`` task."""

    #: The agent may mirror sections across the midline.
    flip: bool = True
    #: User text describing what marks a hemisphere (a notch, an injection...).
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
    #: Build the ``fit_position`` tool (the oblique fitter).
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


@dataclass
class TransformSpec:
    """Knobs of the ``transform`` task."""

    #: Offer direct visual adjustment and landmark tools.
    interactive: bool = True
    #: Offer automatic silhouette / optional Elastix fitting.
    automatic: bool = True
    #: The agent may set the stack-wide cutting angles.
    angles: bool = False
    #: ``fit_affine`` may use the Elastix intensity affine.
    elastix: bool = False


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
    tasks: list[str] = field(default_factory=lambda: list(ALL_TASKS))
    reorder: ReorderSpec = field(default_factory=ReorderSpec)
    position: PositionSpec = field(default_factory=PositionSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    #: Free-form user facts, one line each, passed to the agent verbatim.
    facts: list[str] = field(default_factory=list)
    #: Host-supplied answers for tasks that are OFF, plus the optional
    #: calibration override:
    #: ``{"positions": {filename: mm}, "order": [filename, ...],
    #: "angles": {"pitch": deg, "yaw": deg}, "pixel_size_um": float}``.
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
        if self.plane not in PLANES:
            raise ValueError(f"Unsupported plane {self.plane!r}; expected one of {PLANES}")
        unknown = [task for task in self.tasks if task not in ALL_TASKS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}; expected any of {list(ALL_TASKS)}")

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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        nested = {
            "reorder": ReorderSpec,
            "position": PositionSpec,
            "transform": TransformSpec,
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
