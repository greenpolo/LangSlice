"""The job spec: one filled form per ``langslice linear`` run.

A host (the commands, MCP, the library) fills a :class:`JobSpec` and hands
it to :func:`langslice.doors.agent.engine.run`. Every checkbox a user sees maps
to a field here; nothing else is user-facing. ``tasks`` switches whole
capabilities on and off — a task that is OFF takes its answer from ``inputs``
instead of from the agent, and its tools are not built at all. ``disabled``
takes single tools off on top of what the tasks give.

Every setting is defined once, here, as a field of a frozen Pydantic model:
its type, its default, its validation, a one-line description (the help of its
flag) and, in ``Annotated`` metadata, what the commands add
(:class:`~langslice.job.settings.walk.Cli`). The doors derive from it
(:func:`~langslice.job.settings.walk.leaves`): the same default on every door.
A spec is never changed in place; ``model_copy(update=...)`` makes the one
that differs.

See ``docs/reference/settings.md`` for the design this mirrors.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from langslice.core.atlas.sides import UNSIDED_PLANE, split_side
from langslice.core.space import Plane, as_plane
from langslice.core.workspace import ImageSettings
from langslice.job.settings import frozen as freeze
from langslice.job.settings import tool_table
from langslice.job.settings.tools_on import ToolsOn
from langslice.job.settings.walk import Cli, Holds

Task = Literal["position", "transform", "nonlinear"]
Preprocess = Literal["auto", "none"]
ImageResolution = Literal["low", "medium", "high", "auto"]
Reasoning = Literal["low", "medium", "high", "xhigh", "max"]

ALL_TASKS: tuple[Task, ...] = get_args(Task)

MAX_PARALLEL_TRANSFORMS = 4

DEFAULT_ATLAS = "allen_mouse_25um"

ANGLE_KEYS: tuple[str, ...] = ("pitch", "yaw")

NO_TRANSFORM = "A written affine transform is required: give the section a transform first"


def no_transform_linear_off(on: ToolsOn) -> str:
    """What a fit says of a section with no written transform when Linear is
    off, for the tools *on* has."""
    return (
        "A written affine transform is required, and Linear is off for this job, "
        "so nothing here writes one. Supply the in-plane transforms with the job "
        "(inputs.transforms; --transforms on `langslice job FOLDER init`), or "
        "switch Linear on (task transform)"
        + on.say("elastix_affine", " and run elastix_affine on the section first")
        + "."
    )


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
        raise ValueError(
            "inputs.angles must be {pitch, yaw} for the whole stack, or "
            "{filename: {pitch, yaw}} per section"
        )
    nested = [key for key, item in value.items() if isinstance(item, dict)]
    if not nested:
        return _angle_pair(value, "inputs.angles"), {}
    if len(nested) != len(value):
        flat = sorted(str(key) for key in value if key not in nested)
        raise ValueError(
            f"inputs.angles mixes the stack-wide form ({flat}) with per-section "
            "entries; give either {pitch, yaw} for the whole stack or "
            "{filename: {pitch, yaw}} for each section"
        )
    return None, {
        str(name): _angle_pair(item, f"inputs.angles[{name!r}]") for name, item in value.items()
    }


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
        return None, {
            str(name): _micrometres(um, f"inputs.pixel_size_um[{name!r}]")
            for name, um in value.items()
        }
    return _micrometres(value, "inputs.pixel_size_um"), {}


def supplied_damaged_regions(value: object, plane: str) -> dict[str, list[str]]:
    """Read ``inputs.damaged_regions``: ``{filename: [region entry, ...]}``.

    The host's damage marks per section: atlas region acronyms or ids, each
    optionally one side as the section is shown (``"root:right"``), as
    ``mark_damage`` takes them. The atlas checks the names when the job
    opens; here the shape and the sides.

    Parameters
    ----------
    value
        ``inputs.damaged_regions`` as given (None or empty: none).
    plane
        The job's plane: a side on a sagittal job is refused.

    Returns
    -------
    dict
        Filename -> its entries, as given.

    Raises
    ------
    ValueError
        For another shape, an unknown side, or a side on a sagittal job.
    """
    if value is None or value == {}:
        return {}
    shape = "inputs.damaged_regions must be {filename: [region, ...]}"
    if not isinstance(value, dict):
        raise ValueError(shape)
    marks: dict[str, list[str]] = {}
    entries = cast("dict[object, object]", value)
    for name, regions in entries.items():
        if not isinstance(regions, (list, tuple)):
            raise ValueError(shape)
        listed = [str(entry) for entry in cast("list[object]", regions)]
        sides = [split_side(entry)[1] for entry in listed]
        one_sided = any(side is not None for side in sides)
        if plane == UNSIDED_PLANE and one_sided:
            raise ValueError(
                f"inputs.damaged_regions[{name!r}]: a sagittal section lies "
                "within one hemisphere, so a region cannot be limited to one "
                "side"
            )
        marks[str(name)] = listed
    return marks


class Strict(BaseModel):
    """What every model of the settings shares: no unknown key, no change
    after it is made, and a value of the type it is declared as."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PositionSpec(Strict):
    """Knobs of the ``position`` task; the cutting protocol rides along."""

    thickness_um: Annotated[
        int | None,
        Field(gt=0, description="Known section thickness in microns; omit if unknown"),
        Cli(metavar="UM"),
    ] = None
    interval_um: Annotated[
        int | None,
        Field(
            gt=0,
            description="Known section interval in microns: the spacing of the "
            "sections' initial positions; omit to spread them over the atlas",
        ),
        Cli(metavar="UM"),
    ] = None
    start_mm: Annotated[
        int | float | None,
        Field(
            description="The first section's initial atlas position in mm; omit to "
            "centre the stack in the atlas"
        ),
        Cli(metavar="MM"),
    ] = None
    strict_interval: Annotated[
        bool,
        Field(description="Sections must sit exactly one interval apart"),
        Cli(),
    ] = False
    playbook: Annotated[
        bool,
        Field(
            description="Add optional positioning guidance: compare candidate planes "
            "and regions, write supported placements, revisit uncertainty and "
            "review the stack"
        ),
        Cli(),
    ] = False
    notes: Annotated[
        str,
        Field(
            description="The user's notes for the position task, shown to the agent "
            "with the task (one line each; PositionSpec.notes)"
        ),
        Cli(metavar="TEXT"),
    ] = ""

    @field_validator("start_mm")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("start_mm must be a number of atlas millimetres or null")
        return value

    @model_validator(mode="after")
    def _interval_for_strict(self) -> PositionSpec:
        if self.strict_interval and self.interval_um is None:
            raise ValueError("strict_interval requires a supplied interval_um")
        return self


class TransformSpec(Strict):
    """Knobs of the ``transform`` task."""

    hemisphere_cue: Annotated[
        str,
        Field(description="What marks a hemisphere in these sections (a notch, an injection...)"),
        Cli(metavar="TEXT"),
    ] = ""
    angles: Annotated[
        bool,
        Field(description="Let the agent set the cutting angles"),
        Cli(),
    ] = False
    max_parallel: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_PARALLEL_TRANSFORMS,
            description="How many sections one transform call fits at once (a host's cap)",
        ),
    ] = MAX_PARALLEL_TRANSFORMS
    notes: Annotated[
        str,
        Field(
            description="The user's notes for the transform task, shown to the agent "
            "with the task (one line each; TransformSpec.notes)"
        ),
        Cli(metavar="TEXT"),
    ] = ""


class NonlinearSpec(Strict):
    """Knobs of the ``nonlinear`` task: a deformation per section on top of
    its linear placement, fitted to the stain or, for a section the agent
    chooses to trace, to the borders the image model draws.

    ``provider`` is the image model's access method (``providers/registry``);
    ``"none"``, the default, runs the task without an image model: no
    ``trace_borders``, no traced section images; naming a provider switches
    ``trace_borders`` on (``JobSpec.enabled``). A door that switches
    ``trace_borders`` on without naming one takes the connection saved on
    this computer
    (:func:`langslice.job.settings.defaults.choose_provider`). No section
    ever needs a trace. ``"custom"``
    (:data:`langslice.core.connections.CUSTOM_PROVIDER`): an image model the
    caller hands the library itself (``langslice.open_job(...,
    image_model=...)``); a door that is handed none offers no
    ``trace_borders``.
    """

    provider: Annotated[
        str,
        Field(
            description="Image provider for the nonlinear task's trace_borders "
            "(openai-oauth, openai-api, gemini-api): naming one gives the agent "
            "trace_borders. Not given: with --enabled trace_borders, the "
            "OpenAI sign-in or key saved in the LangSlice window's Settings; "
            "otherwise none (the nonlinear task fits deformations to the stain "
            "alone, with no image model)"
        ),
        Cli(metavar="PROVIDER", deferred=True),
    ] = "none"
    image_model: Annotated[
        str | None,
        Field(
            description="Image model for the nonlinear tool; default is the provider's image model"
        ),
        Cli(metavar="MODEL"),
    ] = None
    notes: Annotated[
        str,
        Field(
            description="The user's notes for the nonlinear task, shown to the agent "
            "with the task (one line each; NonlinearSpec.notes)"
        ),
        Cli(metavar="TEXT"),
    ] = ""
    require_deformation: Annotated[
        bool,
        Field(
            description="Nonlinear task: submit is refused until every section carries "
            "a deformation (NonlinearSpec.require_deformation); otherwise a "
            "deformation is the agent's choice"
        ),
        Cli(group="session"),
    ] = False
    image_prompt: Annotated[
        str | None,
        Field(
            description="A UTF-8 text file holding the base prompt trace_borders sends "
            "the image model, in place of LangSlice's own "
            "(NonlinearSpec.image_prompt); {plane} in it becomes the section "
            "plane. It must describe the provider's attachment order, and its "
            "traces are recorded as an untested profile. The agent's "
            "per-section edit starts from it. Default: LangSlice's prompt for "
            "the provider"
        ),
        Cli(kind="file", metavar="FILE"),
    ] = None

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        from langslice.core.connections import (
            CANONICAL_PROVIDERS,
            CUSTOM_PROVIDER,
            canonical_provider,
        )

        accepted = (*CANONICAL_PROVIDERS, CUSTOM_PROVIDER)
        if canonical_provider(value) not in accepted:
            raise ValueError(f"nonlinear.provider must be one of {accepted}; got {value!r}")
        return value

    @field_validator("image_prompt")
    @classmethod
    def _prompt_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("nonlinear.image_prompt must be non-empty text or null")
        return value.strip()

    @model_validator(mode="after")
    def _prompt_needs_a_model(self) -> NonlinearSpec:
        if self.image_prompt is not None and not self.uses_image_model:
            raise ValueError(
                "nonlinear.image_prompt needs an image model; the nonlinear provider is 'none'"
            )
        return self

    @model_serializer(mode="wrap")
    def _compact(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = cast("dict[str, Any]", handler(self))
        if not data.get("require_deformation"):
            data.pop("require_deformation", None)
        if data.get("image_prompt") is None:
            data.pop("image_prompt", None)
        return data

    @property
    def uses_image_model(self) -> bool:
        """Whether the image model is part of this run (provider is not ``none``)."""
        from langslice.core.connections import canonical_provider

        return canonical_provider(str(self.provider or "")) != "none"


class Inputs(Strict):
    """What a host already knows per section, the keys of ``JobSpec.inputs``.

    The spec keeps the mapping as given; this model says which keys exist,
    what each is and how the commands take it.
    """

    order: Annotated[
        Any,
        Field(description="Host-supplied order: a JSON file path or inline JSON list of filenames"),
        Cli(kind="json", metavar="JSON"),
    ] = None
    positions: Annotated[
        Any,
        Field(
            description="Host-supplied positions: a JSON file path or inline JSON mapping "
            "filename -> mm"
        ),
        Cli(kind="json", metavar="JSON"),
    ] = None
    angles: Annotated[
        Any,
        Field(
            description="Host-given cutting angles: {pitch, yaw} degrees for the whole stack "
            "(--pitch, --yaw), or {filename: {pitch, yaw}} per section (--section-angles)"
        ),
    ] = None
    orientation: Annotated[
        Any,
        Field(
            description="Host-supplied orientation: JSON file or inline mapping of "
            'filenames to {"flip": true|false, "rotation_deg": 0|90|180|270}'
        ),
        Cli(kind="json", metavar="JSON"),
    ] = None
    transforms: Annotated[
        Any,
        Field(
            description="Host-supplied calibrated linear transforms: JSON file or "
            "inline mapping of filenames to saved transform records"
        ),
        Cli(kind="json", metavar="JSON"),
    ] = None
    damaged: Annotated[
        Any,
        Field(
            description="Notes on damaged tissue: JSON file or inline mapping of "
            "filenames to notes; notes alone do not exclude atlas regions from "
            "fits"
        ),
        Cli(kind="json", metavar="JSON"),
    ] = None
    damaged_regions: Annotated[
        Any,
        Field(
            description="Damage marks: JSON file or inline mapping of filenames to "
            'atlas region entries ("CTX", "root:right": a side of the section as '
            "it is shown), which every fit leaves out, as mark_damage takes them"
        ),
        Cli(kind="json", metavar="JSON"),
    ] = None
    pixel_size_um: Annotated[
        Any,
        Field(
            description="Micrometres per pixel of the section images: one number for "
            "every file, or a JSON file or inline mapping of filenames to "
            "micrometres per pixel. Default (and for a file the mapping does not "
            "name): read from each file's TIFF/OME tags"
        ),
        Cli(kind="json", metavar="UM|JSON"),
    ] = None
    channel_names: Annotated[
        Any,
        Field(description="The names of the image files' channels, in order (a host's)"),
    ] = None


INPUT_KEYS: tuple[str, ...] = tuple(Inputs.model_fields)

_TOOL_LISTS = (
    "Tools the agent does not get, on top of what the tasks give "
    "(repeatable), for example zoom. Valid names: " + ", ".join(tool_table.valid_names())
)
_START_OFF = (
    "Tools that start off and the agent gets in this job (repeatable): "
    + ", ".join(tool_table.default_off_names())
    + ". trace_borders uses the saved OpenAI sign-in or key unless "
    "--nonlinear-provider names another image model"
)


class JobSpec(Strict):
    """One whole ``langslice linear`` run, host-agnostic."""

    image_folder: Annotated[
        str, Field(description="The folder of section images (the commands' positional argument)")
    ]
    tasks: Annotated[
        list[Task],
        Field(
            description="Comma-separated subset of position,transform,nonlinear: "
            "position places and orders the sections, transform aligns them "
            "in-plane, nonlinear deforms each linearly aligned section "
            "(fireants_syn, and trace_borders with an image model). A task left "
            "out takes its answer from the flags that supply it (--positions, "
            "--transforms ...). Default: all three, except a task the supplied "
            "answers already settle: position when --positions names every "
            "section, transform when --transforms does, and with --registration "
            "only nonlinear (the imported linear registration kept as it is)"
        ),
        Cli(kind="csv", deferred=True),
    ] = Field(default_factory=lambda: list(ALL_TASKS))
    atlas: Annotated[str, Field(description="BrainGlobe atlas name"), Cli()] = DEFAULT_ATLAS
    plane: Annotated[
        Plane,
        Field(
            description="Slicing plane (normal axis). Position is interpreted along this axis "
            "(AP for coronal, ML for sagittal, DV for horizontal)."
        ),
        Cli(),
    ] = "coronal"
    preprocess: Annotated[
        Preprocess,
        Field(
            description="How each section's preprocessed image is made: auto blends the "
            "channels with local contrast enhancement, none takes the file as "
            "read, in gray. It is the image the agent sees first, the one the "
            "image model is shown, and the one every fit reads unless the agent "
            "names another section image. The image files are never modified"
        ),
        Cli(),
    ] = "auto"
    image_resolution: Annotated[
        ImageResolution,
        Field(description="Size of the pictures the agent is shown; auto lets it choose per call"),
        Cli(),
    ] = "low"
    disabled: Annotated[
        list[str],
        Field(description=_TOOL_LISTS),
        Cli(kind="repeat_csv", metavar="NAME[,NAME...]"),
    ] = Field(default_factory=list)
    enabled: Annotated[
        list[str],
        Field(description=_START_OFF),
        Cli(kind="repeat_csv", metavar="NAME[,NAME...]"),
    ] = Field(default_factory=list)
    facts: Annotated[
        list[str],
        Field(description="A fact about this stack, passed to the agent verbatim (repeatable)"),
        Cli(kind="repeat", metavar="TEXT"),
    ] = Field(default_factory=list)
    inputs: Annotated[
        dict[str, Any],
        Field(description="What the host already knows per section (the keys of Inputs)"),
        Holds(Inputs),
    ] = Field(default_factory=dict)
    frozen: Annotated[
        dict[str, list[str]],
        Field(
            description="What the agent may not change: JSON file or inline mapping of a "
            'filename (or "*" for every section) to a list of properties, any of '
            "position, angles, orientation, affine, deformation, damage; a frozen "
            "property keeps the value the supplied inputs give it"
        ),
        Cli(kind="json", metavar="JSON", deferred=True),
    ] = Field(default_factory=dict)
    out: Annotated[
        str | None,
        Field(description="Results JSON path. Default: <job folder>/exports/linear_results.json"),
        Cli(),
    ] = None
    job_dir: Annotated[
        str | None,
        Field(
            description="Put the job folder (checkpoint, undo history, pictures, "
            "results) here instead of <image_folder>/langslice. A folder holding "
            "the job of another image folder is refused"
        ),
        Cli(metavar="PATH"),
    ] = None
    host_preprocessing: Annotated[
        dict[str, Any] | None,
        Field(description="A host's own preprocessing settings (the library's)"),
    ] = None
    position: PositionSpec = Field(default_factory=PositionSpec)
    transform: TransformSpec = Field(default_factory=TransformSpec)
    nonlinear: NonlinearSpec = Field(default_factory=NonlinearSpec)
    plugins: Annotated[
        list[str],
        Field(
            description="Add the tools of a tool plugin to the agent's toolbox "
            "(repeatable): a Python module and the function in it that returns "
            "the tools. Importing it runs its code; the job records which plugin "
            "ran (docs/developers/plugins.md)"
        ),
        Cli(
            group="plugins",
            kind="repeat_csv",
            metavar="MODULE:FACTORY",
        ),
    ] = Field(default_factory=list)
    plugin_config: Annotated[
        dict[str, Any],
        Field(
            description="Options for a plugin's factory (repeatable): the plugin's "
            "reference, '=', then a JSON object or the path of a JSON file"
        ),
        Cli(
            group="plugins",
            kind="pairs",
            metavar="MODULE:FACTORY=JSON",
        ),
    ] = Field(default_factory=dict)
    model: Annotated[
        str | None,
        Field(description="Model name for the agent session"),
        Cli(group="session"),
    ] = None
    reasoning: Annotated[
        Reasoning | None,
        Field(
            description="Reasoning effort for models that expose one. Default: the provider's own"
        ),
        Cli(group="session"),
    ] = None
    debrief: Annotated[
        bool,
        Field(description="Skip the post-submit debrief question (what tools the agent missed)"),
        Cli(group="session"),
    ] = True
    force_view: Annotated[
        bool,
        Field(
            description="The change and fitting tools always return their picture: "
            "their view argument is not offered (JobSpec.force_view)"
        ),
        Cli(group="session"),
    ] = False
    resume: Annotated[
        bool,
        Field(description="Ignore any existing checkpoint and start over"),
        Cli(group="resume"),
    ] = True

    @field_validator("inputs", mode="before")
    @classmethod
    def _inputs_checked(cls, value: Any) -> dict[str, Any]:
        try:
            held = dict(value or {})
        except (TypeError, ValueError):
            raise ValueError(f"inputs must be a mapping of {list(INPUT_KEYS)}") from None
        strange = sorted(str(key) for key in held if key not in INPUT_KEYS)
        if strange:
            raise ValueError(
                f"Unknown inputs key(s) {strange}; inputs takes only {list(INPUT_KEYS)}"
            )
        supplied_angles(held.get("angles"))
        supplied_pixel_sizes(held.get("pixel_size_um"))
        return held

    @field_validator("disabled", "enabled", mode="after")
    @classmethod
    def _tool_names(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(name.strip() for name in value))

    @field_validator("plugins", mode="after")
    @classmethod
    def _plugin_refs(cls, value: list[str]) -> list[str]:
        if not all(ref.strip() for ref in value):
            raise ValueError("plugins must be a list of module:factory references")
        return list(dict.fromkeys(ref.strip() for ref in value))

    @field_validator("plugin_config", mode="after")
    @classmethod
    def _plugin_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not all(isinstance(item, dict) for item in value.values()):
            raise ValueError("plugin_config must map a plugin reference to an options object")
        return value

    @field_validator("frozen", mode="before")
    @classmethod
    def _frozen_clean(cls, value: Any) -> dict[str, list[str]]:
        return freeze.clean(value)

    @model_validator(mode="after")
    def _settled(self) -> JobSpec:
        supplied_damaged_regions(self.inputs.get("damaged_regions"), self.plane)
        plugins = list(self.plugins)
        enabled = list(self.enabled)
        named = [ref for ref in plugins if ref in tool_table.ENGINES]
        if named:
            plugins = [ref for ref in plugins if ref not in named]
            enabled = list(
                dict.fromkeys(
                    [
                        *enabled,
                        *(
                            tool.name
                            for tool in tool_table.TOOLS.values()
                            if tool.default_off
                            and tool.engine in named
                            and tool.name not in self.disabled
                        ),
                    ]
                )
            )
        if (
            self.nonlinear.uses_image_model
            and "trace_borders" not in self.disabled
            and "trace_borders" not in enabled
        ):
            enabled.append("trace_borders")
        reason = tool_table.check_enabled(enabled)
        if reason is not None:
            raise ValueError(reason)
        both = [name for name in enabled if name in self.disabled]
        if both:
            raise ValueError(f"{both} are in both enabled and disabled")
        known = {*plugins, *tool_table.ENGINES}
        stray = sorted(set(self.plugin_config) - known)
        if stray:
            raise ValueError(
                f"plugin_config names {stray}, which are no plugin of this "
                "job and no engine of LangSlice "
                f"({', '.join(tool_table.ENGINES)})"
            )
        settled = self.model_copy(update={"plugins": plugins, "enabled": enabled})
        if plugins:
            tool_table.tools_for(settled)
        supplied = settled.inputs.get("positions") or {}
        unsupplied = sorted(
            name
            for name, held in settled.frozen.items()
            if name != freeze.EVERY and "position" in held and name not in supplied
        )
        if unsupplied:
            raise ValueError(
                f"frozen: the position of {unsupplied} is frozen, and a position has no "
                "default: supply it in inputs.positions"
            )
        reason = tool_table.refusal(settled)
        if reason is not None:
            raise ValueError(reason)
        return settled

    @model_serializer(mode="wrap")
    def _compact(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = cast("dict[str, Any]", handler(self))
        if data.get("job_dir") is None:
            data.pop("job_dir", None)
        for key in ("force_view", "disabled", "enabled", "frozen", "plugins", "plugin_config"):
            if not data.get(key):
                data.pop(key, None)
        return data

    def has(self, task: str) -> bool:
        """Whether *task* is on for this run: in ``tasks``, and not every one
        of its properties frozen on every section (``frozen["*"]``): a task
        with nothing left to write is off."""
        return task in self.tasks and not (
            set(freeze.TASK_PROPERTIES[task]) <= freeze.everywhere(self.frozen)
        )

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
            self.active_tasks, angles_free=bool(self.has("transform") and self.transform.angles)
        )

    def frozen_for(self, section_id: str) -> frozenset[str]:
        """The properties frozen on one section: its own, ``frozen["*"]``
        and what the tasks that are off supply."""
        return freeze.for_section(self.frozen, section_id) | self.frozen_everywhere

    def off(self, name: str) -> bool:
        """Whether the tool *name* is off: disabled, or taken off by what is
        frozen."""
        return tool_table.is_off(self.switched_off, name)

    @property
    def tools_on(self) -> ToolsOn:
        """The tools that are off for this run, to word texts by
        (:attr:`switched_off`)."""
        return ToolsOn(frozenset(self.switched_off))

    @property
    def switched_off(self) -> list[str]:
        """Every name that is off for this run: ``disabled`` and, after it,
        the tools that can change nothing because everything
        they write is frozen on every section
        (:func:`langslice.job.settings.tool_table.off_names`). What every door
        and every text renders by."""
        return tool_table.off_names(self)

    def image_settings(self) -> ImageSettings:
        """Return the settings of this job the core reads.

        Returns
        -------
        ImageSettings
            The atlas and plane, how the sections are shown, the pixel
            sizes and channel names the host gave, the picture level, the
            settings a border trace depends on, the whole form as text and
            what a fit says of a section with no written transform.
        """
        inputs: dict[str, object] = self.inputs
        supplied = inputs.get("pixel_size_um")
        stack_um, file_um = supplied_pixel_sizes(supplied)
        names = inputs.get("channel_names")
        channel_names = None
        if isinstance(names, list):
            listed = cast("list[object]", names)
            channel_names = tuple(str(name) for name in listed)
        form = self.to_dict()
        nonlinear = cast("dict[str, object]", form["nonlinear"])
        return ImageSettings(
            atlas=self.atlas,
            plane=as_plane(self.plane),
            preprocess=self.preprocess,
            host_preprocessing=self.host_preprocessing,
            supplied_pixel_size=supplied,
            stack_pixel_size_um=stack_um,
            file_pixel_sizes_um=file_um,
            channel_names=channel_names,
            picture_level=self.image_resolution,
            trace_settings=tuple(nonlinear.items()),
            settings_text=json.dumps(form, sort_keys=True, default=str),
            missing_transform=self.missing_transform(),
        )

    def missing_transform(self) -> str:
        """Return what a fit says of a section with no written transform.

        Returns
        -------
        str
            :data:`NO_TRANSFORM`; for a job whose Linear task is off,
            :func:`no_transform_linear_off` for its disabled tools.
        """
        if self.has("transform"):
            return NO_TRANSFORM
        return no_transform_linear_off(ToolsOn(frozenset(self.disabled)))

    @property
    def thickness_mm(self) -> float | None:
        value = self.position.thickness_um
        return value / 1000.0 if value is not None else None

    @property
    def interval_mm(self) -> float | None:
        value = self.position.interval_um
        return value / 1000.0 if value is not None else None

    @property
    def start_mm(self) -> float | None:
        """The first section's initial position (``position.start_mm``)."""
        value = self.position.start_mm
        return float(value) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        """The spec as ``job.json`` holds it: JSON values, a setting at its
        empty value left out where the file never listed it."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Rebuild from :meth:`to_dict` output.

        ``ValueError`` naming each problem in a sentence: an unknown key as
        ``Unknown job setting(s) [...]`` (a nested one as ``section.key``), a
        refused value by its own message, a wrong type by its place.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            problems = exc.errors()
            unknown = sorted(
                ".".join(str(part) for part in problem["loc"])
                for problem in problems
                if problem["type"] == "extra_forbidden"
            )
            said = [f"Unknown job setting(s) {unknown}"] if unknown else []
            for problem in problems:
                if problem["type"] == "extra_forbidden":
                    continue
                place = ".".join(str(part) for part in problem["loc"])
                text = problem["msg"].removeprefix("Value error, ")
                said.append(
                    f"{place}: {text}" if place and problem["type"] != "value_error" else text
                )
            raise ValueError("; ".join(said)) from None

    def with_changes(self, **updates: Any) -> JobSpec:
        """A spec like this one with *updates* (``plugins=...``, ``inputs=...``),
        checked as a new spec is."""
        return self.from_dict({**self.to_dict(), **updates})
