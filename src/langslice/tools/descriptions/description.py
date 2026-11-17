"""The shape of a tool's description, and the facts of a run that word it.

Each tool's description is data, one :class:`ToolDescription` per tool, in
the module of its family (``looking``, ``changing``, ``fitting``,
``landmarks``, ``bookkeeping``); ``render`` lays it out for each door.

=================  ===================================================
field              what the model reads
=================  ===================================================
``summary``        the first line
``body``           the paragraphs under it; one that renders empty goes
``arguments``      ``Args:``, one entry per argument the run offers
``returns``        ``Returns:`` and its text
``prompt_heading`` the paragraph before the run's base image prompt
``slots``          passages a fact of the run words: ``${name}``
=================  ===================================================

A text is prose at no indentation, broken into lines as the model reads
it; a line too long for the source ends in a backslash, which joins it to
the next. A passage that names a tool or an option is written with what
it needs (:mod:`langslice.job.settings.needs`), and an argument names what
it needs in ``needs``. A slot is ``${name}``, filled before the marks are
rendered.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum, auto

from langslice.core.fits.trace.prompts import border_correction_tool_prompt
from langslice.core.fits.trace.trace import PLANE_PLACEHOLDER
from langslice.core.space import as_plane
from langslice.job.plugins import LoadedPlugin, of_spec
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import tools_for

REQUIRED = inspect.Parameter.empty

AGENT_DOOR = "agent"

CLI_DOOR = "cli"

DOORS = (AGENT_DOOR, "mcp", CLI_DOOR)


class Offered(Enum):
    """The fact of a run an argument exists in, beside what it needs on."""

    ALWAYS = auto()
    SIZE_CHOSEN = auto()
    VIEW_CHOSEN = auto()
    POSITIONING = auto()
    UNCAPPED = auto()


@dataclass(frozen=True)
class DescriptionFacts:
    """What of a run changes its tools' descriptions and arguments."""

    auto: bool = False
    door: str = AGENT_DOOR
    forced_view: bool = False
    positions: bool = True
    uncapped: bool = False
    prompt: str = ""
    switched_off: frozenset[str] = frozenset()
    absent: frozenset[str] = frozenset()
    plugins: tuple[LoadedPlugin, ...] = ()
    brain_scale: float = 1.0

    @classmethod
    def from_spec(
        cls,
        spec: JobSpec,
        *,
        auto: bool,
        image_model: bool = True,
        door: str = AGENT_DOOR,
        prompt: str | None = None,
        uncapped: bool | None = None,
        brain_scale: float = 1.0,
    ) -> DescriptionFacts:
        """Return the facts of a run of ``spec``.

        Parameters
        ----------
        spec
            The job's settings.
        auto
            Whether the caller chooses each picture's size.
        image_model
            False when the door cannot reach the job's image model, so no
            base prompt is described.
        door
            The door the run is declared for, one of :data:`DOORS`.
        prompt
            The image model's own base prompt; None: the job's own
            (``nonlinear.image_prompt``), else LangSlice's for the run's
            plane and provider.
        uncapped
            Whether the door caps no call's pictures; None: only the CLI.
        brain_scale
            The job's atlas brain's size next to the adult mouse's.

        Returns
        -------
        DescriptionFacts
            The facts the run's tool descriptions are rendered with.
        """
        traced = spec.has("nonlinear") and spec.nonlinear.uses_image_model
        if not (traced and image_model):
            prompt = ""
        elif prompt is None:
            prompt = _job_prompt(spec)

        plugins = tuple(of_spec(spec))
        absent = frozenset(
            name
            for name, tool in tools_for(spec).items()
            if not tool.when(spec)
        )
        if uncapped is None:
            uncapped = door == CLI_DOOR
        return cls(
            auto=auto,
            door=door,
            forced_view=spec.force_view,
            positions=spec.has("position"),
            uncapped=uncapped,
            prompt=prompt,
            switched_off=frozenset(spec.switched_off),
            absent=absent,
            plugins=plugins,
            brain_scale=brain_scale,
        )


def _job_prompt(spec: JobSpec) -> str:
    own = spec.nonlinear.image_prompt
    if own is not None:
        return own.replace(PLANE_PLACEHOLDER, spec.plane)
    plane = as_plane(spec.plane)
    return border_correction_tool_prompt(
        plane, provider=spec.nonlinear.provider
    )


@dataclass(frozen=True)
class PerCap:
    """A passage worded for a door that caps a call's pictures or not."""

    capped: str
    uncapped: str


@dataclass(frozen=True)
class PerDoor:
    """A passage the agent CLI words its own way (how a long call answers)."""

    agent: str
    cli: str


@dataclass(frozen=True)
class Distance:
    """A distance stated for the adult mouse, scaled to the job's brain.

    ``words`` holds ``{}`` where the number of micrometres goes;
    ``mouse_words``, when given, is the passage at the mouse's own scale,
    whose line breaks there.
    """

    um: float
    words: str
    mouse_words: str = ""


Slot = PerCap | PerDoor | Distance


@dataclass(frozen=True)
class Argument:
    """One argument of a tool: its type, its default and its text."""

    annotation: object
    default: object
    text: str
    _: KW_ONLY
    needs: str = ""
    offered: Offered = Offered.ALWAYS


@dataclass(frozen=True)
class ToolDescription:
    """One tool's description as data (see the module text)."""

    summary: str
    body: tuple[str, ...] = ()
    arguments: Mapping[str, Argument] = field(
        default_factory=dict[str, Argument]
    )
    returns: str = ""
    prompt_heading: str = ""
    slots: Mapping[str, Slot] = field(default_factory=dict[str, Slot])
