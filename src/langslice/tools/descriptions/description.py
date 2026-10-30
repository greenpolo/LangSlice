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
from typing import cast

from langslice.core.fits.trace.prompts import border_correction_tool_prompt
from langslice.core.fits.trace.trace import PLANE_PLACEHOLDER
from langslice.core.space import Plane
from langslice.job.plugins import LoadedPlugin, of_spec
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import tools_for

#: The default of an argument the call must give.
REQUIRED = inspect.Parameter.empty

#: LangSlice's own agent's door (and the library's), whose first message
#: carries the opening pictures and the status table as parts of their own.
AGENT_DOOR = "agent"

#: The agent CLI's door, which words some passages its own way.
CLI_DOOR = "cli"

#: The doors a tool is declared for: ``agent``, ``mcp`` (Claude Desktop) and
#: ``cli`` (the agent CLI).
DOORS = (AGENT_DOOR, "mcp", CLI_DOOR)


class Offered(Enum):
    """The fact of a run an argument exists in, beside what it needs on."""

    #: In every run.
    ALWAYS = auto()
    #: Where the caller chooses each picture's size (``resolution``).
    SIZE_CHOSEN = auto()
    #: Where the host does not force the change tools' pictures on (``view``).
    VIEW_CHOSEN = auto()
    #: Where the Positioning task is on (``position_sections``' sections).
    POSITIONING = auto()
    #: Where the door caps no call's pictures (``look``'s ``sheet``).
    UNCAPPED = auto()


@dataclass(frozen=True)
class DescriptionFacts:
    """What of a run changes its tools' descriptions and arguments."""

    #: The caller chooses each picture's size (``look``'s ``resolution``).
    auto: bool = False
    #: Who reads the declaration (:data:`DOORS`): how a background tool
    #: answers.
    door: str = AGENT_DOOR
    #: The host forces the change tools' pictures on: no ``view`` argument.
    forced_view: bool = False
    #: ``position_sections`` takes ``sections`` (the Positioning task is on).
    positions: bool = True
    #: The door puts no per-call cap on pictures or sections (the agent CLI
    #: and the library; ``Job.picture_cap`` None).
    uncapped: bool = False
    #: The base image prompt ``trace_borders`` carries; empty: none.
    prompt: str = ""
    #: What the job switches off (``JobSpec.switched_off``).
    switched_off: frozenset[str] = frozenset()
    #: The tools the run lacks because their task is off. A tool plugin's
    #: text renders by these as well as by ``switched_off`` (a plugin
    #: sentence that needs ``elastix_affine`` stands only where the Linear
    #: task is on); the native descriptions render by ``switched_off`` alone.
    absent: frozenset[str] = frozenset()
    #: The tool plugins the job loads, compared by identity: their tools are
    #: declared from their own stubs. Empty: no plugin text anywhere.
    plugins: tuple[LoadedPlugin, ...] = ()
    #: The job's atlas brain's size next to the adult mouse's
    #: (``core.atlas.brain_size.brain_scale``): a :class:`Distance` is the
    #: adult mouse's times it.
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
        absent: frozenset[str] = frozenset()
        if plugins:
            tools = tools_for(spec)
            absent = frozenset(
                name for name, tool in tools.items() if not tool.when(spec)
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
    plane = cast("Plane", spec.plane)  # the spec checks it is one
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


#: A passage a fact of the run words.
Slot = PerCap | PerDoor | Distance


@dataclass(frozen=True)
class Argument:
    """One argument of a tool: its type, its default and its text."""

    #: The type every door's schema is built from.
    annotation: object
    #: The value a call that leaves the argument out gets, or
    #: :data:`REQUIRED`. A list or dict default is read, never changed.
    default: object
    #: What the model reads under the argument's name.
    text: str
    _: KW_ONLY
    #: The names the argument needs on, as a needs mark writes them
    #: (``look:overlay``, ``look:atlas|look:positioning``); empty: none.
    needs: str = ""
    #: The fact of a run the argument exists in.
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
