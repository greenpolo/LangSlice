"""DeepSlice as a LangSlice tool, ``deepslice_place``: the adapter.

It is a tool of every job (its row of the tool table names
``langslice.engines.deepslice.plugin:tools``).

Options (``plugin_config``; any other is refused when the tool is built):
``ensemble`` (true: the paper's two-network model, DeepSlice's own default
for mouse), ``long_edge`` (3072: the long edge, in
pixels, of the image written for DeepSlice; a file of that size or smaller
goes in with its delivered pixels, and DeepSlice shrinks it to 299 itself)
and ``cache_folder`` (where its weights are kept; empty is
``~/.cache/langslice/deepslice``). DeepSlice 1.2.8 is vendored beside this
module and runs in this process (``engine.py``) on TensorFlow, imported
only when the tool runs; without the ``langslice[engines]`` extra, or when
its weights cannot be fetched, a call is refused ``ENGINE_UNAVAILABLE``.

Offered on the Allen mouse atlases and ``ccfv3augmented_mouse_25um``,
coronal jobs only; absent (and unnamed in any other tool's text) elsewhere.
Its inputs are the sections as the agent oriented them (quarter turn and
flip), the delivered pixels, no preprocessing. Its parts: ``positions``,
``angles`` and, only while the transform task is on and ``elastix_affine``
is on (the marks in the description), ``transform``. A call writes the
parts asked for as one undo step (``writes.py``); an empty ``write``
proposes and writes nothing. Every call saves
``plugins/deepslice_place/calls/<n>_place/``: the images written, DeepSlice's run
(``engine/``), ``out.json`` (the run's request and answer),
``settings.json`` and ``rows.json`` (``reply.py``).

Signs. With ``order`` anterior first, DeepSlice's depth falls along the
index, so the interval it is given is NEGATIVE; the tool passes
``-interval`` and the agent never sees or chooses a sign.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from langslice.engines._blocks import call_folder
from langslice.engines._options import known_options
from langslice.engines._positioning import (
    listed,
    long_edge,
    option,
    write_png,
)
from langslice.engines._weights import weights_folder
from langslice.engines.deepslice import engine, reply, writes
from langslice.engines.deepslice.engine import EngineRequest
from langslice.engines.deepslice.network import ENGINE
from langslice.engines.deepslice.proposal import (
    ProposalRow,
    proposal_rows,
    voted_angles,
)
from langslice.engines.deepslice.reply import Answer, Arguments, CallFacts
from langslice.engines.deepslice.writes import (
    PROPERTY,
    AnswerPart,
    PartChoice,
    Written,
)
from langslice.plugin_api import (
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
    ToolResult,
    ToolsOn,
    ToolSpec,
)
from langslice.plugins.declared import engine_tool

OPTIONS = ("ensemble", "long_edge", "cache_folder")
NAMED, MARKED = "named", "marked damage"


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    ensemble
        Whether DeepSlice averages its two mouse networks.
    long_edge
        The long edge, in pixels, of the image written for DeepSlice.
    cache_folder
        Where DeepSlice's weights are kept; empty is
        ``~/.cache/langslice/deepslice``.
    """

    ensemble: bool = True
    long_edge: int = 3072
    cache_folder: str = ""

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> Options:
        """Read the options a job's ``plugin_config`` gives.

        Parameters
        ----------
        config
            The plugin's entry of ``plugin_config``; None gives the defaults.

        Returns
        -------
        Options
            The options.

        Raises
        ------
        PluginError
            For an option the plugin does not take.
        ValueError
            For a value of the wrong kind.
        """
        given = known_options("deepslice_place", config, OPTIONS)
        return cls(
            ensemble=option(given, "ensemble", cls.ensemble),
            long_edge=long_edge(given),
            cache_folder=option(given, "cache_folder", cls.cache_folder),
        )

    @property
    def weights_folder(self) -> Path:
        """Return the folder that keeps DeepSlice's weights."""
        return weights_folder(ENGINE, self.cache_folder)


EVERY_SECTION: list[str] = []
FOLDER_ORDER: list[str] = []


def _deepslice_description(on: ToolsOn) -> str:
    rough_transform = on.say(
        "elastix_affine", ", and a rough in-plane transform (scale,\nturn, shift)"
    )
    turn_first = on.say(
        "interactive_transform",
        "Turn and flip every section the right way before the call, because the stack's "
        "angle is a vote of every section's own angle.",
        "The stack's angle is a vote of every section's own angle.",
    )
    transform_replaced = on.say(
        "elastix_affine",
        ", and\na transform replaces the section's transform and clears its deformation",
    )
    damage_named = on.say("mark_damage", " (mark_damage)")
    parts = on.listed(
        [
            ("", '"positions" (every run section\'s position)'),
            ("", '"angles" (the stack\'s cutting angles)'),
            (
                "elastix_affine",
                '"transform" (each section\'s in-plane transform, poor on damaged sections)',
            ),
        ],
        last="or",
    )
    change_in_words = on.say(
        "elastix_affine",
        ", the proposed in-plane change in words (turn, scale, and shift as a\n"
        "    share of the width and height)",
    )
    return f"""\
Place sections with DeepSlice, a network trained on mouse brain sections.

DeepSlice looks at each section on its own, without the atlas, and
estimates the plane it was cut at: its position along the slicing axis
and its two tilts{rough_transform}. It then combines the sections: the stack's cutting angles
are a weighted vote of every run section's own angles, sections near the
middle of the brain counting most, and, when order is given, positions
are re-spaced along the cutting order. {turn_first} The reply lists each section's own pitch and \
yaw, so
a section mounted the other way shows. It works on
whole coronal sections; a section beyond the ends of the atlas (the tip
of the olfactory bulb, the caudal medulla) or heavily damaged
pulls every prediction off unless it is a bad section.

Writes the parts named in write, as one undo step: a changed position
or angle clears the deformation of each section it moves{transform_replaced}.
A section whose property is frozen answers FROZEN and the rest land. With
write empty nothing is written and the reply is the proposal. Runs in
about half a minute for a stack of forty.

Args:
    sections: Filenames to run (the extension may be left off); empty
        means every section. The angle vote and the spacing use exactly
        these.
    order: The cutting order, anterior to posterior, as filenames; it
        must list every section run, once. Given, DeepSlice re-orders
        the sections by it and spaces them evenly, by the interval the
        job gives, else by an estimate of its own (the reply says which).
        Empty: each section keeps the position DeepSlice predicts for it
        and no spacing is applied. DeepSlice cannot know the cutting
        order; give it only the order you have established.
    bad_sections: Filenames DeepSlice leaves out of its angle vote and
        its spacing fit (damaged, hemisected, outside the atlas). Such a
        section is still placed, gets the stack's angles and, with
        order, a position implied by its neighbours. Left out: every
        section with marked damage{damage_named}. An empty list: none.
    write: The parts to write, a list of {parts}. Left out:
        every part the job lets you write. Empty: write nothing and show the
        proposal.
    view: True returns the picture described below; false returns none.

Returns:
    The stack: its cutting angles now and as proposed, the spacing used
    and whether DeepSlice estimated it, the bad sections and why. Per
    section, in the proposed order: filename, proposed position in
    millimetres, the change from its current position, its own pitch
    and yaw before the vote, whether it is a bad section{change_in_words}, and what was written \
(or FROZEN
    and the like). With view true, the
    positioning picture of the written sections, or, with nothing
    written, of the proposal, captioned as not written."""


def deepslice_place(
    *,
    sections: list[str] = EVERY_SECTION,
    order: list[str] = FOLDER_ORDER,
    bad_sections: list[str] | None = None,
    write: list[str] | None = None,
    view: bool = False,
) -> None: ...


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's one tool, ``deepslice_place``.

    Parameters
    ----------
    config
        The plugin's entry of the job's ``plugin_config`` (``ensemble``,
        ``long_edge``, ``cache_folder``), when the job has one.

    Returns
    -------
    list of ToolSpec
        The tool, offered on the Allen mouse atlases and the augmented
        CCFv3 atlas, coronal jobs only.
    """
    options = Options.from_config(config)
    meta = {
        "method": "DeepSlice 1.2.8",
        "commit": "b0d6965",
        "paper": "Carey et al. 2023, Nat Commun",
        "engine": "langslice.engines.deepslice",
        "ensemble": options.ensemble,
    }
    return [
        engine_tool(
            "deepslice_place",
            deepslice_place,
            Handler(options),
            description=_deepslice_description,
            version="1",
            meta=meta,
        )
    ]


@dataclass(frozen=True)
class Handler:
    """Answer ``deepslice_place`` calls with the plugin's options.

    Parameters
    ----------
    options
        The plugin's options.
    """

    options: Options

    def __call__(
        self,
        context: ToolContext,
        sections: list[str] | None = None,
        order: list[str] | None = None,
        bad_sections: list[str] | None = None,
        write: list[str] | None = None,
        view: bool = False,
    ) -> ToolResult:
        """Run DeepSlice on a call's sections and write what it asks.

        Parameters
        ----------
        context
            The tool call's context.
        sections
            Filenames to run; None or empty for every section.
        order
            The cutting order as filenames; None or empty for none.
        bad_sections
            Filenames left out of DeepSlice's vote; None for the sections
            with marked damage.
        write
            The parts to write; None for every part the job allows.
        view
            Whether the reply carries the positioning picture.

        Returns
        -------
        ToolResult
            The reply and, with ``view``, the positioning pictures.
        """
        arguments = Arguments(
            sections=list(sections or []),
            order=list(order or []),
            bad_sections=listed(bad_sections),
            write=listed(write),
            view=view,
        )
        return place(context, self.options, arguments)


def place(context: ToolContext, options: Options, arguments: Arguments) -> ToolResult:
    """Run DeepSlice on a call's sections and write the parts it asks.

    Parameters
    ----------
    context
        The tool call's context.
    options
        The plugin's options.
    arguments
        The call's arguments.

    Returns
    -------
    ToolResult
        The reply (the stack, the per-section rows, what was written) and,
        with ``view``, the positioning pictures.

    Raises
    ------
    ToolError
        ``ENGINE_UNAVAILABLE`` without the extra or its weights; ``BAD_ARGUMENT``
        for arguments that cannot run; ``FIT_FAILED`` when DeepSlice fails.
    """
    engine.require_engine()
    views = {section.id: section for section in context.reads.sections()}
    run_ids, ordered = _run_and_order(context, arguments)
    choice = _asked(context, views, run_ids, arguments.write)
    bad = _bad_sections(context, run_ids, arguments.bad_sections)
    interval = _job_interval(context)
    signed = None
    if ordered and interval:
        signed = -interval

    weights = engine.fetched_weights(options.weights_folder)
    folder = call_folder(context, "place")
    images = tuple((section, _write_png(context, section, folder, options)) for section in run_ids)
    request = EngineRequest(
        images=images,
        order=tuple(ordered),
        bad_sections=tuple(bad),
        interval_um=signed,
        ensemble=options.ensemble,
        weights=weights,
        out_folder=folder / "engine",
    )
    run = engine.run_deepslice(request)
    engine.save_run(request, run, folder / "out.json")

    rows = proposal_rows(context, run)
    rows.sort(key=_proposed_order)
    stack = voted_angles(rows)
    if choice.writes:
        written = writes.write(context, run, rows, stack, choice)
    else:
        written = writes.proposed(context, run, rows, stack, choice)

    facts = CallFacts(
        views=views,
        run_ids=run_ids,
        ordered=bool(ordered),
        bad=bad,
        interval_um=interval,
        angles_frozen=any("angles" in view.frozen for view in views.values()),
    )
    answer = Answer(facts, choice, run, rows, stack, written)
    settings = reply.Settings(
        arguments=arguments,
        interval_um_signed_passed=signed,
        ensemble=options.ensemble,
        long_edge=options.long_edge,
    )
    reply.save_record(answer, settings, folder)
    pictures = _pictures(context, answer, view=arguments.view)
    return reply.answered(answer, pictures, view_sentence=not arguments.view)


def _proposed_order(row: ProposalRow) -> tuple[bool, float, str]:
    """Order rows by proposed position; unplaced rows last."""
    return row.position_mm is None, row.position_mm or 0.0, row.id


def _run_and_order(context: ToolContext, arguments: Arguments) -> tuple[list[str], list[str]]:
    """Return the section ids run and the cutting order as ids."""
    run = [section.id for section in context.reads.sections()]
    if arguments.sections:
        run = [context.reads.section(name).id for name in arguments.sections]
    if len(set(run)) != len(run):
        raise ToolError(RefusalCode.BAD_ARGUMENT, "a section is named twice in sections")

    ordered = [context.reads.section(name).id for name in arguments.order]
    repeated = len(set(ordered)) != len(ordered)
    if ordered and (repeated or set(ordered) != set(run)):
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "order must list every section run exactly once, anterior to posterior",
        )
    if ordered and len(run) < 2:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "order needs at least two sections")

    return run, ordered


def _bad_sections(context: ToolContext, run: list[str], names: list[str] | None) -> dict[str, str]:
    """Return the bad sections and why: those named, else the marked."""
    if names is None:
        bad = {section: MARKED for section in run if context.reads.section(section).damaged_regions}
    else:
        bad = _named_bad(context, run, names)
    if len(bad) >= len(run):
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "every section is a bad section; DeepSlice needs at least one section to vote",
        )
    return bad


def _named_bad(context: ToolContext, run: list[str], names: list[str]) -> dict[str, str]:
    """Return the bad sections a call names, each run."""
    named = [context.reads.section(name).id for name in names]
    outside = [section for section in named if section not in run]
    if outside:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"bad_sections names sections that are not run: {outside}",
        )
    return dict.fromkeys(named, NAMED)


def _asked(
    context: ToolContext,
    views: Mapping[str, SectionView],
    run: Sequence[str],
    write: list[str] | None,
) -> PartChoice:
    """Resolve the parts a call writes: those named, else every allowed."""
    available = [AnswerPart.POSITIONS, AnswerPart.ANGLES]
    if is_transform_available(context):
        available.append(AnswerPart.TRANSFORM)
    if write is None:
        parts = _writable(available, views, run)
        return PartChoice(available=available, parts=parts, propose_only=False)

    unknown = [part for part in write if part not in available]
    if unknown:
        takes = " and ".join(repr(str(part)) for part in available)
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"write takes {takes} (a list; empty writes nothing); got {unknown}",
        )
    parts = [part for part in available if part in write]
    return PartChoice(available=available, parts=parts, propose_only=not write)


def _writable(
    available: list[AnswerPart],
    views: Mapping[str, SectionView],
    run: Sequence[str],
) -> list[AnswerPart]:
    """Return the parts some section run lets the agent write."""
    parts: list[AnswerPart] = []
    for part in available:
        if any(PROPERTY[part] not in views[name].frozen for name in run):
            parts.append(part)
    return parts


def is_transform_available(context: ToolContext) -> bool:
    """Tell whether the ``transform`` part exists for a job.

    It exists exactly where the description names it: the transform task
    is on, ``elastix_affine`` is on (so its sentences stand), and some
    section's affine is not frozen.

    Parameters
    ----------
    context
        The tool call's context.

    Returns
    -------
    bool
        True where a call may write in-plane transforms.
    """
    if not context.task_on("transform"):
        return False
    switched_off = frozenset(str(name) for name in context.spec.get("disabled", ()))
    if not ToolsOn(switched_off).has("elastix_affine"):
        return False
    return not all("affine" in view.frozen for view in context.reads.sections())


def _job_interval(context: ToolContext) -> float | None:
    """Return the job's section interval in micrometres, or None."""
    interval = context.spec["position"].get("interval_um")
    return None if interval is None else float(interval)


def _write_png(context: ToolContext, section: str, folder: Path, options: Options) -> Path:
    """Write a section as the agent oriented it, its delivered pixels."""
    raster = context.reads.section_raster(section, level=options.long_edge, raw=True)
    return write_png(raster.array, folder, section)


def _pictures(context: ToolContext, answer: Answer, *, view: bool) -> list[Image.Image]:
    """Return the positioning pictures, none without ``view``.

    The written sections as they now stand; with nothing written, the
    proposal, captioned as not written.
    """
    if not view:
        return []
    if _has_landed(answer.choice, answer.written):
        positions = answer.written.positions
        moved = [row.id for row in answer.rows if row.id in positions]
        return context.pictures.positioning_picture(moved or [row.id for row in answer.rows])
    return _proposal_pictures(context, answer)


def _proposal_pictures(context: ToolContext, answer: Answer) -> list[Image.Image]:
    """Return the positioning pictures of the proposal, not written."""
    proposed = answer.choice.shown
    with_position = AnswerPart.POSITIONS in proposed
    angles = answer.stack if AnswerPart.ANGLES in proposed else None
    proposal: dict[object, dict[str, object]] = {}
    for row in answer.rows:
        if row.refused is not None:
            continue
        plane: dict[str, object] = {}
        if with_position and row.position_mm is not None:
            plane["position_mm"] = row.position_mm
        if angles:
            plane["angles"] = angles
        proposal[row.id] = plane
    return context.pictures.positioning_picture(list(proposal), proposal=proposal)


def _has_landed(choice: PartChoice, written: Written) -> bool:
    """Tell whether a call wrote positions or the stack's angles."""
    landed = bool(written.positions) or written.angles_written
    return choice.writes and landed
