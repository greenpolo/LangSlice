"""Offer Brainways' atlas-position network to the agent as one tool.

It is a tool of every job (its row of the tool table names
``langslice.engines.brainways.plugin:tools``).
Options (``plugin_config``): ``device`` (``auto``, ``cpu``, ``cuda`` or a
numbered GPU such as ``cuda:1``; ``auto`` takes the GPU when torch sees
one), ``long_edge`` (the long edge, pixels, of the one channel handed to
Brainways, which makes its own 1024-pixel thumbnail; 3072) and
``cache_folder`` (where Brainways' weights are kept; empty is
``~/.cache/langslice/brainways``). Any other option is refused when the
tool is built.

The network runs in LangSlice's process (``network.py`` and ``images.py``
beside this file: the part of Brainways 0.1.16.3 that finds a position,
vendored under the GPL-3.0). It needs torch, a core dependency; a call
where torch does not import, or where the weights cannot be fetched, is
refused with ``ENGINE_UNAVAILABLE``.

One tool, ``brainways_ap``, offered on ``whs_sd_rat_39um`` coronal jobs
only and absent elsewhere. Its input is ONE raw channel of each section as
the agent oriented it (quarter turn and flip): the job's first raw channel
unless the call names another. Brainways gives the position only: no
angles, no in-plane fit. ``write`` is a list of parts, here only
``"positions"``; empty proposes and writes nothing. Every call saves
``plugins/brainways_ap/calls/<n>_ap/``: the images handed to the network,
``settings.json`` and ``rows.json``.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.engines import _devices
from langslice.engines._blocks import call_folder
from langslice.engines._options import known_options
from langslice.engines._positioning import (
    clamped_reply,
    clamped_rows,
    listed,
    long_edge,
    option,
    rounded,
    row_errors,
    view_note,
    write_png,
)
from langslice.engines.brainways import images, network
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolsOn,
    ToolSpec,
    WriteResult,
)
from langslice.plugins.declared import engine_tool

PARTS = ("positions",)
ATLAS = "whs_sd_rat_39um"
ATLAS_SHAPE = (1024, 512, 512)
ATLAS_UM = 39.0
ENGINE_VERSION = "Brainways 0.1.16.3 (1c43b0b), vendored"
OPTIONS = ("device", "long_edge", "cache_folder")
PROPOSAL_NOTE = (
    "view true here draws the proposal; the positioning view of look "
    "shows the sections at their current positions."
)


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options, from ``plugin_config``.

    Attributes
    ----------
    device
        Where the network runs: ``auto`` (the GPU when torch sees one),
        ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    long_edge
        The long edge, pixels, of the channel handed to Brainways.
    cache_folder
        Where Brainways' weights are kept; empty is the default folder.
    """

    device: str = _devices.AUTO
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
            The options, each checked.

        Raises
        ------
        PluginError
            For an option the plugin does not take.
        ValueError
            For a value of the wrong kind.
        """
        given = known_options("brainways_ap", config, OPTIONS)
        device = option(given, "device", cls.device)
        if not _devices.DEVICE_OPTION.fullmatch(device):
            raise ValueError(f"device is auto, cpu, cuda or cuda:N; got {device!r}")
        return cls(
            device=device,
            long_edge=long_edge(given),
            cache_folder=option(given, "cache_folder", cls.cache_folder),
        )


@dataclass(frozen=True)
class Engine:
    """Hold what one call runs every section with.

    Attributes
    ----------
    options
        The plugin's options.
    network
        Brainways' network, loaded on its device.
    planes
        The atlas planes' embeddings, made as the search asks for them.
    folder
        The call's folder, ``plugins/brainways_ap/calls/<n>_ap/``.
    """

    options: Options
    network: network.Network
    planes: network.AtlasEmbeddings
    folder: Path


@dataclass(frozen=True)
class Proposal:
    """Hold one section's answer from the network.

    Attributes
    ----------
    id
        The section's filename.
    ap
        The atlas plane the search ended on, an index along the slicing
        axis.
    position_mm
        That plane's position, atlas millimetres from the anterior edge.
    seconds
        How long the section took.
    """

    id: str
    ap: int
    position_mm: float
    seconds: float


@dataclass(frozen=True)
class Arguments:
    """Hold a call's arguments as given.

    Attributes
    ----------
    sections
        The filenames asked for; empty for every section.
    channel
        The raw channel asked for; empty for each section's first.
    write
        The parts to write; None when not given.
    view
        Whether the reply carries the picture.
    """

    sections: list[str]
    channel: str
    write: list[str] | None
    view: bool


@dataclass(frozen=True)
class CallRecord:
    """Hold what a call saves in its folder's ``settings.json``.

    Attributes
    ----------
    arguments
        The call's arguments as given.
    run
        The sections run, by filename.
    channel_used
        The raw channel read, per section.
    proposal_only
        True when the call wrote nothing.
    device
        The torch device the network ran on.
    gpu
        The GPU's name, None on the CPU.
    sha256
        The weights' SHA-256.
    torch
        The torch version.
    timings_s
        ``model_load_s`` and ``inference_s``, seconds.
    """

    arguments: Arguments
    run: list[str]
    channel_used: dict[str, str]
    proposal_only: bool
    device: str
    gpu: str | None
    sha256: str
    torch: str
    timings_s: dict[str, float]


EVERY_SECTION: list[str] = []


def _brainways_description(on: ToolsOn) -> str:
    turn_first = on.say("interactive_transform", "; turn and flip a section the right way first")
    return f"""\
Find each section's position along the slicing axis with Brainways.

Brainways, a network trained on rat and mouse brain sections, compares one
channel of a section, by itself, with the atlas's coronal planes: it says
which of two planes is the more anterior, and a search over the planes
finds the one the section cannot be told apart from. The answer is that
plane's position and nothing else: it estimates no cutting angles and no
in-plane placement, and each section is placed independently of the
others. It works on whole coronal sections{turn_first}. It can miss by millimetres on a single \
section, so check the result with the positioning view of look.

Writes the positions as one undo step when write holds "positions"; a
changed position clears the deformation of each section it moves. A
section whose position is frozen answers FROZEN and the rest land. With
write empty nothing is written and the reply is the proposal. Runs in
seconds for a stack of forty.

Args:
    sections: Filenames to run (the extension may be left off); empty
        means every section.
    channel: The raw channel Brainways reads, by name; empty means the
        section's first raw channel. It reads one channel.
    write: ["positions"] writes the positions; empty writes nothing and
        shows the proposal. Left out: ["positions"].
    view: True returns the picture described below; false returns none.

Returns:
    Per section, in the proposed order: filename, proposed position in
    millimetres, the change from its current position, and what was
    written (or FROZEN and the like). With view true, the
    positioning picture of the written sections, or, with nothing
    written, of the proposal, captioned as not written."""


def brainways_ap(
    *,
    sections: list[str] = EVERY_SECTION,
    channel: str = "",
    write: list[str] | None = None,
    view: bool = False,
) -> None: ...


def run_brainways_ap(
    context: ToolContext,
    *,
    options: Options,
    sections: Sequence[str] | None = None,
    channel: str = "",
    write: Sequence[str] | None = None,
    view: bool = False,
) -> ToolResult:
    """Run ``brainways_ap``: propose each section's position, and write it.

    Parameters
    ----------
    context
        The call's plugin context.
    options
        The plugin's options.
    sections
        Filenames of the sections to run; None or empty runs every section.
    channel
        The raw channel to read; empty is each section's first.
    write
        ``["positions"]`` writes; empty writes nothing; None is
        ``["positions"]``.
    view
        True returns the positioning picture.

    Returns
    -------
    ToolResult
        The reply (``written``, ``columns``, ``rows``, ``channel`` and,
        when they apply, ``refused``, ``clamped``, ``view``) and the
        picture.

    Raises
    ------
    ToolError
        ``BAD_ARGUMENT`` for a bad argument, ``ENGINE_UNAVAILABLE`` when the network
        cannot run here, ``FIT_FAILED`` for an atlas it does not fit.
    """
    prior_mm = {view.id: view.position_mm for view in context.reads.sections()}
    run = _sections_to_run(context, sections)
    propose_only = _propose_only(write)
    chosen = {sid: _channel_for(context, sid, channel) for sid in run}
    reference = _reference_volume(context)
    folder = call_folder(context, "ap")

    started = time.monotonic()
    device, net = _network(options)
    load_s = time.monotonic() - started
    engine = Engine(options, net, network.AtlasEmbeddings(net, reference), folder)
    proposals = [_propose(context, engine, sid, chosen[sid]) for sid in run]
    proposals.sort(key=lambda row: (row.position_mm, row.id))

    result = _write_positions(context, proposals, propose_only)
    written = _written(result)

    columns, table = _table(proposals, prior_mm, result, propose_only)
    reply: dict[str, object] = {
        "written": _written_word(written, propose_only),
        "columns": columns,
        "rows": table,
    }
    used = sorted(set(chosen.values()))
    reply["channel"] = used if len(used) > 1 else used[0]
    if result is not None and not result.ok:
        reply["refused"] = f"{result.error}: {result.message}"
    clamped = clamped_rows(result)
    if clamped:
        reply["clamped"] = clamped_reply(clamped)
    if not view:
        reply["view"] = view_note(bool(written), PROPOSAL_NOTE)

    record = CallRecord(
        arguments=Arguments(list(sections or []), channel, listed(write), view),
        run=run,
        channel_used=chosen,
        proposal_only=propose_only,
        device=device,
        gpu=_devices.gpu_name(device),
        sha256=net.sha256,
        torch=network.torch_version(),
        timings_s={
            "model_load_s": load_s,
            "inference_s": sum(row.seconds for row in proposals),
        },
    )
    _save_call(folder, record, columns, table, proposals)
    pictures = _pictures(context, proposals, written, view=view)
    return ToolResult(reply, pictures)


def _write_positions(
    context: ToolContext, proposals: list[Proposal], propose_only: bool
) -> WriteResult | None:
    if propose_only:
        return None
    rows = [{"id": row.id, "position_mm": row.position_mm} for row in proposals]
    with context.writes.writing():
        return context.writes.set_positions(rows)


def _sections_to_run(context: ToolContext, sections: Sequence[str] | None) -> list[str]:
    if sections:
        run = [context.reads.section(name).id for name in sections]
    else:
        run = [view.id for view in context.reads.sections()]
    if len(set(run)) != len(run):
        raise ToolError(RefusalCode.BAD_ARGUMENT, "a section is named twice in sections")
    return run


def _propose_only(write: Sequence[str] | None) -> bool:
    if write is None:
        return False
    unknown = [part for part in write if part not in PARTS]
    if unknown:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f'write takes a list: ["positions"] or empty (writes nothing); got {unknown}',
        )
    return not write


def _channel_for(context: ToolContext, sid: str, channel: str) -> str:
    names = context.reads.channels(sid)
    if not names:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"{sid} has no raw channel")
    if channel and channel not in names:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"{sid} has no channel {channel!r}; its channels: {', '.join(names)}",
        )
    return channel or names[0]


def _reference_volume(context: ToolContext) -> network.ReferenceVolume:
    atlas = context.reads.atlas()
    shape = tuple(atlas.shape)
    off_grid = any(abs(um - ATLAS_UM) > 1e-6 for um in atlas.resolution_um)
    if atlas.name != ATLAS or shape != ATLAS_SHAPE or off_grid:
        raise ToolError(
            RefusalCode.FIT_FAILED,
            f"Brainways: {atlas.name} loaded with shape {list(shape)} and "
            f"resolution {list(atlas.resolution_um)} um; expected "
            f"{ATLAS} {list(ATLAS_SHAPE)} at {ATLAS_UM} um",
        )
    return network.ReferenceVolume.from_voxels(context.frames.atlas_volume())


def _network(options: Options) -> tuple[str, network.Network]:
    try:
        device = _devices.resolve_device(options.device)
        gpu_missing = _devices.on_gpu(device) and not _devices.cuda_available()
    except ImportError as error:
        raise ToolError(
            RefusalCode.ENGINE_UNAVAILABLE,
            f"Brainways cannot run here: torch does not import ({error}). Use another method, "
            "or ask the user to install it.",
        ) from error
    if gpu_missing:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"Brainways: device {options.device} was asked for and torch sees no GPU",
        )
    try:
        net = network.load_network(device, cache_folder=options.cache_folder)
    except (OSError, ValueError) as error:
        raise ToolError(
            RefusalCode.ENGINE_UNAVAILABLE,
            f"Brainways cannot run here: its weights could not be fetched "
            f"from Hugging Face ({network.REPOSITORY}): {error}. Use another method, "
            "or ask the user to install it.",
        ) from error
    return device, net


def _propose(context: ToolContext, engine: Engine, sid: str, channel: str) -> Proposal:
    started = time.monotonic()
    raster = context.reads.section_raster(
        sid, level=engine.options.long_edge, raw=True, images=[channel]
    )
    pixels = np.asarray(raster.array, np.uint8)[:, :, 0]
    write_png(pixels, engine.folder, sid)

    thumbnail = images.section_thumbnail(pixels)
    limits = network.AP_LIMITS[ATLAS]
    ap = network.find_ap(engine.network, thumbnail, engine.planes, limits)
    position_mm = ap * ATLAS_UM / 1000.0
    return Proposal(sid, ap, position_mm, time.monotonic() - started)


def _written(result: WriteResult | None) -> set[str]:
    if result is None or not result.ok:
        return set()
    return set(result.written)


def _written_word(written: set[str], propose_only: bool) -> str:
    if propose_only:
        return "nothing: proposal only"
    if written:
        return "positions"
    return "nothing"


def _table(
    proposals: list[Proposal],
    prior_mm: dict[str, float],
    result: WriteResult | None,
    propose_only: bool,
) -> tuple[list[str], list[list[str | float | None]]]:
    written = _written(result)
    errors = row_errors(result)
    columns = ["section", "position_mm", "change_mm"]
    if not propose_only:
        columns.append("written")

    table: list[list[str | float | None]] = []
    for row in proposals:
        change = rounded(row.position_mm - prior_mm[row.id], 3)
        cell: list[str | float | None] = [
            row.id,
            rounded(row.position_mm, 3),
            change,
        ]
        if propose_only:
            table.append(cell)
        elif row.id in written:
            table.append([*cell, "position"])
        else:
            table.append([*cell, errors.get(row.id, "nothing")])
    return columns, table


def _pictures(
    context: ToolContext,
    proposals: list[Proposal],
    written: set[str],
    *,
    view: bool,
) -> list[Image.Image]:
    if not view:
        return []
    if written:
        ordered = sorted(written, key=lambda sid: context.reads.section(sid).index)
        return context.pictures.positioning_picture(ordered)
    proposal: dict[object, Mapping[str, object]] = {
        row.id: {"position_mm": row.position_mm} for row in proposals
    }
    return context.pictures.positioning_picture([row.id for row in proposals], proposal=proposal)


def _save_call(
    folder: Path,
    record: CallRecord,
    columns: list[str],
    table: list[list[str | float | None]],
    proposals: list[Proposal],
) -> None:
    settings = {
        **asdict(record),
        "atlas": ATLAS,
        "model_id": network.MODEL_ID,
        "weights_revision": network.REVISION,
        "engine": ENGINE_VERSION,
        "ap_limits": list(network.AP_LIMITS[ATLAS]),
    }
    (folder / "settings.json").write_text(
        json.dumps(settings, indent=1), encoding="utf-8", newline="\n"
    )
    rows = {
        "columns": columns,
        "rows": table,
        "proposal": [asdict(row) for row in proposals],
    }
    (folder / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Declare ``brainways_ap`` with the job's options.

    Parameters
    ----------
    config
        The plugin's entry of the job's ``plugin_config`` (``device``,
        ``long_edge``, ``cache_folder``), when the job has one.

    Returns
    -------
    list of ToolSpec
        The one tool.
    """
    options = Options.from_config(config)
    meta = {
        "method": "Brainways 0.1.16.3",
        "paper": "Kantor et al. 2025, Neuropsychopharmacology",
        "engine": "langslice.engines.brainways",
        "commit": "1c43b0b",
        "weights": network.REVISION,
    }
    handler = functools.partial(run_brainways_ap, options=options)
    return [
        engine_tool(
            "brainways_ap",
            brainways_ap,
            handler,
            description=_brainways_description,
            version="2",
            meta=meta,
        )
    ]
