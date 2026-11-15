"""Offer Brainways' atlas-position network to the agent as one tool.

It is a tool of every job (its row of the tool table names
``langslice.engines.brainways.plugin:tools``).
Options (``plugin_config``): ``device`` (``auto``, ``cuda`` or ``cpu``;
``auto`` takes the GPU when torch sees one), ``long_edge`` (the long edge,
pixels, of the one channel handed to Brainways, which makes its own
1024-pixel thumbnail; 3072) and ``cache_folder`` (where Brainways' weights
are kept; empty is ``~/.cache/langslice/brainways``).

The network runs in LangSlice's process (``network.py`` and ``images.py``
beside this file: the part of Brainways 0.1.16.3 that finds a position,
vendored under the GPL-3.0). It needs torch, a core dependency; a call
where torch does not import, or where the weights cannot be fetched, is
refused with ``ENGINE_MISSING``.

One tool, ``brainways_ap``, offered on ``whs_sd_rat_39um`` coronal jobs
only and absent elsewhere. Its input is ONE raw channel of each section as
the agent oriented it (quarter turn and flip): the job's first raw channel
unless the call names another. Brainways gives the position only: no
angles, no in-plane fit. ``write`` is a list of parts, here only
``"positions"``; empty proposes and writes nothing. Every call saves
``plugins/brainways_ap/calls/<n>/``: the images handed to the network,
``settings.json`` and ``rows.json``.
"""

from __future__ import annotations

import enum
import functools
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from langslice.engines.brainways import images, network
from langslice.engines.brainways.network import Device
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    WriteResult,
)

#: The tool's one writable part.
PARTS = ("positions",)
#: The atlas Brainways' rat model was trained on, as this tool reads it:
#: its shape and voxel size, which ``position_mm = ap * 0.039`` assumes.
ATLAS = "whs_sd_rat_39um"
ATLAS_SHAPE = (1024, 512, 512)
ATLAS_UM = 39.0
#: What Brainways this is, kept with every call's settings.
ENGINE_VERSION = "Brainways 0.1.16.3 (1c43b0b), vendored"
#: The command that installs what the network needs.
INSTALL = 'pip install "langslice[engines]"'
#: The options ``plugin_config`` may give.
OPTIONS = ("device", "long_edge", "cache_folder")


class RowStatus(enum.StrEnum):
    """Name the status of a row ``set_positions`` answers with."""

    OK = "ok"


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options, from ``plugin_config``.

    Attributes
    ----------
    device
        Where the network runs; ``auto`` takes the GPU when torch sees one.
    long_edge
        The long edge, pixels, of the channel handed to Brainways.
    cache_folder
        Where Brainways' weights are kept; empty is the default folder.
    """

    device: Device = Device.AUTO
    long_edge: int = 3072
    cache_folder: str = ""

    @classmethod
    def from_configs(
        cls, configs: Sequence[Mapping[str, str | int]]
    ) -> Options:
        """Read the options a job's ``plugin_config`` gives.

        Parameters
        ----------
        configs
            The plugin's entry of ``plugin_config``, when the job has one;
            empty gives the defaults.

        Returns
        -------
        Options
            The options, each checked.

        Raises
        ------
        ValueError
            For an unknown option or a value of the wrong kind.
        """
        given: dict[str, str | int] = {}
        for config in configs:
            given.update(config)
        unknown = sorted(set(given) - set(OPTIONS))
        if unknown:
            raise ValueError(
                f"brainways_ap takes the options {', '.join(OPTIONS)}; "
                f"got {', '.join(unknown)}"
            )
        device = str(given.get("device", Device.AUTO))
        if device not in tuple(Device):
            raise ValueError(
                f"device is one of {', '.join(Device)}; got {device!r}"
            )
        return cls(
            device=Device(device),
            long_edge=int(given.get("long_edge", cls.long_edge)),
            cache_folder=str(given.get("cache_folder", cls.cache_folder)),
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
        The call's folder, ``plugins/brainways_ap/calls/<n>/``.
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
class Clamped:
    """Hold a written position the atlas range clamped.

    Attributes
    ----------
    id
        The section's filename.
    proposed_mm
        The position proposed, millimetres.
    used_mm
        The position written, millimetres.
    """

    id: str
    proposed_mm: float
    used_mm: float


@dataclass(frozen=True)
class CallRecord:
    """Hold what a call saves in its folder's ``settings.json``.

    Attributes
    ----------
    arguments
        The call's arguments as given: ``sections``, ``channel``,
        ``write``, ``view``.
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

    arguments: dict[str, list[str] | str | bool | None]
    run: list[str]
    channel_used: dict[str, str]
    proposal_only: bool
    device: Device
    gpu: str | None
    sha256: str
    torch: str
    timings_s: dict[str, float]


# --- the agent-facing declaration -------------------------------------------


def brainways_ap(
    *,
    sections: list[str] = [],  # noqa: B006 - a declaration's default
    channel: str = "",
    write: list[str] | None = None,
    view: bool = False,
) -> None:
    """Find each section's position along the slicing axis with Brainways.

    Brainways, a network trained on rat and mouse brain sections, compares one
    channel of a section, by itself, with the atlas's coronal planes: it says
    which of two planes is the more anterior, and a search over the planes
    finds the one the section cannot be told apart from. The answer is that
    plane's position and nothing else: it estimates no cutting angles and no
    in-plane placement, and each section is placed independently of the
    others. It works on whole coronal sections[[?interactive_transform ~
    ; turn and flip a section the right way first]]. It can miss by millimetres ~
    on a single section, so check the result[[?look:positioning  ~
    with the positioning view of look]].

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
        written (or FROZEN and the like). With view true, [[?look:positioning the
        positioning picture of the written sections, or, with nothing
        written, of the proposal, captioned as not written||a picture
        of the written sections]].
    """
    ...


# --- the call ---------------------------------------------------------------


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
        ``BAD_ARGS`` for a bad argument, ``ENGINE_MISSING`` when the network
        cannot run here, ``FIT_FAILED`` for an atlas it does not fit.
    """
    prior_mm = {view.id: view.position_mm for view in context.sections()}
    run = _sections_to_run(context, sections)
    propose_only = _propose_only(write)
    chosen = {sid: _channel_for(context, sid, channel) for sid in run}
    reference = _reference_volume(context)
    folder = _call_folder(context)

    started = time.monotonic()
    device, net = _network(options)
    load_s = time.monotonic() - started
    engine = Engine(
        options, net, network.AtlasEmbeddings(net, reference), folder
    )
    proposals = [_propose(context, engine, sid, chosen[sid]) for sid in run]
    proposals.sort(key=lambda row: (row.position_mm, row.id))

    result = _write_positions(context, proposals, propose_only)
    written = _written(result)

    # the reply: the table, what was written, the channel read
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
    clamped = _clamped(result)
    if clamped:
        reply["clamped"] = [
            {"id": row.id, "asked_mm": row.proposed_mm, "used_mm": row.used_mm}
            for row in clamped
        ]
    if not view and _positioning_on(context):
        reply["view"] = _view_note(written)

    record = CallRecord(
        arguments={
            "sections": list(sections or []),
            "channel": channel,
            "write": _listed(write),
            "view": view,
        },
        run=run,
        channel_used=chosen,
        proposal_only=propose_only,
        device=device,
        gpu=network.gpu_name(device),
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


def _listed(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    return list(values)


def _write_positions(
    context: ToolContext, proposals: list[Proposal], propose_only: bool
) -> WriteResult | None:
    # one undo step for every position, or nothing for a proposal
    if propose_only:
        return None
    rows = [
        {"id": row.id, "position_mm": row.position_mm} for row in proposals
    ]
    with context.writing():
        return context.set_positions(rows)


def _sections_to_run(
    context: ToolContext, sections: Sequence[str] | None
) -> list[str]:
    if sections:
        run = [context.section(name).id for name in sections]
    else:
        run = [view.id for view in context.sections()]
    if len(set(run)) != len(run):
        raise ToolError(
            RefusalCode.BAD_ARGS, "a section is named twice in sections"
        )
    return run


def _propose_only(write: Sequence[str] | None) -> bool:
    if write is None:
        return False
    unknown = [part for part in write if part not in PARTS]
    if unknown:
        raise ToolError(
            RefusalCode.BAD_ARGS,
            f'write takes a list: ["positions"] or empty (writes nothing); '
            f"got {unknown}",
        )
    return not write


def _channel_for(context: ToolContext, sid: str, channel: str) -> str:
    names = context.channels(sid)
    if not names:
        raise ToolError(RefusalCode.BAD_ARGS, f"{sid} has no raw channel")
    if channel and channel not in names:
        raise ToolError(
            RefusalCode.BAD_ARGS,
            f"{sid} has no channel {channel!r}; its channels: "
            f"{', '.join(names)}",
        )
    return channel or names[0]


def _reference_volume(context: ToolContext) -> network.ReferenceVolume:
    atlas = context.atlas()
    shape = tuple(atlas.shape)
    off_grid = any(abs(um - ATLAS_UM) > 1e-6 for um in atlas.resolution_um)
    if atlas.name != ATLAS or shape != ATLAS_SHAPE or off_grid:
        raise ToolError(
            RefusalCode.FIT_FAILED,
            f"Brainways: {atlas.name} loaded with shape {list(shape)} and "
            f"resolution {list(atlas.resolution_um)} um; expected "
            f"{ATLAS} {list(ATLAS_SHAPE)} at {ATLAS_UM} um",
        )
    return network.ReferenceVolume.from_voxels(context.atlas_volume())


def _call_folder(context: ToolContext) -> Path:
    calls = context.work_dir() / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in calls.iterdir() if p.name.isdigit()]
    folder = calls / f"{max(numbers, default=0) + 1:04d}"
    folder.mkdir()
    return folder


def _network(options: Options) -> tuple[Device, network.Network]:
    # torch and the weights, or ENGINE_MISSING
    try:
        device = network.resolve_device(options.device)
    except ImportError as error:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"Brainways cannot run here: torch does not import ({error}); "
            f"install LangSlice with its engines: {INSTALL}",
        ) from error
    if device == Device.CUDA and not network.cuda_available():
        raise ToolError(
            RefusalCode.BAD_ARGS,
            "Brainways: device cuda was asked for and torch sees no GPU",
        )
    try:
        net = network.load_network(device, cache_folder=options.cache_folder)
    except (OSError, ValueError) as error:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"Brainways cannot run here: its weights could not be fetched "
            f"from Hugging Face ({network.REPOSITORY}): {error}",
        ) from error
    return device, net


def _propose(
    context: ToolContext, engine: Engine, sid: str, channel: str
) -> Proposal:
    started = time.monotonic()
    raster = context.section_raster(
        sid, level=engine.options.long_edge, raw=True, images=[channel]
    )
    # one channel: the raster holds it as three equal planes
    pixels = np.asarray(raster.array, np.uint8)[:, :, 0]
    name = sid if sid.lower().endswith(".png") else sid + ".png"
    path = engine.folder / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)

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


def _round(value: float, digits: int) -> float:
    return round(value, digits) + 0.0  # + 0.0: no "-0.0"


def _table(
    proposals: list[Proposal],
    prior_mm: dict[str, float | None],
    result: WriteResult | None,
    propose_only: bool,
) -> tuple[list[str], list[list[str | float | None]]]:
    # per section: filename, position, change from the position it held
    # before the call, and what was written
    written = _written(result)
    errors = _row_errors(result)
    columns = ["section", "position_mm", "change_mm"]
    if not propose_only:
        columns.append("written")

    table: list[list[str | float | None]] = []
    for row in proposals:
        before = prior_mm[row.id]
        change = None
        if before is not None:
            change = _round(row.position_mm - before, 3)
        cell: list[str | float | None] = [
            row.id,
            _round(row.position_mm, 3),
            change,
        ]
        if propose_only:
            table.append(cell)
        elif row.id in written:
            table.append([*cell, "position"])
        else:
            table.append([*cell, errors.get(row.id, "nothing")])
    return columns, table


def _row_errors(result: WriteResult | None) -> dict[str, str]:
    if result is None or not result.ok:
        return {}
    errors: dict[str, str] = {}
    for row in cast("list[dict[str, object]]", result.rows):  # JSON values
        if row.get("status") != RowStatus.OK:
            errors[str(row.get("id"))] = str(row.get("error", "NOT_WRITTEN"))
    return errors


def _clamped(result: WriteResult | None) -> list[Clamped]:
    if result is None or not result.ok:
        return []
    return [
        Clamped(
            str(row["id"]),
            _millimetres(row["clamped_from_mm"]),
            _millimetres(row["position_mm"]),
        )
        for row in cast("list[dict[str, object]]", result.rows)
        if "clamped_from_mm" in row
    ]


def _millimetres(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"a position is a number; got {value!r}")
    return float(value)


def _positioning_on(context: ToolContext) -> bool:
    listed = cast("list[object]", context.spec.get("disabled") or [])
    disabled = [str(name) for name in listed]
    return "look:positioning" not in disabled and "look" not in disabled


def _view_note(written: set[str]) -> str:
    if written:
        return (
            "The positioning view of look shows the sections at these "
            "positions."
        )
    return (
        "view true here draws the proposal; the positioning view of look "
        "shows the sections at their current positions."
    )


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
        ordered = sorted(written, key=lambda sid: context.section(sid).index)
        return context.positioning_picture(ordered)
    proposal: dict[object, Mapping[str, object]] = {
        row.id: {"position_mm": row.position_mm} for row in proposals
    }
    return context.positioning_picture(
        [row.id for row in proposals], proposal=proposal
    )


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
    (folder / "rows.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8", newline="\n"
    )


# --- the factory ------------------------------------------------------------


def tools(*config: Mapping[str, str | int]) -> list[ToolSpec]:
    """Declare ``brainways_ap`` with the job's options.

    Parameters
    ----------
    *config
        The plugin's entry of the job's ``plugin_config`` (``device``,
        ``long_edge``, ``cache_folder``), when the job has one; the loader
        passes it as the one argument.

    Returns
    -------
    list of ToolSpec
        The one tool.
    """
    options = Options.from_configs(config)
    meta = {
        "method": "Brainways 0.1.16.3",
        "paper": "Kantor et al. 2025, Neuropsychopharmacology",
        "engine": "langslice.engines.brainways",
        "commit": "1c43b0b",
        "weights": network.REVISION,
    }
    handler = functools.partial(run_brainways_ap, options=options)
    return [
        ToolSpec(
            "brainways_ap",
            brainways_ap,
            handler,
            kind="write",
            group="Positioning",
            task="position",
            writes=("position",),
            needs=(("look:positioning",),),
            atlases=(ATLAS,),
            planes=("coronal",),
            version="2",
            meta=meta,
        )
    ]
