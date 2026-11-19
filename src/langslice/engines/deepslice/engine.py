"""One DeepSlice run in LangSlice's process, on the images the plugin wrote.

DeepSlice 1.2.8's inference is vendored beside this module (``network.py``,
``alignment.py``; MIT, ``LICENSE``). TensorFlow and ``tf_keras`` come with
the ``langslice[engines]`` extra and are imported when a run starts, never
when LangSlice loads, so a LangSlice without the extra imports this module
and refuses the tool's call (``ENGINE_MISSING``, ``require_engine``).
TensorFlow runs on the CPU (``hidden_gpus``), with
``TF_USE_LEGACY_KERAS=1`` and ``TF_CPP_MIN_LOG_LEVEL=2``, each set unless
the environment sets it. One run at a time in a process (``RUN_LOCK``), in
a fresh Keras session (``fresh_keras_session``), with what Keras prints
caught in the run's ``deepslice_stdout.log``.

The run, in the order of DeepSlice's own pipeline: each image's anchoring
(``DSModel.predict``), the table sorted by cutting index when an order is
given and by predicted depth otherwise, a copy of it (each section's own
answer), the angle vote over the good sections, applied to every section
(``propagate_angles``), and with an order the spacing along it
(``enforce_index_spacing``). Its files, in the run's folder: ``final.json``
and ``own.json``, QUINT files of the final and the own anchorings with the
section ids as filenames (``nr``: the index, or with no order each
section's place in the final table, as DeepSlice's ``save_predictions``
numbers it).

``interval_um`` is DeepSlice's ``section_thickness`` in micrometres,
signed: positive when the predicted depth (QuickNII y) grows with the
index; None lets DeepSlice estimate it (weighted, bad sections left out).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from langslice.engines.deepslice import alignment, network
from langslice.engines.deepslice.network import Weights
from langslice.plugin_api import RefusalCode, ToolError

REQUIRED_MODULES = ("tensorflow", "tf_keras", "h5py")
INSTALL = 'pip install "langslice[engines]"'
TARGET = "ABA_Mouse_CCFv3_2017_25um.cutlas"
ALIGNER = "prerelease_1.0.0"
VERSION = "1.2.8 (b0d6965)"
RUN_LOCK = threading.Lock()


class _KerasBackend(Protocol):
    def clear_session(self) -> None: ...


class _Keras(Protocol):
    @property
    def backend(self) -> _KerasBackend: ...


class _DeviceConfig(Protocol):
    def set_visible_devices(self, devices: list[object], device_type: str) -> None: ...


class _TensorFlow(Protocol):
    @property
    def config(self) -> _DeviceConfig: ...


@dataclass(frozen=True)
class EngineRequest:
    """Ask one DeepSlice run.

    Parameters
    ----------
    images
        Each section's id and the image written for it, as the agent
        oriented the section.
    order
        The cutting order, anterior to posterior, listing every image once;
        empty for no index step.
    bad_sections
        Ids left out of DeepSlice's angle vote and spacing fit.
    interval_um
        DeepSlice's signed ``section_thickness`` (module text); None for its
        own estimate. Given only with ``order``.
    ensemble
        Whether DeepSlice averages its two mouse networks.
    weights
        DeepSlice's weight files.
    out_folder
        The folder the run writes its files in.
    """

    images: tuple[tuple[str, Path], ...]
    order: tuple[str, ...]
    bad_sections: tuple[str, ...]
    interval_um: float | None
    ensemble: bool
    weights: Weights
    out_folder: Path


@dataclass(frozen=True)
class SectionAnswer:
    """Hold DeepSlice's answer for one section.

    Parameters
    ----------
    nr
        The section's number in the QUINT files.
    bad
        Whether it was a bad section.
    width, height
        Its image's size in pixels.
    anchoring
        The QuickNII anchoring after the whole pipeline, nine numbers.
    own_anchoring
        The anchoring right after the prediction, before the batch steps.
    """

    nr: int
    bad: bool
    width: int
    height: int
    anchoring: tuple[float, ...]
    own_anchoring: tuple[float, ...]


@dataclass(frozen=True)
class Spacing:
    """Hold the spacing DeepSlice used along the cutting order.

    Parameters
    ----------
    interval_um_given
        The signed interval it was given; None when it estimated one.
    section_thickness_um_used
        The signed interval it spaced by.
    estimated
        Whether DeepSlice estimated it.
    voxel_um
        The atlas voxel DeepSlice counts in.
    direction
        DeepSlice's own reading of the index-ordered depths,
        ``rostro-caudal`` or ``caudal-rostro``.
    """

    interval_um_given: float | None
    section_thickness_um_used: float
    estimated: bool
    voxel_um: float
    direction: str


@dataclass(frozen=True)
class Versions:
    """Hold the versions a DeepSlice run used.

    Parameters
    ----------
    deepslice
        The DeepSlice vendored, version and commit.
    tensorflow
        TensorFlow's version.
    tf_use_legacy_keras
        ``TF_USE_LEGACY_KERAS`` as the run had it.
    """

    deepslice: str
    tensorflow: str
    tf_use_legacy_keras: str


@dataclass(frozen=True)
class EngineRun:
    """Hold what one DeepSlice run answered.

    Parameters
    ----------
    sections
        Each section's answer by id, in DeepSlice's final order (by cutting
        index with an order, else by predicted depth).
    final_json, own_json
        The QUINT files of the final and the own anchorings.
    propagation
        The (DV, ML) angles of the two passes of the angle vote.
    spacing
        The spacing used with an order; None without.
    versions
        The versions DeepSlice ran at.
    timings_s
        Seconds per step.
    """

    sections: dict[str, SectionAnswer]
    final_json: Path
    own_json: Path
    propagation: tuple[tuple[float, float], ...]
    spacing: Spacing | None
    versions: Versions
    timings_s: dict[str, float]


@dataclass(frozen=True)
class _Table:
    """The sections in DeepSlice's table order, with their anchorings."""

    ids: list[str]
    anchorings: npt.NDArray[np.float64]
    numbers: npt.NDArray[np.int64]
    bad: npt.NDArray[np.bool_]
    widths: list[int]
    heights: list[int]


def require_engine() -> None:
    """Refuse a call DeepSlice cannot run here, before anything runs.

    Raises
    ------
    ToolError
        ``ENGINE_MISSING`` when a package of the extra is not installed.
    """
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"DeepSlice cannot run here: {', '.join(missing)} not installed "
            f"(install LangSlice's engines extra: {INSTALL})",
        )


def fetched_weights(cache_folder: Path) -> Weights:
    """Return DeepSlice's weight files, fetching them on first use.

    Parameters
    ----------
    cache_folder
        The folder that keeps them.

    Returns
    -------
    Weights
        The three files, each checked by its SHA-256.

    Raises
    ------
    ToolError
        ``ENGINE_MISSING`` when a file cannot be fetched or is not
        DeepSlice's.
    """
    try:
        return network.fetch_weights(cache_folder)
    except (OSError, ValueError) as error:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"DeepSlice cannot run here: its weights could not be fetched "
            f"into {cache_folder.expanduser()} ({error})",
        ) from error


def run_deepslice(request: EngineRequest) -> EngineRun:
    """Run DeepSlice on a request, in this process.

    Parameters
    ----------
    request
        The images, order, bad sections and interval, and the run's folder.

    Returns
    -------
    EngineRun
        Each section's answer, the files, the angles and spacing used.

    Raises
    ------
    ToolError
        ``FIT_FAILED`` for anything DeepSlice or TensorFlow raised (its
        traceback in the run's ``deepslice_error.log``).
    """
    started = time.monotonic()
    request.out_folder.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    with RUN_LOCK, contextlib.redirect_stdout(log):
        try:
            return _pipeline(request, started)
        except Exception as error:
            (request.out_folder / "deepslice_error.log").write_text(
                traceback.format_exc(), encoding="utf-8", newline="\n"
            )
            raise ToolError(
                RefusalCode.FIT_FAILED,
                f"DeepSlice: {type(error).__name__}: {error}",
            ) from error
        finally:
            if "tf_keras" in sys.modules:
                fresh_keras_session()
            (request.out_folder / "deepslice_stdout.log").write_text(
                log.getvalue(), encoding="utf-8", newline="\n"
            )


def _pipeline(request: EngineRequest, started: float) -> EngineRun:
    """Run DeepSlice's steps in its own pipeline's order (module text)."""
    timings: dict[str, float] = {}
    hidden_gpus()
    fresh_keras_session()
    timings["import_s"] = time.monotonic() - started

    step = time.monotonic()
    prediction = network.predict(
        [path for _, path in request.images],
        request.weights,
        ensemble=request.ensemble,
    )
    table = _sorted_table(request, prediction)
    own = table.anchorings.copy()
    timings["predict_s"] = time.monotonic() - step

    step = time.monotonic()
    anchorings, votes = alignment.propagated(table.anchorings, table.bad)
    timings["propagate_s"] = time.monotonic() - step

    spacing = None
    if request.order:
        step = time.monotonic()
        anchorings, spacing = _spaced(request, table, anchorings)
        timings["spacing_s"] = time.monotonic() - step

    numbers = table.numbers
    if not request.order:
        numbers = np.arange(1, len(table.ids) + 1)
    final = _Table(table.ids, anchorings, numbers, table.bad, table.widths, table.heights)
    final_json = request.out_folder / "final.json"
    own_json = request.out_folder / "own.json"
    _write_quint(final, final_json, numbered=True)
    _write_quint(_with_anchorings(table, own), own_json, numbered=bool(request.order))

    timings["total_s"] = time.monotonic() - started
    return EngineRun(
        sections=_answers(final, own),
        final_json=final_json,
        own_json=own_json,
        propagation=tuple(votes),
        spacing=spacing,
        versions=_versions(),
        timings_s=timings,
    )


def fresh_keras_session() -> None:
    """Clear Keras's global session, so the next model is built anew.

    Keras keeps every model built in a process in one session and names a
    rebuilt model's layers apart from the earlier ones; DeepSlice loads its
    weights by layer name, so a second model in a process, built without
    this, answers otherwise than the first (by up to 32 atlas voxels on a
    test stack) and raises nothing.
    """
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    keras = cast("_Keras", importlib.import_module("tf_keras"))
    keras.backend.clear_session()


def hidden_gpus() -> None:
    """Import TensorFlow with every GPU hidden from it.

    TensorFlow takes a GPU's whole memory when it sees one, which the fits
    on torch in the same process need; DeepSlice runs on the CPU. Once
    TensorFlow has set up its devices they cannot change: then they were
    set by an earlier run, which hid them.
    """
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tensorflow = cast("_TensorFlow", importlib.import_module("tensorflow"))
    with contextlib.suppress(RuntimeError):
        tensorflow.config.set_visible_devices([], "GPU")


def _sorted_table(request: EngineRequest, prediction: network.Prediction) -> _Table:
    """Return the predictions as DeepSlice's table: by index, else depth."""
    ids = [section for section, _ in request.images]
    rank = {section: index + 1 for index, section in enumerate(request.order)}
    numbers = np.array([rank.get(section, 0) for section in ids], np.int64)
    keys = numbers if request.order else prediction.anchorings[:, 1]
    rows = cast("list[int]", np.argsort(keys, kind="stable").tolist())
    bad = set(request.bad_sections)
    return _Table(
        ids=[ids[row] for row in rows],
        anchorings=prediction.anchorings[rows],
        numbers=numbers[rows],
        bad=np.array([ids[row] in bad for row in rows], np.bool_),
        widths=[prediction.widths[row] for row in rows],
        heights=[prediction.heights[row] for row in rows],
    )


def _spaced(
    request: EngineRequest,
    table: _Table,
    anchorings: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], Spacing]:
    """Space the sections along the index; say what DeepSlice used."""
    spaced, used = alignment.spaced(
        anchorings, table.numbers, table.bad, thickness_um=request.interval_um
    )
    thickness_um = request.interval_um
    if thickness_um is None:
        thickness_um = used.thickness_voxels * alignment.VOXEL_UM
    spacing = Spacing(
        interval_um_given=request.interval_um,
        section_thickness_um_used=float(thickness_um),
        estimated=used.estimated,
        voxel_um=alignment.VOXEL_UM,
        direction=used.direction,
    )
    return spaced, spacing


def _with_anchorings(table: _Table, anchorings: npt.NDArray[np.float64]) -> _Table:
    """Return a table with other anchorings, in the same order."""
    return _Table(
        table.ids,
        anchorings,
        table.numbers,
        table.bad,
        table.widths,
        table.heights,
    )


def _write_quint(table: _Table, path: Path, *, numbered: bool) -> None:
    """Write a QUINT file of a table, filenames the section ids."""
    numbers = _ints(table.numbers)
    anchorings = _anchorings(table.anchorings)
    slices: list[dict[str, object]] = []
    for row, section in enumerate(table.ids):
        number = numbers[row] if numbered else None
        slices.append(
            {
                "filename": section,
                "anchoring": list(anchorings[row]),
                "height": table.heights[row],
                "width": table.widths[row],
                "nr": number,
                "markers": [],
            }
        )
    document = {"name": "", "target": TARGET, "aligner": ALIGNER}
    path.write_text(
        json.dumps({**document, "slices": slices}),
        encoding="utf-8",
        newline="\n",
    )


def _answers(final: _Table, own: npt.NDArray[np.float64]) -> dict[str, SectionAnswer]:
    """Return each section's answer, in the final table's order."""
    numbers = _ints(final.numbers)
    flags = cast("list[bool]", final.bad.tolist())
    anchorings = _anchorings(final.anchorings)
    own_anchorings = _anchorings(own)
    return {
        section: SectionAnswer(
            nr=numbers[row],
            bad=flags[row],
            width=final.widths[row],
            height=final.heights[row],
            anchoring=anchorings[row],
            own_anchoring=own_anchorings[row],
        )
        for row, section in enumerate(final.ids)
    }


def _ints(values: npt.NDArray[np.int64]) -> list[int]:
    """Return an array's elements as Python ints (numpy leaves them bare)."""
    return cast("list[int]", values.tolist())


def _anchorings(values: npt.NDArray[np.float64]) -> list[tuple[float, ...]]:
    """Return an array's rows as tuples of Python floats."""
    rows = cast("list[list[float]]", values.tolist())
    return [tuple(row) for row in rows]


def _versions() -> Versions:
    """Return the versions a run used."""
    return Versions(
        deepslice=VERSION,
        tensorflow=importlib.metadata.version("tensorflow"),
        tf_use_legacy_keras=os.environ.get("TF_USE_LEGACY_KERAS", ""),
    )


def save_run(request: EngineRequest, run: EngineRun, path: Path) -> None:
    """Write a run's request and answer as one JSON file.

    Parameters
    ----------
    request
        What the run was asked.
    run
        What it answered.
    path
        The file to write (the call's ``out.json``).
    """
    text = json.dumps(
        {"request": asdict(request), "run": asdict(run)},
        indent=1,
        default=str,
    )
    path.write_text(text, encoding="utf-8", newline="\n")
