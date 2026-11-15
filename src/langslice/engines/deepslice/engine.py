"""DeepSlice 1.2.8 run in LangSlice's process, on the images the plugin wrote.

DeepSlice and the TensorFlow it runs on come with the ``langslice[engines]``
extra and are imported when a run starts, never when LangSlice loads, so a
LangSlice without the extra imports this module and refuses the tool's call
(``ENGINE_MISSING``, ``require_engine``). TensorFlow runs on the CPU
(``hidden_gpus``), with ``TF_USE_LEGACY_KERAS=1`` (DeepSlice imports
``tensorflow.keras``, which TensorFlow 2.21 gives as ``tf_keras`` only with
it) and ``TF_CPP_MIN_LOG_LEVEL=2``, each set unless the environment sets it.
One run at a time in a process (``RUN_LOCK``), in a fresh Keras session
(``fresh_keras_session``: Keras keeps one global session, and a model built
again in it answers otherwise), with what DeepSlice prints caught for the
run's log.

DeepSlice downloads its weights on first use into its own package folder,
``DeepSlice/metadata/weights/`` (the ImageNet Xception start and the two
mouse networks of the ensemble, about 260 MB); a folder this process cannot
write, or a file there that is no weight file, is refused before the run.

The run, in the order DeepSlice's own pipeline takes it: the images are
staged (``staged_names``; hard links, else copies), then ``DSModel``,
``predict(staged, ensemble, section_numbers=bool(order))``, a copy of the
predictions (each section's own answer), ``set_bad_sections`` (only when
some are given, each by an anchored pattern on its staged name: DeepSlice
matches by regular-expression search), ``propagate_angles()``, with an
order ``enforce_index_spacing(section_thickness)``, and
``save_predictions``. Its files, in the run's folder: ``staged/``,
DeepSlice's own ``deepslice.json`` and ``deepslice.csv`` (staged names),
``final.json`` and ``own.json`` (QUINT files of the final and the own
anchorings, filenames the section ids) and ``deepslice_stdout.log``.

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
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Container, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from langslice.plugin_api import RefusalCode, ToolError

#: The extra's packages by import name: DeepSlice, the TensorFlow it runs
#: on and the Keras 2 its code is written for.
REQUIRED_MODULES = ("DeepSlice", "tensorflow", "tf_keras")
#: The command a missing package's refusal names.
INSTALL = 'pip install "langslice[engines]"'
#: The one species run: the mouse networks (the rat model is not offered).
SPECIES = "mouse"
#: DeepSlice 1.2.8's mouse weight files under its package's ``metadata``
#: folder (its ``config.json``): the ImageNet Xception it starts from and
#: the ensemble's two networks, each read on every run.
MOUSE_WEIGHTS = (
    "weights/xception_weights_tf_dim_ordering_tf_kernels.h5",
    "weights/Allen_Mixed_Best.h5",
    "weights/Synthetic_data_final.hdf5",
)
#: The first bytes of every HDF5 file: a weight file without them is not one.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
#: DeepSlice 1.2.8's mouse atlas (its ``config.json``): the QUINT files'
#: ``target`` and the voxel its spacing is counted in.
TARGET = "ABA_Mouse_CCFv3_2017_25um.cutlas"
VOXEL_UM = 25.0
#: DeepSlice 1.2.8's name for itself in a QUINT file (``aligner``).
ALIGNER = "prerelease_1.0.0"
#: The nine numbers of a QuickNII anchoring, as DeepSlice names its columns.
COLUMNS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
#: The suffixes DeepSlice reads an image by.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
#: What ``propagate_angles`` prints after each of its two passes.
ANGLES_PRINTED = re.compile(r"DV angle: (\S+)\s+ML angle: (\S+)")
#: One DeepSlice run at a time in this process (see the module text).
RUN_LOCK = threading.Lock()

#: A value of DeepSlice's prediction table, read row by row.
Cell = str | float | int | bool


class _DeviceConfig(Protocol):
    def set_visible_devices(
        self, devices: list[object], device_type: str
    ) -> None: ...


class _KerasBackend(Protocol):
    def clear_session(self) -> None: ...


class _Keras(Protocol):
    @property
    def backend(self) -> _KerasBackend: ...


class _TensorFlow(Protocol):
    @property
    def config(self) -> _DeviceConfig: ...


class _Column(Protocol):
    @property
    def values(self) -> object: ...


class _Table(Protocol):
    @property
    def columns(self) -> Container[str]: ...

    def copy(self) -> _Table: ...

    def to_dict(self, orient: str) -> list[dict[str, Cell]]: ...

    def __getitem__(self, key: str) -> _Column: ...


class _Model(Protocol):
    predictions: _Table

    def predict(
        self, *, image_directory: str, ensemble: bool, section_numbers: bool
    ) -> None: ...

    def set_bad_sections(self, bad_sections: list[str]) -> None: ...

    def propagate_angles(self) -> None: ...

    def enforce_index_spacing(
        self, *, section_thickness: float | None
    ) -> None: ...

    def save_predictions(self, filename: str) -> None: ...


@dataclass(frozen=True)
class _DeepSlice:
    """DeepSlice's entry points this run calls, typed."""

    model: Callable[[str], _Model]
    ordered: Callable[[_Table], _Table]
    depths: Callable[[_Table], Sequence[float]]
    direction: Callable[[npt.NDArray[np.float64]], str]
    thickness: Callable[..., float]


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
    out_folder
        The folder the run writes its files in.
    """

    images: tuple[tuple[str, Path], ...]
    order: tuple[str, ...]
    bad_sections: tuple[str, ...]
    interval_um: float | None
    ensemble: bool
    out_folder: Path


@dataclass(frozen=True)
class SectionAnswer:
    """Hold DeepSlice's answer for one section.

    Parameters
    ----------
    staged
        The name the image was staged under.
    nr
        The section number DeepSlice gave it.
    bad
        Whether it was a bad section.
    width, height
        The staged image's size in pixels.
    anchoring
        The QuickNII anchoring after the whole pipeline, nine numbers.
    own_anchoring
        The anchoring right after ``predict``, before the batch steps.
    """

    staged: str
    nr: int | None
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
        DeepSlice's version.
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
        The (DV, ML) angles of the two passes of ``propagate_angles``.
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


def require_engine() -> None:
    """Refuse a call DeepSlice cannot run here, before anything runs.

    Raises
    ------
    ToolError
        ``ENGINE_MISSING``: a package of the extra is not installed, the
        weights folder cannot be written while a weight file is missing, or
        a weight file is not one.
    """
    missing = [
        name
        for name in REQUIRED_MODULES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"DeepSlice cannot run here: {', '.join(missing)} not installed "
            f"(install LangSlice's engines extra: {INSTALL})",
        )

    folder = _metadata_folder()
    for name in MOUSE_WEIGHTS:
        path = folder / name
        if not path.exists() and not _is_writable(path.parent):
            raise ToolError(
                RefusalCode.ENGINE_MISSING,
                f"DeepSlice cannot run here: it downloads its mouse weights "
                f"on first use into {path.parent}, which this process "
                "cannot write",
            )
        if path.exists() and not _is_weight_file(path):
            raise ToolError(
                RefusalCode.ENGINE_MISSING,
                f"DeepSlice cannot run here: {path} is not a weight file "
                "(an interrupted download); delete it and DeepSlice "
                "downloads it again",
            )


def _metadata_folder() -> Path:
    """Return DeepSlice's ``metadata`` folder, without importing it."""
    spec = importlib.util.find_spec("DeepSlice")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))) / "metadata"


def _is_writable(folder: Path) -> bool:
    """Tell whether this process can make a file in a folder."""
    try:
        with tempfile.TemporaryFile(dir=folder):
            return True
    except OSError:
        return False


def _is_weight_file(path: Path) -> bool:
    """Tell whether a file starts as an HDF5 file does."""
    with path.open("rb") as handle:
        return handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE


def staged_names(ids: Sequence[str], order: Sequence[str]) -> dict[str, str]:
    """Name each section's staged image.

    With an order, ``section_sNNN.png``, NNN the 1-based rank in it
    (DeepSlice reads exactly one ``_s<digits>`` per name, and a section id
    may carry its own); without, the section id, ``.png`` appended unless
    it ends in an image suffix.

    Parameters
    ----------
    ids
        The section ids of the run.
    order
        The cutting order; empty for none.

    Returns
    -------
    dict
        The staged name by section id.

    Raises
    ------
    ToolError
        ``BAD_ARGS`` when an id holds a path separator or two ids would
        share a name.
    """
    if order:
        width = max(3, len(str(len(order))))
        return {
            section: f"section_s{rank + 1:0{width}d}.png"
            for rank, section in enumerate(order)
        }

    names: dict[str, str] = {}
    for section in ids:
        if "/" in section or "\\" in section:
            raise ToolError(
                RefusalCode.BAD_ARGS,
                f"DeepSlice: section id {section!r} holds a path separator",
            )
        has_suffix = Path(section).suffix.lower() in IMAGE_SUFFIXES
        names[section] = section if has_suffix else f"{section}.png"
    if len(set(names.values())) != len(names):
        raise ToolError(
            RefusalCode.BAD_ARGS,
            "DeepSlice: two section ids would share a staged name",
        )
    return names


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
        ``BAD_ARGS`` for a request DeepSlice cannot take (the staging);
        ``FIT_FAILED`` for anything DeepSlice or TensorFlow raised (its
        traceback in the run's ``deepslice_error.log``).
    """
    started = time.monotonic()
    request.out_folder.mkdir(parents=True, exist_ok=True)
    names = _stage(request)
    log = io.StringIO()
    with RUN_LOCK, contextlib.redirect_stdout(log):
        try:
            return _pipeline(request, names, log, started)
        except Exception as error:  # noqa: BLE001 - anything DeepSlice or TensorFlow raises fails the call
            (request.out_folder / "deepslice_error.log").write_text(
                traceback.format_exc(), encoding="utf-8", newline="\n"
            )
            raise ToolError(
                RefusalCode.FIT_FAILED,
                f"DeepSlice: {type(error).__name__}: {error}",
            ) from error
        finally:
            if "tf_keras" in sys.modules:  # the run's model is let go
                fresh_keras_session()
            (request.out_folder / "deepslice_stdout.log").write_text(
                log.getvalue(), encoding="utf-8", newline="\n"
            )


def _stage(request: EngineRequest) -> dict[str, str]:
    """Stage the request's images under DeepSlice's names; return them."""
    ids = [section for section, _ in request.images]
    names = staged_names(ids, request.order)
    folder = request.out_folder / "staged"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        old.unlink()

    for section, image_path in request.images:
        target = folder / names[section]
        try:
            os.link(image_path, target)
        except OSError:  # another volume, or a file system without links
            shutil.copyfile(image_path, target)
    return names


def _pipeline(
    request: EngineRequest,
    names: dict[str, str],
    log: io.StringIO,
    started: float,
) -> EngineRun:
    """Call DeepSlice in its own pipeline's order (the module text)."""
    timings: dict[str, float] = {}
    deepslice = _load_deepslice()
    fresh_keras_session()
    timings["import_s"] = time.monotonic() - started

    step = time.monotonic()
    model = deepslice.model(SPECIES)
    timings["load_s"] = time.monotonic() - step

    step = time.monotonic()
    model.predict(
        image_directory=str(request.out_folder / "staged"),
        ensemble=request.ensemble,
        section_numbers=bool(request.order),
    )
    timings["predict_s"] = time.monotonic() - step
    own = model.predictions.copy()

    if request.bad_sections:
        step = time.monotonic()
        model.set_bad_sections(
            [
                r"(^|/)" + re.escape(names[section]) + "$"
                for section in request.bad_sections
            ]
        )
        timings["bad_sections_s"] = time.monotonic() - step

    step = time.monotonic()
    mark = log.tell()
    model.propagate_angles()
    timings["propagate_s"] = time.monotonic() - step
    # Each match is the pattern's two groups, the DV and ML angles.
    printed = cast(
        "list[tuple[str, str]]",
        ANGLES_PRINTED.findall(log.getvalue()[mark:]),
    )
    propagation = tuple((float(dv), float(ml)) for dv, ml in printed)

    spacing = None
    if request.order:
        step = time.monotonic()
        spacing = _spaced(deepslice, model, request.interval_um)
        timings["spacing_s"] = time.monotonic() - step

    step = time.monotonic()
    model.save_predictions(str(request.out_folder / "deepslice"))
    to_id = {name: section for section, name in names.items()}
    final = model.predictions.to_dict("records")
    own_rows = own.to_dict("records")
    final_json = request.out_folder / "final.json"
    own_json = request.out_folder / "own.json"
    _write_quint(final, final_json, to_id)
    _write_quint(own_rows, own_json, to_id)
    timings["save_s"] = time.monotonic() - step

    sections = _answers(final, own_rows, to_id)
    timings["total_s"] = time.monotonic() - started
    return EngineRun(
        sections=sections,
        final_json=final_json,
        own_json=own_json,
        propagation=propagation,
        spacing=spacing,
        versions=_versions(),
        timings_s=timings,
    )


def _load_deepslice() -> _DeepSlice:
    """Import TensorFlow (on the CPU) and DeepSlice; return its calls."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    hidden_gpus()
    package = importlib.import_module("DeepSlice")
    spacing = importlib.import_module(
        "DeepSlice.coord_post_processing.spacing_and_indexing"
    )
    depth = importlib.import_module(
        "DeepSlice.coord_post_processing.depth_estimation"
    )
    # DeepSlice has no type information: its entry points are typed here.
    return _DeepSlice(
        model=cast("Callable[[str], _Model]", package.DSModel),
        ordered=cast(
            "Callable[[_Table], _Table]", spacing.enforce_section_ordering
        ),
        depths=cast(
            "Callable[[_Table], Sequence[float]]",
            depth.calculate_brain_center_depths,
        ),
        direction=cast(
            "Callable[[npt.NDArray[np.float64]], str]",
            spacing.determine_direction_of_indexing,
        ),
        thickness=cast(
            "Callable[..., float]",
            spacing.calculate_average_section_thickness,
        ),
    )


def fresh_keras_session() -> None:
    """Clear Keras's global session, so the next model is built anew.

    Keras keeps every model built in a process in one session and names a
    rebuilt model's layers apart from the earlier ones; DeepSlice loads its
    weights by layer name, so a second ``DSModel`` in a process, built
    without this, answers otherwise than the first (by up to 32 atlas
    voxels on a test stack) and raises nothing.
    """
    # tf_keras has no type information here: typed by hand.
    keras = cast("_Keras", importlib.import_module("tf_keras"))
    keras.backend.clear_session()


def hidden_gpus() -> None:
    """Import TensorFlow with every GPU hidden from it.

    TensorFlow takes a GPU's whole memory when it sees one, which the fits
    on torch in the same process need; DeepSlice runs on the CPU. Once
    TensorFlow has set up its devices they cannot change: then they were
    set by an earlier run, which hid them.
    """
    # TensorFlow's module has no type information here: typed by hand.
    tensorflow = cast("_TensorFlow", importlib.import_module("tensorflow"))
    with contextlib.suppress(RuntimeError):
        tensorflow.config.set_visible_devices([], "GPU")


def _spaced(
    deepslice: _DeepSlice, model: _Model, interval_um: float | None
) -> Spacing:
    """Space the sections along the index; say what DeepSlice used."""
    # DeepSlice's own reading of the batch, on a copy, for the reply
    probe = deepslice.ordered(model.predictions.copy())
    depths = np.array(deepslice.depths(probe), dtype=float)
    direction = deepslice.direction(depths)
    if interval_um is None:
        flags = (
            probe["bad_section"].values
            if "bad_section" in probe.columns
            else None
        )
        used = VOXEL_UM * deepslice.thickness(
            probe["nr"],
            section_depth=depths,
            bad_sections=flags,
            species=SPECIES,
        )
    else:
        used = float(interval_um)

    model.enforce_index_spacing(section_thickness=interval_um)
    return Spacing(
        interval_um_given=interval_um,
        section_thickness_um_used=used,
        estimated=interval_um is None,
        voxel_um=VOXEL_UM,
        direction=direction,
    )


def _file_name(row: dict[str, Cell]) -> str:
    """Return the name of a table row's image, without its folder."""
    return Path(str(row["Filenames"])).name


def _section_number(row: dict[str, Cell]) -> int | None:
    """Return a table row's section number, None when it has none."""
    return int(row["nr"]) if "nr" in row else None


def _anchoring(row: dict[str, Cell]) -> tuple[float, ...]:
    """Return a table row's nine anchoring numbers."""
    return tuple(float(row[column]) for column in COLUMNS)


def _write_quint(
    rows: list[dict[str, Cell]], path: Path, to_id: dict[str, str]
) -> None:
    """Write a QUINT file of table rows, filenames the section ids."""
    slices = [
        {
            "filename": to_id[_file_name(row)],
            "anchoring": list(_anchoring(row)),
            "height": int(row["height"]),
            "width": int(row["width"]),
            "nr": _section_number(row),
            "markers": [],
        }
        for row in rows
    ]
    document = {"name": "", "target": TARGET, "aligner": ALIGNER}
    path.write_text(
        json.dumps({**document, "slices": slices}),
        encoding="utf-8",
        newline="\n",
    )


def _answers(
    final: list[dict[str, Cell]],
    own: list[dict[str, Cell]],
    to_id: dict[str, str],
) -> dict[str, SectionAnswer]:
    """Return each section's answer, in the final table's order."""
    own_by_name = {_file_name(row): _anchoring(row) for row in own}
    answers: dict[str, SectionAnswer] = {}
    for row in final:
        name = _file_name(row)
        answers[to_id[name]] = SectionAnswer(
            staged=name,
            nr=_section_number(row),
            bad=bool(row.get("bad_section", False)),
            width=int(row["width"]),
            height=int(row["height"]),
            anchoring=_anchoring(row),
            own_anchoring=own_by_name[name],
        )
    if set(answers) != set(to_id.values()):
        raise RuntimeError(
            f"DeepSlice returned {len(answers)} of {len(to_id)} sections"
        )
    return answers


def _versions() -> Versions:
    """Return the versions a run used."""
    return Versions(
        deepslice=importlib.metadata.version("DeepSlice"),
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
