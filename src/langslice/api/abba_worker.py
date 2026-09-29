"""JVM-free workers for snapshots exported by the separate Fiji connector.

The host owns image export, native registration actions and the ABBA/AP axis
mapping. This module consumes already calibrated snapshots and emits native-world
geometry; it never starts Java or imports abba_python.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.api.models import PreprocessPreviewRequest, PreprocessPreviewResult
    from langslice.linear.spec import JobSpec

Emit = Callable[[dict[str, Any]], None]
_SUPPORTED_ATLASES = {"allen_mouse_10um", "allen_mouse_25um", "allen_mouse_50um"}
#: Subfolder of the snapshot folder holding the blended images the agent is
#: shown, when the host sends preprocessing settings or multi-page snapshots.
AGENT_VIEW_FOLDER = "agent_view"


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _json_event(value: Any) -> Any:
    """Forward public event data, excluding media bytes and private reasoning."""
    if isinstance(value, dict):
        return {str(key): _json_event(item) for key, item in value.items()
                if key not in {"data", "thought_signature", "encrypted_content"}}
    if isinstance(value, (list, tuple)):
        return [_json_event(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _host_updates(
    current: dict[str, Any], previous: dict[str, Any], tasks: list[str],
    geometry: dict[str, tuple[int, int]], pixel_size_um: float,
    locked: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Host rows for what changed from *previous* to *current*.

    A locked section's in-plane geometry belongs to the host: no orientation
    or transform row is ever emitted for it.
    """
    from langslice.integrations.abba_affine import normalized_to_abba_affine
    from langslice.integrations.abba_spline import spline_world_landmarks

    prior = {row["id"]: row for row in previous["slices"]}
    updates = []
    for row in current["slices"]:
        name = row["id"]
        if name not in prior or name not in geometry:
            raise ValueError("Agent checkpoint contains an unknown snapshot")
        old = prior[name]
        update: dict[str, Any] = {"id": name}
        if {"position", "reorder"}.intersection(tasks) and (
            row.get("position_mm") != old.get("position_mm")
            or row.get("index_corrected") != old.get("index_corrected")
        ):
            position = float(row["position_mm"])
            if not math.isfinite(position):
                raise ValueError("Agent position must be finite")
            update["position_mm"] = position
        if "reorder" in tasks and name not in locked and any(
            row.get(key) != old.get(key) for key in ("flip", "rotation_deg")
        ):
            update.update(
                flip=bool(row.get("flip")), rotation_deg=int(row.get("rotation_deg") or 0),
            )
        if ("transform" in tasks and name not in locked
                and row.get("transform") != old.get("transform")):
            transform = row.get("transform")
            if transform is None:
                update["affine_mm"] = None
            elif transform.get("spline") is not None:
                source, target = spline_world_landmarks(
                    transform["spline"], size=geometry[name], pixel_size_um=pixel_size_um,
                    rotation_deg=int(row.get("rotation_deg") or 0),
                )
                update.update(spline_source_mm=source.tolist(), spline_target_mm=target.tolist())
            else:
                update["affine_mm"] = normalized_to_abba_affine(
                    transform["params"], size=geometry[name], pixel_size_um=pixel_size_um,
                    rotation_deg=int(row.get("rotation_deg") or 0),
                ).tolist()
        if len(update) > 1:
            updates.append(update)
    return updates


def _preprocessing(value: Any) -> dict[str, Any] | None:
    """The host's preprocessing settings, shape-checked, or None when absent."""
    if value is None:
        return None
    from langslice.api.models import PreprocessingSettings

    return PreprocessingSettings.model_validate(value).model_dump()


def _stage_agent_view(
    folder: Path, paths: list[str], settings: dict[str, Any] | None,
) -> Path | None:
    """Blend every snapshot into the one grayscale image the agent is shown.

    Snapshots may be multi-page TIFFs, one page per exported channel. When the
    host sends preprocessing settings, or any snapshot has several pages, each
    snapshot is blended by :func:`langslice.image_prep.host_preprocess` — the
    function ``preprocess.preview`` shows the user — and written under
    ``agent_view/`` with its own filename; the engine then shows those files
    with no further preprocessing. Otherwise (no settings, single pages) None:
    the run shows the snapshots through the engine's own automatic path, as
    before these settings existed.
    """

    from langslice.image_prep import (
        host_preprocess,
        host_preprocess_settings,
        page_count,
        read_pages,
    )

    counts = {path: page_count(path) for path in paths}
    if settings is None and all(count == 1 for count in counts.values()):
        return None
    # Validate every snapshot's settings before writing any file.
    for count in counts.values():
        host_preprocess_settings(settings, count)
    staged = folder / AGENT_VIEW_FOLDER
    staged.mkdir(exist_ok=True)
    for path in paths:
        blended = host_preprocess(read_pages(path), settings).convert("L")
        # Same filename: the filename is the section's identity.
        blended.save(staged / Path(path).name, format="TIFF")
    return staged


@dataclass
class PreparedLinear:
    """Validated host job and its native-world checkpoint translation."""

    folder: Path
    spec: JobSpec
    geometry: dict[str, tuple[int, int]]
    calibration: float
    locked: frozenset[str]
    final_updates: list[dict[str, Any]]
    trace_dir: Path | None


def prepare_linear(params: dict[str, Any]) -> PreparedLinear:
    """Validate and stage host snapshots identically for ADK and MCP."""
    from PIL import Image

    from langslice.linear import engine
    from langslice.linear.discovery import discover_slices
    from langslice.linear.spec import JobSpec

    folder = Path(params["image_folder"]).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Snapshot folder must be a directory")
    calibration = _finite_positive(params["pixel_size_um"], "Snapshot pixel size")
    paths = discover_slices(str(folder))
    geometry = {}
    for path in paths:
        with Image.open(path) as image:
            geometry[Path(path).name] = image.size
    angles = params.get("host_angles_deg") or {"pitch": 0.0, "yaw": 0.0}
    if not isinstance(angles, dict) or any(
        not math.isfinite(float(value)) or abs(float(value)) > 1e-8
        for value in angles.values()
    ):
        raise ValueError("The Fiji connector currently requires a flat atlas plane")
    positions = params["positions_mm"]
    if not geometry or not isinstance(positions, dict) or set(positions) != set(geometry):
        raise ValueError("Host positions must match every exported snapshot filename exactly")
    positions = {name: float(value) for name, value in positions.items()}
    # Newly imported ABBA sections can sit anterior to the atlas. Preserve that
    # host starting position; the positioning task can then move them into range.
    if not all(math.isfinite(value) for value in positions.values()):
        raise ValueError("Host positions must be finite BrainGlobe AP millimetres")
    locked = params.get("locked") or []
    if not isinstance(locked, list) or any(
        not isinstance(name, str) or name not in geometry for name in locked
    ):
        raise ValueError("Locked slice names must identify exported snapshots")
    damaged = params.get("damaged") or {}
    if not isinstance(damaged, dict) or any(
        name not in geometry or not isinstance(note, str) for name, note in damaged.items()
    ):
        raise ValueError("Damaged slices must map exported snapshot names to note text")
    preprocessing = _preprocessing(params.get("preprocessing"))
    spec_data = dict(params.get("spec") or {})
    if spec_data.get("inputs"):
        raise ValueError("Snapshot inputs are supplied by the host, not agent settings")
    inputs: dict[str, Any] = {
        "positions": positions, "order": sorted(positions, key=lambda name: positions[name]),
        "pixel_size_um": calibration, "angles": {"pitch": 0.0, "yaw": 0.0},
    }
    if locked:
        inputs["locked"] = sorted(set(locked))
    if damaged:
        inputs["damaged"] = dict(damaged)
    spec_data.update(image_folder=str(folder), resume=False, inputs=inputs)
    spec_data.setdefault("debrief", False)
    spec = JobSpec.from_dict(spec_data)
    if spec.plane != "coronal" or spec.atlas not in _SUPPORTED_ATLASES:
        raise ValueError("The Fiji connector currently requires coronal Allen Mouse snapshots")
    if spec.transform.angles:
        raise ValueError("Cutting-angle updates are not verified for the Fiji connector")
    if not spec.tasks:
        raise ValueError("Select at least one LangSlice task")
    # Blend last: every refusal above comes before any file is written.
    staged = _stage_agent_view(folder, paths, preprocessing)
    if staged is not None:
        # The agent is shown exactly the blended images; results stay beside
        # the snapshots.
        spec.image_folder = str(staged)
        spec.preprocess = "none"
        spec.out = spec.out or str(folder / engine.RESULTS_FILENAME)
    registered = params.get("registered_slices") or []
    if not isinstance(registered, list) or any(name not in geometry for name in registered):
        raise ValueError("Registered slice names must identify exported snapshots")
    return PreparedLinear(folder, spec, geometry, calibration, frozenset(locked), [],
                          _trace_dir(params.get("trace_dir")))


def checkpoint_callback(prepared: PreparedLinear, emit: Emit) -> Callable[[Any], None]:
    """Create a per-job delta tracker; the first checkpoint is ingestion."""
    previous: dict[str, Any] | None = None
    start: dict[str, Any] | None = None
    since_start: list[dict[str, Any]] = []

    def checkpoint(state: Any) -> None:
        nonlocal previous, start, since_start
        current = state.to_dict()
        initial = previous is None or start is None
        if previous is None or start is None:
            start = current
            updates: list[dict[str, Any]] = []
            since_start = []
        else:
            updates = _host_updates(
                current, previous, prepared.spec.tasks, prepared.geometry,
                prepared.calibration, prepared.locked,
            )
            since_start = _host_updates(
                current, start, prepared.spec.tasks, prepared.geometry,
                prepared.calibration, prepared.locked,
            )
        emit({"kind": "checkpoint", "initial": initial, "state": current,
              "host_updates": updates, "updates_since_start": since_start})
        previous = current
        prepared.final_updates = since_start

    return checkpoint


def run_linear(params: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run exported snapshots; positions_mm are BrainGlobe AP millimetres.

    Required: image_folder, pixel_size_um, positions_mm (filename -> mm).
    Optional: spec (JobSpec fields), registered_slices (snapshot filenames),
    locked (snapshot filenames whose in-plane geometry the agent may not
    change), damaged (filename -> note, flags the agent may not clear) and
    preprocessing (``preprocess.preview`` settings for multi-page snapshots)
    and trace_dir (save this run's full agent trace there; the result then
    names the files written).
    Checkpoint events carry initial=True with no mutations for ingestion.
    Later host_updates are replacement corrections relative to the previous
    checkpoint, never cumulative transforms; every checkpoint also carries
    updates_since_start (from the ingested state), and the result carries
    final_updates (ingested state -> final state), in the same row format.
    """
    from langslice.linear import engine

    prepared = prepare_linear(params)
    folder, spec = prepared.folder, prepared.spec
    checkpoint = checkpoint_callback(prepared, emit)
    trace_dir = prepared.trace_dir
    before = set(trace_dir.glob("*.jsonl")) if trace_dir is not None else set()
    with _tracing(trace_dir):
        state = asyncio.run(engine.run(
            spec, on_write=checkpoint,
            emit=lambda message: emit({"kind": "log", "message": message}),
            on_event=lambda event: emit({"kind": "agent_event", "event": _json_event(event)}),
        ))
    # Observers are deliberately isolated by the engine. A final explicit
    # conversion ensures a failed native geometry export cannot report success.
    checkpoint(state)
    result: dict[str, Any] = {"state": state.to_dict(), "output_dir": str(folder),
                              "final_updates": prepared.final_updates}
    if trace_dir is not None:
        result["trace_files"] = sorted(
            str(path) for path in set(trace_dir.glob("*.jsonl")) - before
        )
    return result


def _trace_dir(value: Any) -> Path | None:
    """The folder a host asked traces to be saved in, created; None when off."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("trace_dir must be a folder path")
    folder = Path(value).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@contextmanager
def _tracing(folder: Path | None) -> Iterator[None]:
    """Point the session trace at *folder* for one run, then restore."""
    import os

    from langslice.linear.trace import TRACE_DIR_ENV

    if folder is None:
        yield
        return
    previous = os.environ.get(TRACE_DIR_ENV)
    os.environ[TRACE_DIR_ENV] = str(folder)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TRACE_DIR_ENV, None)
        else:
            os.environ[TRACE_DIR_ENV] = previous


def preview_preprocess(request: PreprocessPreviewRequest) -> PreprocessPreviewResult:
    """Write the grayscale image the agent would be shown for one snapshot.

    The same :func:`langslice.image_prep.host_preprocess` ``linear.run``
    stages the agent's images with, before any agent-side resizing. No model,
    no atlas, no engine import.
    """
    from langslice.api.models import PreprocessPreviewResult
    from langslice.image_prep import host_preprocess, read_pages

    source = Path(request.image_path).expanduser().resolve(strict=True)
    output = Path(request.output_path).expanduser()
    if output.suffix.lower() != ".png":
        raise ValueError("Preview output must be a .png file")
    output.parent.mkdir(parents=True, exist_ok=True)
    image = host_preprocess(read_pages(source), request.preprocessing.model_dump()).convert("L")
    image.save(output, format="PNG")
    return PreprocessPreviewResult(
        output_path=str(output), width=image.width, height=image.height,
    )


def run_nonlinear(params: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Refine host placement; return atlas -> histology pairs in fixed-grid pixels."""
    import numpy as np
    import tifffile

    from langslice.integrations.abba import LangSliceAbbaConfig, compute_registration_landmarks

    coords_path = Path(params["coords_path"]).expanduser().resolve(strict=True)
    histology_path = Path(params["histology_path"]).expanduser().resolve(strict=True)
    if coords_path.suffix.lower() == ".npy":
        coords = np.load(coords_path, allow_pickle=False)
    else:
        coords = np.moveaxis(tifffile.imread(coords_path), 0, -1)
    histology = tifffile.imread(histology_path)
    if histology.ndim == 3 and histology.shape[0] == 1:
        histology = histology[0]
    if (coords.ndim != 3 or coords.shape[-1] != 3 or histology.ndim != 2
            or coords.shape[:2] != histology.shape or not np.isfinite(coords).all()
            or not np.isfinite(histology).all()):
        raise ValueError("Finite AP/DV/ML coordinates and grayscale histology must share one grid")
    config = LangSliceAbbaConfig(**(params.get("config") or {}))
    if config.atlas_name not in _SUPPORTED_ATLASES:
        raise ValueError("The Fiji connector currently requires the Allen Mouse atlas")
    _finite_positive(config.voxel_size_um, "Registration voxel size")
    if not isinstance(config.landmark_grid, int) or config.landmark_grid < 2:
        raise ValueError("Landmark grid must contain at least two points per axis")
    emit({"kind": "log", "message": "Refining ABBA's current atlas placement"})
    source, target = compute_registration_landmarks(coords, histology, config)
    if (source.ndim != 2 or source.shape[1] != 2 or len(source) < 3
            or target.shape != source.shape or not np.isfinite(source).all()
            or not np.isfinite(target).all()):
        raise ValueError("Registration returned invalid paired landmarks")
    return {"source_points": source.tolist(), "target_points": target.tolist(),
            "coordinate_frame": "fixed_grid_pixels"}
