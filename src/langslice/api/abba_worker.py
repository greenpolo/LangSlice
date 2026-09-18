"""JVM-free workers for snapshots exported by the separate Fiji connector.

The host owns image export, native registration actions and the ABBA/AP axis
mapping. This module consumes already calibrated snapshots and emits native-world
geometry; it never starts Java or imports abba_python.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

Emit = Callable[[dict[str, Any]], None]
_SUPPORTED_ATLASES = {"allen_mouse_10um", "allen_mouse_25um", "allen_mouse_50um"}


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
) -> list[dict[str, Any]]:
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
        if "reorder" in tasks and any(
            row.get(key) != old.get(key) for key in ("flip", "rotation_deg")
        ):
            update.update(
                flip=bool(row.get("flip")), rotation_deg=int(row.get("rotation_deg") or 0),
            )
        if "transform" in tasks and row.get("transform") != old.get("transform"):
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


def run_linear(params: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run exported snapshots; positions_mm are BrainGlobe AP millimetres.

    Required: image_folder, pixel_size_um, positions_mm (filename -> mm).
    Optional: spec (JobSpec fields), registered_slices (snapshot filenames).
    Checkpoint events carry initial=True with no mutations for ingestion.
    Later host_updates are replacement corrections, never cumulative transforms.
    """
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
    spec_data = dict(params.get("spec") or {})
    if spec_data.get("inputs"):
        raise ValueError("Snapshot inputs are supplied by the host, not agent settings")
    spec_data.update(image_folder=str(folder), resume=False, inputs={
        "positions": positions, "order": sorted(positions, key=lambda name: positions[name]),
        "pixel_size_um": calibration, "angles": {"pitch": 0.0, "yaw": 0.0},
    })
    spec_data.setdefault("debrief", False)
    spec = JobSpec.from_dict(spec_data)
    if spec.plane != "coronal" or spec.atlas not in _SUPPORTED_ATLASES:
        raise ValueError("The Fiji connector currently requires coronal Allen Mouse snapshots")
    if spec.transform.angles:
        raise ValueError("Cutting-angle updates are not verified for the Fiji connector")
    if not spec.tasks:
        raise ValueError("Select at least one LangSlice task")
    registered = params.get("registered_slices") or []
    if not isinstance(registered, list) or any(name not in geometry for name in registered):
        raise ValueError("Registered slice names must identify exported snapshots")
    if spec.has("reorder") and registered:
        raise ValueError("Turn off ordering/orientation for slices with existing registrations")
    previous: dict[str, Any] | None = None

    def checkpoint(state: Any) -> None:
        nonlocal previous
        current = state.to_dict()
        initial = previous is None
        updates = [] if previous is None else _host_updates(
            current, previous, spec.tasks, geometry, calibration,
        )
        emit({"kind": "checkpoint", "initial": initial, "state": current,
              "host_updates": updates})
        previous = current

    state = asyncio.run(engine.run(
        spec, on_write=checkpoint,
        emit=lambda message: emit({"kind": "log", "message": message}),
        on_event=lambda event: emit({"kind": "agent_event", "event": _json_event(event)}),
    ))
    # Observers are deliberately isolated by the engine. A final explicit
    # conversion ensures a failed native geometry export cannot report success.
    checkpoint(state)
    return {"state": state.to_dict(), "output_dir": str(folder)}


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
