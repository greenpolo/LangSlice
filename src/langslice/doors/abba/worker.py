"""JVM-free workers for snapshots exported by the separate Fiji connector.

The host owns image export, native registration actions and the ABBA/AP axis
mapping. This module consumes already calibrated snapshots and emits native-world
geometry; it never starts Java or imports abba_python.
"""
from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.core.spec import JobSpec
    from langslice.doors.abba.contract import (
        PreprocessPreviewRequest,
        PreprocessPreviewResult,
    )

from langslice.doors.abba.checkpoints import checkpoint_callback

Emit = Callable[[dict[str, Any]], None]
#: The keys of a ``linear.run`` request (``mcp.prepare`` adds ``host_channel``);
#: any other is refused.
REQUEST_KEYS = ("image_folder", "pixel_size_um", "positions_mm", "z_offset_mm", "angles_deg",
                "existing_warp", "damaged", "channel_names", "spec", "preprocessing")


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def public_event(value: Any) -> Any:
    """Forward public event data, excluding media bytes and private reasoning."""
    if isinstance(value, dict):
        return {str(key): public_event(item) for key, item in value.items()
                if key not in {"data", "thought_signature", "encrypted_content"}}
    if isinstance(value, (list, tuple)):
        return [public_event(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _preprocessing(value: Any) -> dict[str, Any] | None:
    """The host's preprocessing settings, shape-checked, or None when absent."""
    if value is None:
        return None
    from langslice.doors.abba.contract import PreprocessingSettings

    return PreprocessingSettings.model_validate(value).model_dump()


def _host_appearance(
    paths: list[str], settings: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The default appearance a host's snapshots are shown in, validated.

    Snapshots may be multi-page TIFFs, one page per exported channel. When the
    host sends preprocessing settings, or any snapshot has several pages, the
    run shows each section as
    :func:`langslice.core.sections.reading.host_preprocess` blends it — the
    function ``preprocess.preview`` shows the user — but the blend is the job's
    opening ``preprocessed`` image (``JobSpec.host_preprocessing``): the
    snapshots stay the run's images and their pages stay readable as raw
    channels, for the agent's ``look`` and ``make_image``. Otherwise (no
    settings, single pages) None: the engine's own automatic path.
    """
    from langslice.core.sections import reading

    counts = {path: reading.page_count(path) for path in paths}
    if settings is None and all(count == 1 for count in counts.values()):
        return None
    # Validate every snapshot's settings before the run starts.
    for count in counts.values():
        reading.host_preprocess_settings(settings, count)
    return dict(settings or {"mode": "auto"})


def _channel_names(value: Any) -> list[str] | None:
    """Optional host names for the exported pages, one per page."""
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        raise ValueError("channel_names must be a list of names, one per exported page")
    cleaned = [name.strip() or f"ch{index + 1}" for index, name in enumerate(value)]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("channel_names must be distinct")
    return cleaned


@dataclass
class PreparedLinear:
    """Validated host job and its native-world checkpoint translation."""

    folder: Path
    spec: JobSpec
    geometry: dict[str, tuple[int, int]]
    calibration: float
    #: The ABBA session's facts the job keeps for reproducibility (stored in
    #: ``job.json`` under ``host.abba``): ``angles_deg``, ``z_offset_mm`` (the
    #: connector converts positions with it; the worker's positions are
    #: BrainGlobe AP mm), ``existing_warp``.
    abba: dict[str, Any] = field(default_factory=dict)


def prepare_linear(params: dict[str, Any]) -> PreparedLinear:
    """Validate host snapshots and build their job identically for ADK and MCP.

    ``ValueError`` naming any key not in :data:`REQUEST_KEYS`."""
    from PIL import Image

    from langslice.core.sections.discovery import discover_slices
    from langslice.core.spec import JobSpec

    unknown = sorted(set(params) - set(REQUEST_KEYS))
    if unknown:
        raise ValueError(f"Unknown request key(s) {unknown}")
    folder = Path(params["image_folder"]).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Snapshot folder must be a directory")
    calibration = _finite_positive(params["pixel_size_um"], "Snapshot pixel size")
    paths = discover_slices(str(folder))
    geometry = {}
    for path in paths:
        with Image.open(path) as image:
            geometry[Path(path).name] = image.size
    pitch, yaw = _host_angles(params)
    z_offset = params.get("z_offset_mm")
    if z_offset is not None:
        z_offset = float(z_offset)
        if not math.isfinite(z_offset):
            raise ValueError("z_offset_mm must be finite")
    positions = params["positions_mm"]
    if not geometry or not isinstance(positions, dict) or set(positions) != set(geometry):
        raise ValueError("Host positions must match every exported snapshot filename exactly")
    positions = {name: float(value) for name, value in positions.items()}
    # Newly imported ABBA sections can sit anterior to the atlas. Preserve that
    # host starting position; the positioning task can then move them into range.
    if not all(math.isfinite(value) for value in positions.values()):
        raise ValueError("Host positions must be finite BrainGlobe AP millimetres")
    damaged = params.get("damaged") or {}
    if not isinstance(damaged, dict) or any(
        name not in geometry or not isinstance(note, str) for name, note in damaged.items()
    ):
        raise ValueError("Damaged slices must map exported snapshot names to note text")
    existing_warp = params.get("existing_warp") or []
    if not isinstance(existing_warp, list) or any(
        not isinstance(name, str) or name not in geometry for name in existing_warp
    ):
        raise ValueError("existing_warp names must identify exported snapshots")
    preprocessing = _preprocessing(params.get("preprocessing"))
    channel_names = _channel_names(params.get("channel_names"))
    spec_data = dict(params.get("spec") or {})
    if spec_data.get("inputs"):
        raise ValueError("Snapshot inputs are supplied by the host, not agent settings")
    inputs: dict[str, Any] = {
        "positions": positions, "order": sorted(positions, key=lambda name: positions[name]),
        "pixel_size_um": calibration, "angles": {"pitch": pitch, "yaw": yaw},
    }
    if damaged:
        inputs["damaged"] = dict(damaged)
    if channel_names:
        inputs["channel_names"] = channel_names
    spec_data.update(image_folder=str(folder), resume=False, inputs=inputs)
    spec = JobSpec.from_dict(spec_data)
    unknown = [name for name in spec.frozen if name != "*" and name not in geometry]
    if unknown:
        raise ValueError(f"frozen names must identify exported snapshots; got {unknown}")
    if spec.plane != "coronal" or spec.atlas != "allen_mouse_25um":
        raise ValueError("The Fiji connector currently requires coronal Allen Mouse snapshots")
    if not spec.tasks:
        raise ValueError("Select at least one LangSlice task")
    # The host's blend is the default appearance; the snapshots and their
    # channels stay the run's images (nothing is staged or rewritten).
    spec.host_preprocessing = _host_appearance(paths, preprocessing)
    abba: dict[str, Any] = {"angles_deg": {"pitch_deg": pitch, "yaw_deg": yaw},
                            "z_offset_mm": z_offset, "existing_warp": sorted(set(existing_warp))}
    return PreparedLinear(folder, spec, geometry, calibration, abba)


def _host_angles(params: dict[str, Any]) -> tuple[float, float]:
    """ABBA's stack-wide cutting angles: ``angles_deg`` (``{"pitch_deg",
    "yaw_deg"}`` in degrees, converted from ``ReslicedAtlas`` by the Fiji
    connector, ``AbbaHostSession.anglesDeg``)."""
    value, keys = params["angles_deg"], ("pitch_deg", "yaw_deg")
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"The host's cutting angles must be {{{keys[0]!r}, {keys[1]!r}}} "
                         "in degrees")
    angles = []
    for key in keys:
        angle = float(value[key])
        if not math.isfinite(angle):
            raise ValueError("The host's cutting angles must be finite")
        angles.append(angle)
    return angles[0], angles[1]


def run_linear(params: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run exported snapshots; positions_mm are BrainGlobe AP millimetres.

    Required: image_folder, pixel_size_um, positions_mm (filename -> mm).
    Optional: spec (JobSpec fields; ``spec.frozen`` names, per snapshot
    filename or ``"*"``, what the agent may not change: no host row is ever
    emitted for a frozen property), damaged (filename -> note,
    flags the agent may not clear) and preprocessing (``preprocess.preview``
    settings for multi-page snapshots: the default appearance; the pages
    stay readable as channels), channel_names (one name per exported page),
    angles_deg (ABBA's stack-wide ``{"pitch_deg",
    "yaw_deg"}``: the job's ``inputs.angles``), z_offset_mm (ABBA's
    slicing-axis offset, kept with the job; the connector converts positions
    with it) and existing_warp (snapshot filenames already carrying the
    user's own warp, kept with the job's ABBA facts).
    Any other key is refused (:data:`REQUEST_KEYS`). The run keeps its full
    agent trace as ``langslice linear run`` does: in the job folder's
    ``trace/``, and also in the folder ``LANGSLICE_TRACE_DIR`` names.
    The run ends with one more checkpoint of the final state.
    Checkpoint events carry initial=True with no mutations for ingestion.
    Later host_updates are replacement corrections relative to the previous
    checkpoint, never cumulative transforms. In a job with the ``nonlinear``
    task a row may carry ``warp`` (the applied deformation as landmark pairs
    on top of the affine step, ``{"error": reason}`` when they cannot be
    made, or null to remove LangSlice's warp step; :class:`HostCheckpoints`),
    and a checkpoint ``host_angles`` when the stack-wide angles changed. The
    result is ``state`` and ``output_dir``.
    """
    from langslice.doors.agent import engine
    from langslice.doors.agent.trace import TRACE_DIR_ENV, job_trace_dir
    from langslice.job.layout import locate_job_folder

    prepared = prepare_linear(params)
    folder, spec = prepared.folder, prepared.spec
    checkpoint = checkpoint_callback(prepared, emit)
    # The run's files: the job folder the engine uses (the spec's job_dir,
    # else next to the snapshots, else the read-only fallback under
    # ~/.langslice/jobs/), found as the engine finds it.
    output_dir, _fallback = locate_job_folder(folder, spec.job_dir, register=False)

    def on_event(event: dict[str, Any]) -> None:
        emit({"kind": "agent_event", "event": public_event(event)})

    with _tracing(None if os.environ.get(TRACE_DIR_ENV) else job_trace_dir(output_dir)):
        state = asyncio.run(engine.run(
            spec, on_write=checkpoint, on_open=checkpoint.attach,
            emit=lambda message: emit({"kind": "log", "message": message}),
            on_event=on_event,
        ))
    # Observers are deliberately isolated by the engine. A final explicit
    # conversion ensures a failed native geometry export cannot report success.
    checkpoint(state)
    result: dict[str, Any] = {"state": state.to_dict(), "output_dir": str(output_dir)}
    return result


@contextmanager
def _tracing(folder: Path | None) -> Iterator[None]:
    """Point the session trace at *folder* for one run, then restore."""
    from langslice.doors.agent.trace import TRACE_DIR_ENV

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

    The same :func:`langslice.core.sections.reading.host_preprocess` that gives
    ``linear.run``'s sections their default appearance, before any agent-side
    resizing. No model, no atlas, no engine import.
    """
    from langslice.core.sections import reading
    from langslice.doors.abba.contract import PreprocessPreviewResult

    source = Path(request.image_path).expanduser().resolve(strict=True)
    output = Path(request.output_path).expanduser()
    if output.suffix.lower() != ".png":
        raise ValueError("Preview output must be a .png file")
    output.parent.mkdir(parents=True, exist_ok=True)
    image = reading.host_preprocess(
        reading.read_pages(source), request.preprocessing.model_dump()
    ).convert("L")
    image.save(output, format="PNG")
    return PreprocessPreviewResult(
        output_path=str(output),
        width=image.width,
        height=image.height,
    )
