"""JVM-free workers for snapshots exported by the separate Fiji connector.

The host owns image export, native registration actions and the ABBA/AP axis
mapping. This module consumes already calibrated snapshots and emits native-world
geometry; it never starts Java or imports abba_python.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.core.spec import JobSpec
    from langslice.doors.api.models import PreprocessPreviewRequest, PreprocessPreviewResult

Emit = Callable[[dict[str, Any]], None]
logger = logging.getLogger(__name__)
#: The keys of a ``linear.run`` request (``mcp.prepare`` adds ``host_channel``);
#: any other is refused.
REQUEST_KEYS = ("image_folder", "pixel_size_um", "positions_mm", "z_offset_mm", "angles_deg",
                "existing_warp", "damaged", "channel_names", "spec", "preprocessing")
#: Why a run stopped before submitting (``agent.session``'s stop reasons),
#: as the connector's final message says it.
STOP_REASONS = {
    "quota_budget": "it used this run's share of the usage window (max_quota_percent)",
    "input_context_limit": "a request exceeded the context limit (max_input_tokens)",
}


#: Why an ABBA door refuses a job whose sections carry different cutting
#: angles: ABBA shows one atlas angle for the whole stack (its
#: ``ReslicedAtlas`` rotations). Jobs made from an ABBA session have one.
ABBA_MIXED_ANGLES = (
    "This job was made elsewhere with a cutting angle per section, which ABBA "
    "cannot show: ABBA uses one atlas angle for the whole stack. Open the job in "
    "LangSlice outside ABBA (the command line or Claude) instead."
)


def refuse_mixed_angles(state: Any) -> None:
    """Raise ``ValueError`` (:data:`ABBA_MIXED_ANGLES`) when *state* (a
    ``StackState`` or its ``to_dict()``) has sections that differ in cutting
    angle; nothing for a single-angle job."""
    from langslice.core.state import serialized_mixed_angles

    mixed = (serialized_mixed_angles(state) if isinstance(state, dict)
             else bool(state.mixed_angles))
    if mixed:
        raise ValueError(ABBA_MIXED_ANGLES)


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


def _host_updates(
    current: dict[str, Any], previous: dict[str, Any],
    geometry: dict[str, tuple[int, int]], pixel_size_um: float,
    frozen: Callable[[str], frozenset[str]] = lambda _name: frozenset(),
) -> list[dict[str, Any]]:
    """Host rows for what changed from *previous* to *current*.

    A property frozen on a section (*frozen*: ``JobSpec.frozen_for``, which
    includes what a task that is off supplies) belongs to the host: no row for
    it is ever emitted for that section.
    """
    from langslice.core.abba_affine import normalized_to_abba_affine

    prior = {row["id"]: row for row in previous["slices"]}
    updates = []
    for row in current["slices"]:
        name = row["id"]
        if name not in prior or name not in geometry:
            raise ValueError("Agent checkpoint contains an unknown snapshot")
        old = prior[name]
        update: dict[str, Any] = {"id": name}
        held = frozen(name)
        if "position" not in held and (
            row.get("position_mm") != old.get("position_mm")
            or row.get("index_corrected") != old.get("index_corrected")
        ):
            position = float(row["position_mm"])
            if not math.isfinite(position):
                raise ValueError("Agent position must be finite")
            update["position_mm"] = position
        # Mirror and quarter-turn are part of the in-plane transform task.
        if "orientation" not in held and any(
            row.get(key) != old.get(key) for key in ("flip", "rotation_deg")
        ):
            update.update(
                flip=bool(row.get("flip")), rotation_deg=int(row.get("rotation_deg") or 0),
            )
        if "affine" not in held and row.get("transform") != old.get("transform"):
            transform = row.get("transform")
            if transform is None:
                update["affine_mm"] = None
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
    from langslice.doors.api.models import PreprocessingSettings

    return PreprocessingSettings.model_validate(value).model_dump()


def _host_appearance(
    paths: list[str], settings: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The default appearance a host's snapshots are shown in, validated.

    Snapshots may be multi-page TIFFs, one page per exported channel. When the
    host sends preprocessing settings, or any snapshot has several pages, the
    run shows each section as :func:`langslice.core.image_prep.host_preprocess`
    blends it — the function ``preprocess.preview`` shows the user — but the
    blend is only the DEFAULT appearance (``JobSpec.host_preprocessing``): the
    snapshots stay the run's images and their pages stay readable as raw
    channels, for the agent's ``look`` and channel tools. Otherwise
    (no settings, single pages) None: the engine's own automatic path.
    """
    from langslice.core.image_prep import host_preprocess_settings, page_count

    counts = {path: page_count(path) for path in paths}
    if settings is None and all(count == 1 for count in counts.values()):
        return None
    # Validate every snapshot's settings before the run starts.
    for count in counts.values():
        host_preprocess_settings(settings, count)
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

    from langslice.core.discovery import discover_slices
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


def _stack_angles(data: dict[str, Any]) -> dict[str, float]:
    """A serialized state's stack-wide angles as ``{"pitch_deg", "yaw_deg"}``
    (one angle for the stack: :func:`refuse_mixed_angles` ran first)."""
    from langslice.core.state import ANGLES_KEY

    angles = data[ANGLES_KEY]
    return {"pitch_deg": float(angles["pitch"]), "yaw_deg": float(angles["yaw"])}


class HostCheckpoints:
    """The per-job delta tracker the run calls after every checkpoint
    (``engine.run(on_write=...)``, the MCP door's ``job.observe``); the first
    checkpoint is ingestion.

    Every checkpoint refuses a state whose sections differ in cutting angle
    (:func:`refuse_mixed_angles`) before emitting anything: ABBA shows one
    angle for the whole stack. Rows (``host_updates``) are
    :func:`_host_updates`' linear rows plus, in a job with the ``nonlinear``
    task, each section's ``warp`` (:meth:`_warps`; a section whose
    deformation is frozen has none); ``host_angles`` is added when the
    stack-wide angles changed since the previous checkpoint.

    :meth:`attach` hands it the opened job and its workspace (the engine's
    ``on_open``; the MCP door after opening): the warp rows read applied
    deformation records and section frames through them, and the job keeps
    the ABBA session's facts (``job.json``, ``host.abba``).
    """

    def __init__(self, prepared: PreparedLinear, emit: Emit) -> None:
        self.prepared = prepared
        self.emit = emit
        self.job: Any = None
        self.workspace: Any = None
        self.previous: dict[str, Any] | None = None
        #: (section, record key, linear key) -> the warp row (failures are
        #: not kept: the next checkpoint tries again).
        self._warp_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def attach(self, job: Any, workspace: Any) -> None:
        self.job, self.workspace = job, workspace
        if not self.prepared.abba or not getattr(job, "persist", True):
            return
        from langslice.job.layout import read_job_file, write_job_file

        try:
            held = read_job_file(job.layout) or {}
            host = dict(held.get("host") or {})
            host["abba"] = dict(self.prepared.abba)
            write_job_file(job.layout, host=host)
        except (OSError, ValueError):
            logger.warning("Could not record the ABBA session in the job file", exc_info=True)

    @property
    def warps(self) -> bool:
        return self.prepared.spec.has("nonlinear")

    def __call__(self, state: Any) -> None:
        current = state.to_dict()
        # Before anything reaches ABBA: a stack with an angle per section
        # cannot be shown there (the run's own final call raises it too).
        refuse_mixed_angles(current)
        payload: dict[str, Any] = {"kind": "checkpoint"}
        if self.previous is None:
            updates: list[dict[str, Any]] = []
            payload["initial"] = True
        else:
            payload["initial"] = False
            updates = self._rows(current, self.previous)
            angles = _stack_angles(current)
            if angles != _stack_angles(self.previous):
                payload["host_angles"] = angles
        payload["host_updates"] = updates
        self.emit(payload)
        self.previous = current

    # --- rows -----------------------------------------------------------------------

    def _rows(self, current: dict[str, Any], previous: dict[str, Any]) -> list[dict[str, Any]]:
        prepared = self.prepared
        rows = _host_updates(current, previous, prepared.geometry, prepared.calibration,
                             prepared.spec.frozen_for)
        if not self.warps:
            return rows
        by_id = {row["id"]: row for row in rows}
        for name, warp in self._warps(current, previous).items():
            if name not in by_id:
                by_id[name] = {"id": name}
                rows.append(by_id[name])
            by_id[name]["warp"] = warp
        order = {row["id"]: index for index, row in enumerate(current["slices"])}
        return sorted(rows, key=lambda row: order.get(row["id"], len(order)))

    def _warps(self, current: dict[str, Any], previous: dict[str, Any],
               ) -> dict[str, dict[str, Any] | None]:
        """``{section: warp}`` for the sections whose warp row changes: the
        new deformation's pairs when one was applied or replaced (or its
        placement moved under it; ``{"error": reason}`` when they cannot be
        made), ``None`` when it was cleared, undone or kept linear, and
        whenever the placement changed."""
        from langslice.core.deformation import linear_key
        from langslice.core.state import StackState
        from langslice.job.formats import applied_deformation

        now, before = StackState.from_dict(current), StackState.from_dict(previous)

        def applied(stack: StackState, record: Any) -> str | None:
            held = applied_deformation(stack, record)
            if held is None or "keep_linear" in held:
                return None
            return str(held.get("key") or "") or None

        warps: dict[str, dict[str, Any] | None] = {}
        for record in now.in_order():
            old = before.by_id(record.id)
            moved = old is None or linear_key(now, record) != linear_key(before, old)
            key = applied(now, record)
            had = applied(before, old) if old is not None else None
            if key is not None and (key != had or moved):
                warps[record.id] = self._warp(now, record, key)
            elif key is None and (had is not None or moved):
                warps[record.id] = None
        return warps

    def _warp(self, state: Any, record: Any, key: str) -> dict[str, Any]:
        """The warp row of *record*'s applied deformation *key*, its measured
        error logged; ``{"error": reason}``, with a log message, when its
        record cannot be read or its thin-plate spline folds at every grid
        (the connector keeps that row as a failed update and names it; the
        warp ABBA holds is left as it is)."""
        from langslice.core.deformation import linear_key

        cache_key = (record.id, key, linear_key(state, record))
        if cache_key in self._warp_cache:
            return self._warp_cache[cache_key]
        try:
            row = self._warp_row(state, record, key)
        except (OSError, ValueError) as exc:
            self.emit({"kind": "log", "message": (
                f"{record.id}: its deformation is not sent to ABBA ({exc}).")})
            return {"error": f"its deformation could not be converted ({exc})"}
        else:
            self.emit({"kind": "log", "message": (
                f"{record.id}: deformation sent to ABBA as {row['points']} landmarks; "
                f"ABBA's thin-plate spline differs from it by at most "
                f"{row['max_error_mm'] * 1000:.1f} um (99% of points within "
                f"{row['p99_error_mm'] * 1000:.1f} um).")})
        self._warp_cache[cache_key] = row
        return row

    def _warp_row(self, state: Any, record: Any, key: str) -> dict[str, Any]:
        from langslice.core.abba_warp import warp_world_landmarks
        from langslice.core.maps import section_frame, stored_params

        if self.job is None or self.workspace is None:
            raise ValueError("the job's deformation records are not available here")
        warp = self.job.deformations.get(record.id, key)
        if warp is None:
            raise ValueError("its deformation record cannot be read")
        frame = section_frame(state, self.workspace, record)
        report: dict[str, Any] = {}
        source, target = warp_world_landmarks(
            frame, warp, stored_params(record), size=self.prepared.geometry[record.id],
            pixel_size_um=self.prepared.calibration, diagnostics=report)
        held = record.deformation or {}
        return {"source_mm": source.T.tolist(), "target_mm": target.T.tolist(),
                "record": str(held.get("record") or key),
                "max_error_mm": float(report["max_error_mm"]),
                "p99_error_mm": float(report["p99_error_mm"]),
                "points": int(report["points"])}


def checkpoint_callback(prepared: PreparedLinear, emit: Emit) -> HostCheckpoints:
    """The per-job delta tracker shared by ADK and MCP (:class:`HostCheckpoints`)."""
    return HostCheckpoints(prepared, emit)


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
    ``trace/`` unless ``LANGSLICE_TRACE_DIR`` names another folder.
    The run ends with one more checkpoint of the final state.
    Checkpoint events carry initial=True with no mutations for ingestion.
    Later host_updates are replacement corrections relative to the previous
    checkpoint, never cumulative transforms. In a job with the ``nonlinear``
    task a row may carry ``warp`` (the applied deformation as landmark pairs
    on top of the affine step, ``{"error": reason}`` when they cannot be
    made, or null to remove LangSlice's warp step; :class:`HostCheckpoints`),
    and a checkpoint ``host_angles`` when the stack-wide angles changed. The
    result is ``state``, ``output_dir`` and, when the agent stopped before
    submitting for a budget, ``stopped`` (:data:`STOP_REASONS`).
    """
    from langslice.agent import engine
    from langslice.agent.trace import TRACE_DIR_ENV, job_trace_dir
    from langslice.job.layout import locate_job_folder

    prepared = prepare_linear(params)
    folder, spec = prepared.folder, prepared.spec
    checkpoint = checkpoint_callback(prepared, emit)
    # The run's files: the job folder the engine uses (the spec's job_dir,
    # else next to the snapshots, else the read-only fallback under
    # ~/.langslice/jobs/), found as the engine finds it.
    output_dir, _fallback = locate_job_folder(folder, spec.job_dir, register=False)
    stopped: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        if event.get("kind") == "complete" and event.get("stopped"):
            stopped.append(str(event["stopped"]))
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
    if stopped and stopped[-1] in STOP_REASONS:
        result["stopped"] = STOP_REASONS[stopped[-1]]
    return result


@contextmanager
def _tracing(folder: Path | None) -> Iterator[None]:
    """Point the session trace at *folder* for one run, then restore."""
    from langslice.agent.trace import TRACE_DIR_ENV

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

    The same :func:`langslice.core.image_prep.host_preprocess` that gives
    ``linear.run``'s sections their default appearance, before any
    agent-side resizing. No model, no atlas, no engine import.
    """
    from langslice.core.image_prep import host_preprocess, read_pages
    from langslice.doors.api.models import PreprocessPreviewResult

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
