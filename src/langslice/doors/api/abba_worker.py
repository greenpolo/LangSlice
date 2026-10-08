"""JVM-free workers for snapshots exported by the separate Fiji connector.

The host owns image export, native registration actions and the ABBA/AP axis
mapping. This module consumes already calibrated snapshots and emits native-world
geometry; it never starts Java or imports abba_python.
"""
from __future__ import annotations

import asyncio
import logging
import math
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
_SUPPORTED_ATLASES = {"allen_mouse_10um", "allen_mouse_25um", "allen_mouse_50um"}


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
    current: dict[str, Any], previous: dict[str, Any], tasks: list[str],
    geometry: dict[str, tuple[int, int]], pixel_size_um: float,
    locked: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Host rows for what changed from *previous* to *current*.

    A locked section's in-plane geometry belongs to the host: no orientation
    or transform row is ever emitted for it.
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
        if "position" in tasks and (
            row.get("position_mm") != old.get("position_mm")
            or row.get("index_corrected") != old.get("index_corrected")
        ):
            position = float(row["position_mm"])
            if not math.isfinite(position):
                raise ValueError("Agent position must be finite")
            update["position_mm"] = position
        # Mirror and quarter-turn are part of the in-plane transform task.
        if "transform" in tasks and name not in locked and any(
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
    locked: frozenset[str]
    final_updates: list[dict[str, Any]]
    trace_dir: Path | None
    #: The ABBA session's facts the job keeps for reproducibility (stored in
    #: ``job.json`` under ``host.abba`` and returned as the result's
    #: ``abba``): ``angles_deg``, ``z_offset_mm`` (the connector converts
    #: positions with it; the worker's positions are BrainGlobe AP mm),
    #: ``existing_warp`` and ``keep_warp``.
    abba: dict[str, Any] = field(default_factory=dict)
    #: The stack-wide angles at the end, when they differ from the ingested
    #: ones (``{"pitch_deg", "yaw_deg"}``), else None.
    final_angles: dict[str, float] | None = None


def prepare_linear(params: dict[str, Any]) -> PreparedLinear:
    """Validate host snapshots and build their job identically for ADK and MCP."""
    from PIL import Image

    from langslice.core.discovery import discover_slices
    from langslice.core.spec import JobSpec

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
    existing_warp = params.get("existing_warp") or []
    if not isinstance(existing_warp, list) or any(
        not isinstance(name, str) or name not in geometry for name in existing_warp
    ):
        raise ValueError("existing_warp names must identify exported snapshots")
    # The dialog's "Allow the agent to overwrite existing transforms" reaches
    # the worker as `locked` (registered sections are locked unless it is
    # ticked): a section with the user's own warp that is locked keeps it.
    keep_warp = sorted(set(existing_warp) & set(locked))
    # Sections the user chose not to have aligned first (no ABBA
    # registration, Linear off, Positioning on): Nonlinear leaves them alone.
    nonlinear_skip = params.get("nonlinear_skip") or []
    if not isinstance(nonlinear_skip, list) or any(
        not isinstance(name, str) or name not in geometry for name in nonlinear_skip
    ):
        raise ValueError("nonlinear_skip names must identify exported snapshots")
    preprocessing = _preprocessing(params.get("preprocessing"))
    channel_names = _channel_names(params.get("channel_names"))
    spec_data = dict(params.get("spec") or {})
    if spec_data.get("inputs"):
        raise ValueError("Snapshot inputs are supplied by the host, not agent settings")
    inputs: dict[str, Any] = {
        "positions": positions, "order": sorted(positions, key=lambda name: positions[name]),
        "pixel_size_um": calibration, "angles": {"pitch": pitch, "yaw": yaw},
    }
    if locked:
        inputs["locked"] = sorted(set(locked))
    if keep_warp:
        inputs["keep_warp"] = keep_warp
    if nonlinear_skip:
        inputs["nonlinear_skip"] = sorted(set(nonlinear_skip))
    if damaged:
        inputs["damaged"] = dict(damaged)
    if channel_names:
        inputs["channel_names"] = channel_names
    spec_data.update(image_folder=str(folder), resume=False, inputs=inputs)
    spec_data.setdefault("debrief", False)
    spec = JobSpec.from_dict(spec_data)
    if spec.plane != "coronal" or spec.atlas not in _SUPPORTED_ATLASES:
        raise ValueError("The Fiji connector currently requires coronal Allen Mouse snapshots")
    if not spec.tasks:
        raise ValueError("Select at least one LangSlice task")
    # The host's blend is the default appearance; the snapshots and their
    # channels stay the run's images (nothing is staged or rewritten).
    spec.host_preprocessing = _host_appearance(paths, preprocessing)
    abba: dict[str, Any] = {"angles_deg": {"pitch_deg": pitch, "yaw_deg": yaw},
                            "z_offset_mm": z_offset, "existing_warp": sorted(set(existing_warp)),
                            "keep_warp": keep_warp}
    return PreparedLinear(folder, spec, geometry, calibration, frozenset(locked), [],
                          _trace_dir(params.get("trace_dir")), abba)


def _host_angles(params: dict[str, Any]) -> tuple[float, float]:
    """ABBA's stack-wide cutting angles: ``angles_deg`` (``{"pitch_deg",
    "yaw_deg"}``, read from ``ReslicedAtlas`` with :mod:`langslice.core.abba_angles`'
    signs), else flat."""
    value, keys = params.get("angles_deg") or {}, ("pitch_deg", "yaw_deg")
    if not isinstance(value, dict) or set(value) - set(keys):
        raise ValueError(f"The host's cutting angles must be {{{keys[0]!r}, {keys[1]!r}}} "
                         "in degrees")
    angles = []
    for key in keys:
        angle = float(value.get(key, 0.0))
        if not math.isfinite(angle):
            raise ValueError("The host's cutting angles must be finite")
        angles.append(angle)
    return angles[0], angles[1]


#: A section's warp did not change: no ``warp`` key in its row.
_UNCHANGED = object()


def _stack_angles(data: dict[str, Any]) -> dict[str, float] | None:
    """A serialized state's stack-wide angles as ``{"pitch_deg", "yaw_deg"}``."""
    from langslice.core.state import ANGLES_KEY

    angles = data.get(ANGLES_KEY) or {"pitch": 0.0, "yaw": 0.0}
    if not isinstance(angles, dict):
        return None
    return {"pitch_deg": float(angles.get("pitch") or 0.0),
            "yaw_deg": float(angles.get("yaw") or 0.0)}


class HostCheckpoints:
    """The per-job delta tracker the run calls after every checkpoint
    (``engine.run(on_write=...)``, the MCP door's ``job.observe``); the first
    checkpoint is ingestion.

    Every checkpoint refuses a state whose sections differ in cutting angle
    (:func:`refuse_mixed_angles`) before emitting anything: ABBA shows one
    angle for the whole stack. Rows (``host_updates``, ``updates_since_start``,
    ``prepared.final_updates``) are :func:`_host_updates`' linear rows plus,
    in a job with the ``nonlinear`` task, each section's ``warp``
    (:meth:`_warps`; locked sections included); ``host_angles`` is added
    when the stack-wide angles changed since the previous checkpoint.

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
        self.start: dict[str, Any] | None = None
        #: (section, record key, linear key) -> the warp row value.
        self._warp_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

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
        prepared = self.prepared
        payload: dict[str, Any] = {"kind": "checkpoint"}
        if self.previous is None or self.start is None:
            self.start = current
            updates: list[dict[str, Any]] = []
            since_start: list[dict[str, Any]] = []
            payload["initial"] = True
        else:
            payload["initial"] = False
            updates = self._rows(current, self.previous)
            since_start = self._rows(current, self.start)
            angles = _stack_angles(current)
            if angles is not None and angles != _stack_angles(self.previous):
                payload["host_angles"] = angles
            prepared.final_angles = (angles if angles is not None
                                     and angles != _stack_angles(self.start) else None)
        payload.update(state=current, host_updates=updates, updates_since_start=since_start)
        self.emit(payload)
        self.previous = current
        prepared.final_updates = since_start

    # --- rows -----------------------------------------------------------------------

    def _rows(self, current: dict[str, Any], previous: dict[str, Any]) -> list[dict[str, Any]]:
        prepared = self.prepared
        rows = _host_updates(current, previous, prepared.spec.tasks, prepared.geometry,
                             prepared.calibration, prepared.locked)
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
        placement moved under it), ``None`` when it was cleared, undone or
        kept linear, and whenever the placement changed."""
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

    def _warp(self, state: Any, record: Any, key: str) -> dict[str, Any] | None:
        """The warp row of *record*'s applied deformation *key*, its measured
        error logged; None, with a log message, when its record cannot be
        read or its thin-plate spline folds at every grid."""
        from langslice.core.deformation import linear_key

        cache_key = (record.id, key, linear_key(state, record))
        if cache_key in self._warp_cache:
            return self._warp_cache[cache_key]
        try:
            row = self._warp_row(state, record, key)
        except (OSError, ValueError) as exc:
            row = None
            self.emit({"kind": "log", "message": (
                f"{record.id}: its deformation is not sent to ABBA ({exc}); ABBA shows "
                "the linear placement for this section.")})
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
    Optional: spec (JobSpec fields), locked (snapshot filenames whose
    in-plane geometry the agent may not change), damaged (filename -> note,
    flags the agent may not clear) and preprocessing (``preprocess.preview``
    settings for multi-page snapshots: the default appearance; the pages
    stay readable as channels), channel_names (one name per exported page),
    trace_dir (save this run's full agent trace there; the result then names
    the files written), angles_deg (ABBA's stack-wide ``{"pitch_deg",
    "yaw_deg"}``: the job's ``inputs.angles``), z_offset_mm (ABBA's
    slicing-axis offset, kept with the job; the connector converts positions
    with it) and existing_warp (snapshot filenames already carrying the
    user's own warp: those that are also locked keep it,
    ``inputs.keep_warp``) and nonlinear_skip (snapshot filenames the user
    left out of Nonlinear, ``inputs.nonlinear_skip``).
    The run ends with one more checkpoint of the final state (the connector
    applies its rows; it does not apply ``final_updates`` again).
    Checkpoint events carry initial=True with no mutations for ingestion.
    Later host_updates are replacement corrections relative to the previous
    checkpoint, never cumulative transforms; every checkpoint also carries
    updates_since_start (from the ingested state), and the result carries
    final_updates (ingested state -> final state), in the same row format.
    In a job with the ``nonlinear`` task a row may carry ``warp`` (the
    applied deformation as landmark pairs on top of the affine step, or
    null to remove LangSlice's warp step; :class:`HostCheckpoints`), and a
    checkpoint ``host_angles`` when the stack-wide angles changed. The
    result adds ``abba`` (the session facts kept with the job) and
    ``host_angles`` when the final angles differ from the ingested ones.
    """
    from langslice.agent import engine

    prepared = prepare_linear(params)
    folder, spec = prepared.folder, prepared.spec
    checkpoint = checkpoint_callback(prepared, emit)
    trace_dir = prepared.trace_dir
    before = set(trace_dir.glob("*.jsonl")) if trace_dir is not None else set()
    with _tracing(trace_dir):
        state = asyncio.run(engine.run(
            spec, on_write=checkpoint, on_open=checkpoint.attach,
            emit=lambda message: emit({"kind": "log", "message": message}),
            on_event=lambda event: emit({"kind": "agent_event", "event": public_event(event)}),
        ))
    # Observers are deliberately isolated by the engine. A final explicit
    # conversion ensures a failed native geometry export cannot report success.
    checkpoint(state)
    from langslice.job.layout import locate_job_folder

    # The run's files: the job folder the engine used (the spec's job_dir,
    # else next to the snapshots, else the read-only fallback under
    # ~/.langslice/jobs/), found as the engine found it.
    output_dir, _fallback = locate_job_folder(folder, spec.job_dir, register=False)
    result: dict[str, Any] = {"state": state.to_dict(), "output_dir": str(output_dir),
                              "final_updates": prepared.final_updates,
                              "abba": dict(prepared.abba)}
    if prepared.final_angles is not None:
        result["host_angles"] = prepared.final_angles
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
