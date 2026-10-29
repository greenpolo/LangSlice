"""The host-row tracker: ABBA-world affine and warp rows per checkpoint."""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langslice.job.state_file import stack_to_json

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.doors.abba.worker import Emit, PreparedLinear
logger = logging.getLogger(__name__)
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
    ``Stack`` or its JSON form) has sections that differ in cutting
    angle; nothing for a single-angle job."""
    from langslice.job.state_file import serialized_mixed_angles

    mixed = (serialized_mixed_angles(state) if isinstance(state, dict)
             else bool(state.mixed_angles))
    if mixed:
        raise ValueError(ABBA_MIXED_ANGLES)


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
    from langslice.core.maps.abba import normalized_to_abba_affine

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


def _stack_angles(data: dict[str, Any]) -> dict[str, float]:
    """A serialized state's stack-wide angles as ``{"pitch_deg", "yaw_deg"}``
    (one angle for the stack: :func:`refuse_mixed_angles` ran first)."""
    from langslice.core.stack.rows import ANGLES_KEY

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

    def __call__(self, state: Stack) -> None:
        current = stack_to_json(state)
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
        from langslice.core.fits.deformation import linear_key
        from langslice.job.public_files import applied_deformation
        from langslice.job.state_file import stack_from_json

        now, before = stack_from_json(current), stack_from_json(previous)

        def applied(stack: Stack, section: Section) -> str | None:
            held = applied_deformation(stack, section)
            if held is None:
                return None
            return held.key or None

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

    def _warp(
        self, state: Stack, section: Section, key: str,
    ) -> dict[str, Any]:
        """The warp row of *section*'s applied deformation *key*, its measured
        error logged; ``{"error": reason}``, with a log message, when its
        record cannot be read or its thin-plate spline folds at every grid
        (the connector keeps that row as a failed update and names it; the
        warp ABBA holds is left as it is)."""
        from langslice.core.fits.deformation import linear_key

        cache_key = (section.id, key, linear_key(state, section))
        if cache_key in self._warp_cache:
            return self._warp_cache[cache_key]
        try:
            row = self._warp_row(state, section, key)
        except (OSError, ValueError) as exc:
            self.emit({"kind": "log", "message": (
                f"{section.id}: its deformation is not sent to ABBA "
                f"({exc}).")})
            return {"error": f"its deformation could not be converted ({exc})"}
        else:
            self.emit({"kind": "log", "message": (
                f"{section.id}: deformation sent to ABBA as {row['points']} landmarks; "
                f"ABBA's thin-plate spline differs from it by at most "
                f"{row['max_error_mm'] * 1000:.1f} um (99% of points within "
                f"{row['p99_error_mm'] * 1000:.1f} um).")})
        self._warp_cache[cache_key] = row
        return row

    def _warp_row(
        self, state: Stack, section: Section, key: str,
    ) -> dict[str, Any]:
        from langslice.core.maps.abba import warp_world_landmarks
        from langslice.core.maps.section_maps import (
            section_frame,
            stored_params,
        )

        if self.job is None or self.workspace is None:
            raise ValueError("the job's deformation records are not available here")
        warp = self.job.deformations.get(section.id, key)
        if warp is None:
            raise ValueError("its deformation record cannot be read")
        frame = section_frame(state, self.workspace, section)
        report: dict[str, Any] = {}
        source, target = warp_world_landmarks(
            frame, warp, stored_params(section), size=self.prepared.geometry[section.id],
            pixel_size_um=self.prepared.calibration, diagnostics=report)
        applied = section.deformation
        held = None if applied is None else applied.record
        return {"source_mm": source.T.tolist(), "target_mm": target.T.tolist(),
                "record": held or key,
                "max_error_mm": float(report["max_error_mm"]),
                "p99_error_mm": float(report["p99_error_mm"]),
                "points": int(report["points"])}


def checkpoint_callback(prepared: PreparedLinear, emit: Emit) -> HostCheckpoints:
    """The per-job delta tracker shared by ADK and MCP (:class:`HostCheckpoints`)."""
    return HostCheckpoints(prepared, emit)
