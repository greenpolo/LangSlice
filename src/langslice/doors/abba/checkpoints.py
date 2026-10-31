"""The host-row tracker: ABBA-world affine and warp rows per checkpoint.

:class:`HostCheckpoints` turns each state of a job made from an ABBA
session into the rows ABBA applies (one row per section that changed: its
position, orientation, affine and, with the ``nonlinear`` task, warp) and,
when the stack-wide cutting angles changed, ABBA's new angles. It reads the
job's snapshots and calibration through :class:`HostFacts`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from PIL import Image

from langslice.core.fits.deformation import linear_key
from langslice.core.geometry import thin_plate
from langslice.core.jsonable import JsonValue
from langslice.core.maps import abba as abba_maps
from langslice.core.maps import section_maps, tps_landmarks
from langslice.core.maps.section_maps import section_frame, stored_params
from langslice.core.sections.discovery import discover_slices
from langslice.core.stack.rows import ANGLES_KEY
from langslice.doors.abba.warp_workers import (
    MAX_CONVERSIONS,
    ConvertedWarp,
    WarpInput,
    convert_warp,
    started_with_threads,
    thread_share,
)
from langslice.job.state_file import serialized_mixed_angles, stack_to_json

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.job.settings.spec import JobSpec

logger = logging.getLogger(__name__)

#: The job folder's converted warp rows, one file per conversion
#: (``logs/abba_warps/<digest>.json``, :func:`conversion_digest`): a follower
#: started again (ABBA reopened, the menu entry chosen again) sends a
#: deformation it converted before without converting it again.
WARP_CACHE_DIR = "abba_warps"
#: The modules whose code makes a warp row.
CONVERSION_MODULES = (abba_maps, tps_landmarks, thin_plate, section_maps)
#: The warp row's fields a kept row must hold.
ROW_FIELDS = frozenset({
    "source_mm", "target_mm", "max_error_mm", "p99_error_mm",
    "p95_error_mm", "median_error_mm", "region", "points"})
#: A converted row's key: the section, its record key and its linear key.
WarpKey: TypeAlias = tuple[str, str, str]

#: Where a tracker sends its checkpoints and log lines (one event each).
Emit = Callable[[dict[str, JsonValue]], None]
#: Why an ABBA door refuses a job whose sections carry different cutting
#: angles: ABBA shows one atlas angle for the whole stack (its
#: ``ReslicedAtlas`` rotations). Jobs made from an ABBA session have one.
ABBA_MIXED_ANGLES = (
    "This job was made elsewhere with a cutting angle per section, which ABBA "
    "cannot show: ABBA uses one atlas angle for the whole stack. Open the job in "
    "LangSlice outside ABBA (the command line or Claude) instead."
)


@dataclass(frozen=True)
class HostFacts:
    """What the tracker needs of a job made from an ABBA session."""

    #: The snapshot folder: the job's images.
    folder: Path
    #: The job's settings (what is frozen per section, its tasks).
    spec: JobSpec
    #: Each snapshot's ``(width, height)`` in pixels, by filename.
    geometry: dict[str, tuple[int, int]]
    #: The snapshots' micrometres per pixel.
    calibration: float


def snapshot_sizes(folder: Path) -> dict[str, tuple[int, int]]:
    """Return each snapshot's size in pixels.

    Parameters
    ----------
    folder
        The snapshot folder, read as every job reads its images.

    Returns
    -------
    dict
        ``(width, height)`` by filename.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for path in discover_slices(str(folder)):
        with Image.open(path) as image:
            sizes[Path(path).name] = image.size
    return sizes


def host_facts(spec: JobSpec) -> HostFacts:
    """Read the facts of the job *spec* describes.

    Parameters
    ----------
    spec
        The job's settings; its images are the snapshots and its
        ``inputs.pixel_size_um`` their calibration.

    Returns
    -------
    HostFacts
        The snapshots' folder and sizes, the calibration and the spec.

    Raises
    ------
    ValueError
        When the job gives no single pixel size for its snapshots.
    """
    folder = Path(spec.image_folder)
    calibration: object = spec.inputs.get("pixel_size_um")
    if isinstance(calibration, bool):
        calibration = None
    if not isinstance(calibration, float | int):
        raise ValueError(
            "The job gives no pixel size for its ABBA snapshots "
            "(inputs.pixel_size_um).")
    return HostFacts(folder, spec, snapshot_sizes(folder), float(calibration))


def refuse_mixed_angles(state: Any) -> None:
    """Raise ``ValueError`` (:data:`ABBA_MIXED_ANGLES`) when *state* (a
    ``Stack`` or its JSON form) has sections that differ in cutting
    angle; nothing for a single-angle job."""
    mixed = (serialized_mixed_angles(state) if isinstance(state, dict)
             else bool(state.mixed_angles))
    if mixed:
        raise ValueError(ABBA_MIXED_ANGLES)


def baseline_rows(
    current: dict[str, JsonValue], abba_angles: dict[str, float],
) -> dict[str, JsonValue]:
    """Return the state ABBA shows once a job's own steps are removed.

    Every section of *current* unplaced (no position), unflipped and
    unturned, without an affine or a deformation, on ABBA's stack-wide
    angles: the first checkpoint measured against it carries the job's
    whole state.

    Parameters
    ----------
    current
        The job's state in its JSON form (``stack_to_json``).
    abba_angles
        ABBA's cutting angles, ``{"pitch_deg", "yaw_deg"}``.

    Returns
    -------
    dict
        A state in the same JSON form.
    """
    sections = current["slices"]
    assert isinstance(sections, list)  # stack_to_json's form
    rows: list[JsonValue] = []
    for row in sections:
        assert isinstance(row, dict)
        unplaced: dict[str, JsonValue] = {
            "position_mm": None, "flip": False, "rotation_deg": 0,
            "transform": None, "deformation": None}
        rows.append({**row, **unplaced})
    angles: dict[str, JsonValue] = {
        "pitch": abba_angles["pitch_deg"], "yaw": abba_angles["yaw_deg"]}
    return {**current, "slices": rows, ANGLES_KEY: angles}


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
    angles = data[ANGLES_KEY]
    return {"pitch_deg": float(angles["pitch"]), "yaw_deg": float(angles["yaw"])}


class HostCheckpoints:
    """The per-job delta tracker called with the job's state after every
    checkpoint (the follower, ``abba.follow``).

    Every checkpoint refuses a state whose sections differ in cutting angle
    (:func:`refuse_mixed_angles`) before emitting anything: ABBA shows one
    angle for the whole stack. With no :attr:`previous` the checkpoint is
    ingestion (``initial`` True, no rows); otherwise rows (``host_updates``)
    are :func:`_host_updates`' linear rows plus, in a job with the
    ``nonlinear`` task, each section's ``warp`` (:meth:`_warps`; a section
    whose deformation is frozen has none), and ``host_angles`` is added when
    the stack-wide angles differ from the previous checkpoint's.

    :meth:`attach` hands it the opened job and its workspace: the warp rows
    read applied deformation records and section frames through them.
    """

    def __init__(self, facts: HostFacts, emit: Emit) -> None:
        self.facts = facts
        self.emit = emit
        self.job: Job | None = None
        self.workspace: Workspace | None = None
        #: The state the next checkpoint is measured against (JSON form);
        #: None: the next checkpoint is ingestion.
        self.previous: dict[str, Any] | None = None
        #: (section, record key, linear key) -> the warp row (failures are
        #: not kept: the next checkpoint tries again); the job folder's
        #: ``logs/abba_warps/`` keeps them across followers.
        self._warp_cache: dict[WarpKey, dict[str, JsonValue]] = {}
        #: The conversion processes, started for the first checkpoint that
        #: converts more than one warp (:meth:`close` ends them).
        self._pool: ProcessPoolExecutor | None = None

    def attach(self, job: Job, workspace: Workspace) -> None:
        """Read deformation records and section frames through *job* and *workspace*.

        Parameters
        ----------
        job
            The opened job whose states are checkpointed.
        workspace
            Its workspace, for the sections' frames.
        """
        self.job, self.workspace = job, workspace

    def close(self) -> None:
        """End the conversion processes, if any were started."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @property
    def warps(self) -> bool:
        """Whether rows carry warps: the job has the ``nonlinear`` task."""
        return self.facts.spec.has("nonlinear")

    def __call__(self, state: Stack) -> None:
        current = stack_to_json(state)
        # Before anything reaches ABBA: a stack with an angle per section
        # cannot be shown there.
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
        facts = self.facts
        rows = _host_updates(current, previous, facts.geometry, facts.calibration,
                             facts.spec.frozen_for)
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
        from langslice.job.public_files import applied_deformation
        from langslice.job.state_file import stack_from_json

        now, before = stack_from_json(current), stack_from_json(previous)

        def applied(stack: Stack, section: Section) -> str | None:
            held = applied_deformation(stack, section)
            if held is None:
                return None
            return held.key or None

        warps: dict[str, dict[str, Any] | None] = {}
        changed: list[tuple[Section, str]] = []
        for record in now.in_order():
            old = before.by_id(record.id)
            moved = old is None or linear_key(now, record) != linear_key(before, old)
            key = applied(now, record)
            had = applied(before, old) if old is not None else None
            if key is not None and (key != had or moved):
                changed.append((record, key))
            elif key is None and (had is not None or moved):
                warps[record.id] = None
        warps.update(self._convert(now, changed))
        return warps

    def _warp(
        self, state: Stack, section: Section, key: str
    ) -> dict[str, JsonValue]:
        """Return the warp row of *section*'s deformation *key*.

        :meth:`_convert` for one section.
        """
        return self._convert(state, [(section, key)])[section.id]

    def _convert(
        self, state: Stack, changed: list[tuple[Section, str]]
    ) -> dict[str, dict[str, JsonValue]]:
        """Return the warp rows of the *changed* deformations, by section.

        Each row is logged with its measured error and how long it took;
        ``{"error": reason}``, with a log message, when a record cannot be
        read or its thin-plate spline folds at every grid (the connector
        keeps that row as a failed update and names it; ABBA keeps the
        warp it holds). A row converted before for the same record and
        placement (this follower's memory, then the job folder's
        ``logs/abba_warps/``) is not converted again; one section is
        converted here, several side by side in up to
        :data:`MAX_CONVERSIONS` processes.
        """
        rows, pending = self._known(state, changed)
        if len(pending) == 1:
            section, key, cache_key = pending[0]
            try:
                converted = self._warp_row(state, section, key)
            except (OSError, ValueError) as exc:
                rows[section.id] = self._failed(section, exc)
            else:
                rows[section.id] = self._landed(section, cache_key, converted)
        elif pending:
            started = time.perf_counter()
            outcomes = self._side_by_side(
                state, [(section, key) for section, key, _ in pending]
            )
            for section, _key, cache_key in pending:
                outcome = outcomes[section.id]
                rows[section.id] = (
                    self._failed(section, outcome)
                    if isinstance(outcome, Exception)
                    else self._landed(section, cache_key, outcome)
                )
            seconds = time.perf_counter() - started
            at_once = min(len(pending), MAX_CONVERSIONS)
            self.emit({"kind": "log", "message": (
                f"{len(pending)} deformations converted for ABBA in "
                f"{seconds:.1f} s, up to {at_once} at once.")})
        return rows

    def _known(
        self, state: Stack, changed: list[tuple[Section, str]]
    ) -> tuple[
        dict[str, dict[str, JsonValue]], list[tuple[Section, str, WarpKey]]
    ]:
        """Split *changed* into rows converted before and those to convert."""
        rows: dict[str, dict[str, JsonValue]] = {}
        pending: list[tuple[Section, str, WarpKey]] = []
        for section, key in changed:
            cache_key = (section.id, key, linear_key(state, section))
            row = self._warp_cache.get(cache_key) or self._stored(
                section, cache_key
            )
            if row is None:
                pending.append((section, key, cache_key))
            else:
                self._warp_cache[cache_key] = row
                rows[section.id] = row
        return rows, pending

    def _warp_row(
        self, state: Stack, section: Section, key: str
    ) -> ConvertedWarp:
        """Convert one section's deformation in this process."""
        return convert_warp(self._warp_input(state, section, key))

    def _warp_input(
        self, state: Stack, section: Section, key: str
    ) -> WarpInput:
        """Read what converting *section*'s deformation *key* needs."""
        if self.job is None or self.workspace is None:
            raise ValueError(
                "the job's deformation records are not available here"
            )
        record = self.job.deformations.get(section.id, key)
        if record is None:
            raise ValueError("its deformation record cannot be read")
        frame = section_frame(state, self.workspace, section)
        return WarpInput(
            section.id,
            frame,
            record,
            stored_params(section),
            self.facts.geometry[section.id],
            self.facts.calibration,
        )

    def _side_by_side(
        self, state: Stack, changed: list[tuple[Section, str]]
    ) -> dict[str, ConvertedWarp | Exception]:
        """Convert several sections' deformations in the processes."""
        outcomes: dict[str, ConvertedWarp | Exception] = {}
        futures: dict[str, Future[ConvertedWarp]] = {}
        pool = self._processes()
        with started_with_threads(thread_share(MAX_CONVERSIONS)):
            for section, key in changed:
                try:
                    item = self._warp_input(state, section, key)
                except (OSError, ValueError) as exc:
                    outcomes[section.id] = exc
                else:
                    futures[section.id] = pool.submit(convert_warp, item)
        for name, future in futures.items():
            try:
                outcomes[name] = future.result()
            except (OSError, ValueError) as exc:
                outcomes[name] = exc
            except BrokenProcessPool as exc:
                # A process died (out of memory, killed): a failed row,
                # which the next checkpoint converts again.
                self.close()
                outcomes[name] = ValueError(
                    f"its conversion process ended ({exc})"
                )
        return outcomes

    def _processes(self) -> ProcessPoolExecutor:
        """Return the conversion processes, made on first use."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=MAX_CONVERSIONS,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def _failed(
        self, section: Section, exc: Exception
    ) -> dict[str, JsonValue]:
        """Log a conversion that failed and return its error row."""
        self.emit({"kind": "log", "message": (
            f"{section.id}: its deformation is not sent to ABBA ({exc}).")})
        return {"error": f"its deformation could not be converted ({exc})"}

    def _landed(
        self, section: Section, cache_key: WarpKey, converted: ConvertedWarp
    ) -> dict[str, JsonValue]:
        """Log a conversion that ran, keep its row and return it."""
        applied = section.deformation
        record = None if applied is None else applied.record
        row: dict[str, JsonValue] = {
            "source_mm": [list(axis) for axis in converted.source_mm],
            "target_mm": [list(axis) for axis in converted.target_mm],
            "record": record or cache_key[1],
            "max_error_mm": converted.max_error_mm,
            "p99_error_mm": converted.p99_error_mm,
            "p95_error_mm": converted.p95_error_mm,
            "median_error_mm": converted.median_error_mm,
            "region": converted.region,
            "points": converted.points,
        }
        memory = (
            ""
            if converted.peak_mb is None
            else f", its process at most {converted.peak_mb:.0f} MB"
        )
        self.emit({"kind": "log", "message": (
            f"{section.id}: deformation sent to ABBA as "
            f"{converted.points} landmarks; over the {converted.region}, "
            "ABBA's thin-plate spline places the section "
            f"{converted.median_error_mm * 1000:.1f} um from it at the "
            f"median, within {converted.p99_error_mm * 1000:.1f} um at 99% "
            f"of points and {converted.max_error_mm * 1000:.1f} um at most "
            f"(converted in {converted.seconds:.1f} s{memory}).")})
        self._warp_cache[cache_key] = row
        self._store(section, cache_key, row)
        return row

    # --- the job folder's converted rows ---------------------------------

    def _cache_file(
        self, section: Section, cache_key: WarpKey
    ) -> Path | None:
        """Return where *section*'s row is kept, None without a job."""
        if self.job is None:
            return None
        digest = conversion_digest(
            cache_key,
            stored_params(section),
            self.facts.geometry[section.id],
            self.facts.calibration,
        )
        return self.job.layout.logs_dir / WARP_CACHE_DIR / f"{digest}.json"

    def _stored(
        self, section: Section, cache_key: WarpKey
    ) -> dict[str, JsonValue] | None:
        """Return a row converted before, None when there is none."""
        path = self._cache_file(section, cache_key)
        if path is None:
            return None
        try:
            stored = cast("JsonValue", json.loads(path.read_text("utf-8")))
        except (OSError, ValueError):
            return None
        if not isinstance(stored, dict) or not ROW_FIELDS <= stored.keys():
            return None
        self.emit({"kind": "log", "message": (
            f"{section.id}: deformation sent to ABBA as "
            f"{stored['points']} landmarks, converted before.")})
        return stored

    def _store(
        self, section: Section, cache_key: WarpKey, row: dict[str, JsonValue]
    ) -> None:
        """Keep *row* in the job folder for a later follower."""
        path = self._cache_file(section, cache_key)
        if path is None:
            return
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(row), encoding="utf-8")
            partial.replace(path)
        except OSError:
            logger.warning("Could not keep the converted warp in %s", path)


def conversion_digest(
    cache_key: WarpKey,
    params: tuple[float, ...],
    size: tuple[int, int],
    calibration: float,
) -> str:
    """Name one conversion's inputs and the code that converts.

    Parameters
    ----------
    cache_key
        The section's filename, its record key and its linear key.
    params
        Its six stored numbers.
    size
        The snapshot's width and height in pixels.
    calibration
        The snapshot's micrometres per pixel.

    Returns
    -------
    str
        24 hexadecimal digits.
    """
    text = json.dumps(
        [list(cache_key), list(params), list(size), calibration, CODE_DIGEST]
    )
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _code_digest() -> str:
    """Digest the code that makes a warp row (:data:`CONVERSION_MODULES`)."""
    digest = hashlib.sha256()
    for module in CONVERSION_MODULES:
        if module.__file__ is not None:
            digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


#: The digest of the code that makes a warp row; a change to it makes every
#: kept row a new conversion.
CODE_DIGEST = _code_digest()
