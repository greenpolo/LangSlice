"""Writing a job's derived files: each section's maps and the stack's exports.

``export_maps`` (a verb of the CLI and the library, not of the model-facing
doors) writes, from the stack as it stands:

- per placed section (``sections/<name>/``, :mod:`langslice.job.formats`):
  ``coords.tif``, ``labels.tif``, ``labels_fiji.tif`` + ``labels.csv``,
  ``residual.tif`` (with an applied deformation) and ``maps.json``, on the
  section's working copy (what every picture is drawn from) or with
  *full_resolution* on the file's own pixels (:func:`langslice.core.maps.section_maps`);
- ``exports/quicknii.json`` (every placed section's linear anchoring) and
  ``exports/visualign.json`` (the same, with VisuAlign markers of each
  applied deformation) (:func:`langslice.job.quint.job_export`);
- ``registration.json`` again, so it lists the files just written;
- ``exports/traces/<name>.npz``: per mapped section whose last
  ``trace_borders`` call holds a reply, the model's lines and its raw
  reply on the maps' grid, before any fit
  (:func:`langslice.job.formats.write_trace_reply`);
- ``exports/plugins/<tool>/``: a copy of the layers tool plugins left on the
  sections, with a manifest each (:func:`langslice.job.plugin_store.export_layers`),
  only when a section has layers.

``submit`` writes the stack's exports (:func:`export_stack`: the QuickNII
and VisuAlign files, ``registration.json``, the plugin layers, and the trace
replies on each section's working-copy grid), never the per-section maps.
:func:`section_maps` computes one section's maps in memory, writing nothing.

Nothing in the state changes (no undo step). A job that persists nothing
(``Job.persist`` False, a dry run) writes nothing and reports what would be
written.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.ops.refusal import unknown_sections

if TYPE_CHECKING:
    from langslice.core.deformable.record import DeformableRecord
    from langslice.core.maps import SectionMaps
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Exported:
    """What :func:`export_maps` wrote (or, on a dry run, would write)."""

    #: ``(absolute path, kind)`` of every file written.
    files: list[tuple[str, str]] = field(default_factory=list)
    #: The sections whose maps were written.
    sections: list[str] = field(default_factory=list)
    #: ``{"id", "reason"}`` per section without maps.
    skipped: list[dict[str, str]] = field(default_factory=list)
    full_resolution: bool = False
    written: bool = True
    seconds: float = 0.0


def _warp(job: Job, state: StackState, record: SliceState,
          ) -> tuple[DeformableRecord | None, str | None]:
    """The section's applied deformation record (None: linear) and its
    stored folder; ``ValueError`` when one is applied but cannot be read."""
    from langslice.job.formats import applied_deformation

    held = applied_deformation(state, record)
    if held is None:
        return None, None
    loaded = job.deformations.get(record.id, str(held.get("key", "")))
    if loaded is None:
        raise ValueError("its applied deformation record cannot be read")
    return loaded, str(held.get("record") or "") or None


def applied_record(job: Job, state: StackState, record: SliceState,
                   ) -> DeformableRecord | None:
    """The section's applied deformation record (None: linear);
    ``ValueError`` when one is applied but cannot be read."""
    return _warp(job, state, record)[0]


def export_maps(
    job: Job, workspace: Workspace, ids: Sequence[str] | None = None, *,
    full_resolution: bool = False,
) -> Exported:
    """Write the maps of *ids* (every section when None or empty) and the
    stack's exports (see the module text). A section without a placement
    (no position or transform), or whose files cannot be read, is skipped
    with its reason. ``UNKNOWN_SLICE_IDS`` when *ids* names a section the
    job does not have.

    Beside other writers (a running agent, CLI calls): the stack is read
    under the job lock after a sync (:meth:`~langslice.job.job.Job.writing`),
    each section's maps are computed OUTSIDE the lock from that snapshot,
    and written under it only when the section's inputs are unchanged
    (:func:`langslice.ops.inputs.section_inputs`); a changed section is
    skipped as ``STALE_INPUT``. The exports and ``registration.json`` are
    written last, under the lock, from the stack as it then stands."""
    from langslice.core.layers import atlas_facts
    from langslice.core.maps import placement_problem, section_frame
    from langslice.core.state import StackState
    from langslice.job import formats
    from langslice.ops.inputs import STALE_INPUT, section_inputs

    started = time.perf_counter()
    with job.writing():
        state = StackState.from_dict(job.snapshot())
    records = state.in_order()
    if ids:
        unknown = [str(ref) for ref in ids if state.resolve(ref) is None]
        if unknown:
            raise unknown_sections(state, unknown)
        wanted = {state.resolve(ref).id for ref in ids}  # type: ignore[union-attr]
        targets = [record for record in records if record.id in wanted]
    else:
        targets = records
    layout = job.layout
    persist = job.persist
    files: list[tuple[str, str]] = []
    done: list[str] = []
    skipped: list[dict[str, str]] = []
    atlas = workspace.atlas
    facts = atlas_facts(atlas)
    for record in targets:
        problem = placement_problem(state, record)
        if problem is not None:
            skipped.append({"id": record.id, "reason": problem})
            continue
        try:
            frame = section_frame(state, workspace, record)
            warp, stored = _warp(job, state, record)
        except (OSError, ValueError) as exc:
            skipped.append({"id": record.id, "reason": str(exc)})
            continue
        if not persist:
            folder = layout.section_dir(record.id)
            names = dict(formats.SECTION_FILES)
            if warp is None:
                names.pop("residual")
            files += [(str(folder / name), kind) for kind, name in names.items()]
            if _completed_trace(record) is not None:
                files.append((str(formats.trace_reply_path(layout, record.id)), "trace_reply"))
            done.append(record.id)
            continue
        computed_from = section_inputs(state, record, deformation=True)
        try:
            maps = _section_maps(workspace, frame, warp, full_resolution)
        except (OSError, ValueError) as exc:
            logger.warning("Could not compute the maps of %s", record.id, exc_info=True)
            skipped.append({"id": record.id, "reason": str(exc)})
            continue
        with job.writing():
            current = job.state.by_id(record.id)
            if current is None or section_inputs(job.state, current,
                                                 deformation=True) != computed_from:
                skipped.append({"id": record.id, "reason": f"{STALE_INPUT}: the section "
                                "changed while its maps were computed; export it again"})
                continue
            entry = formats.section_entry(job.state, workspace, layout, current, facts)
            try:
                written = formats.write_section_maps(
                    layout, maps, atlas, parameters_digest=entry["parameters_digest"],
                    deformation_record=stored)
            except (OSError, ValueError) as exc:
                logger.warning("Could not write the maps of %s", record.id, exc_info=True)
                skipped.append({"id": record.id, "reason": str(exc)})
                continue
        files += [(str(path), kind) for path, kind in written]
        files += _export_trace(job, workspace, state, record, maps.grid_size, full_resolution)
        done.append(record.id)
    exports_dir = layout.exports_dir
    if persist:
        with job.writing():
            files += _write_exports(job, workspace, facts)
    else:
        files += [(str(exports_dir / formats.QUICKNII_FILE), "quicknii"),
                  (str(exports_dir / formats.VISUALIGN_FILE), "visualign"),
                  (str(layout.folder / "registration.json"), "registration")]
    return Exported(files=files, sections=done, skipped=skipped,
                    full_resolution=bool(full_resolution), written=persist,
                    seconds=time.perf_counter() - started)


def export_stack(job: Job, workspace: Workspace) -> Exported:
    """Write the stack's exports and ``registration.json`` (no per-section
    maps; see the module text), under the job lock, from the stack as it
    stands, and each placed section's completed trace reply on its working
    copy's grid (``exports/traces/<name>.npz``, as :func:`export_maps`
    writes it). A job that persists nothing writes nothing (``written``
    False)."""
    from langslice.core.layers import atlas_facts
    from langslice.core.maps import placement_problem, section_frame
    from langslice.core.state import StackState

    started = time.perf_counter()
    if not job.persist:
        return Exported(written=False)
    with job.writing():
        state = StackState.from_dict(job.snapshot())
    files: list[tuple[str, str]] = []
    for record in state.in_order():
        if _completed_trace(record) is None or placement_problem(state, record) is not None:
            continue
        try:
            size = section_frame(state, workspace, record).working_size
        except (OSError, ValueError) as exc:
            logger.warning("Could not export the trace reply of %s: %s", record.id, exc)
            continue
        files += _export_trace(job, workspace, state, record,
                               (int(size[0]), int(size[1])), False)
    with job.writing():
        files += _write_exports(job, workspace, atlas_facts(workspace.atlas))
    return Exported(files=files, seconds=time.perf_counter() - started)


def section_maps(
    job: Job, workspace: Workspace, section: str, *, full_resolution: bool = False,
) -> SectionMaps:
    """One section's maps as :func:`export_maps` writes them, in memory and
    written nowhere (:class:`langslice.core.maps.SectionMaps`: ``coords``,
    ``labels``, ``residual`` (None without a deformation), the grid's
    ``pixel_to_atlas_um``, ``grid_size``, ``um_per_px``), from the stack as
    it stands after a sync. ``UNKNOWN_SLICE_IDS`` for a name that is no
    section's; ``ValueError`` saying why for a section without a placement,
    or whose applied record or image file cannot be read."""
    from langslice.core.maps import placement_problem, section_frame
    from langslice.core.state import StackState

    with job.writing():
        state = StackState.from_dict(job.snapshot())
    record = state.resolve(section)
    if record is None:
        raise unknown_sections(state, [str(section)])
    problem = placement_problem(state, record)
    if problem is not None:
        raise ValueError(f"{record.id}: {problem}")
    frame = section_frame(state, workspace, record)
    warp, _stored = _warp(job, state, record)
    return _section_maps(workspace, frame, warp, full_resolution)


def _completed_trace(record: SliceState) -> dict[str, Any] | None:
    """The section's trace record when its last ``trace_borders`` call holds a reply."""
    held = record.image_correction or {}
    return held if held.get("status") == "ok" and held.get("artifact_dir") else None


def _export_trace(
    job: Job, workspace: Workspace, state: StackState, record: SliceState,
    grid_size: tuple[int, int], full_resolution: bool,
) -> list[tuple[str, str]]:
    """Write the section's trace reply on its maps' grid
    (``exports/traces/<name>.npz``, :func:`langslice.job.formats.write_trace_reply`),
    or remove the one an earlier export left when the section holds no
    completed trace now. A reply whose files cannot be read is logged and
    left out."""
    import langslice
    from langslice.core.handoff import correction_fingerprint
    from langslice.core.maps import render_sizes
    from langslice.core.nonlinear.registration_tool import reply_on_grid
    from langslice.job import formats

    layout = job.layout
    path = formats.trace_reply_path(layout, record.id)
    held = _completed_trace(record)
    if held is None:
        if path.exists():
            path.unlink()
        return []
    artifact_dir = str(held["artifact_dir"])
    try:
        file_size = render_sizes(workspace, record.id).file_size
        borders, reply, grid_from_canvas = reply_on_grid(
            layout.resolve(artifact_dir), file_size, grid_size)
        current = held.get("geometry_fingerprint") == correction_fingerprint(
            state, workspace, record.id)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not export the trace reply of %s: %s", record.id, exc)
        return []
    info = {
        "section": record.id, "grid_size": [int(grid_size[0]), int(grid_size[1])],
        "full_resolution": bool(full_resolution),
        "frame": "the image file as stored (no rotation or flip applied), resized to "
                 "grid_size; the grid of the section's maps",
        "grid_from_canvas": grid_from_canvas.tolist(), "artifact_dir": artifact_dir,
        "attempt": held.get("attempt"), "provider": held.get("provider"),
        "model": held.get("model"), "prompt_edited": bool(held.get("prompt_edited")),
        **({"profile": held["profile"], "untested": True} if held.get("untested") else {}),
        **{key: held[key] for key in ("include", "exclude") if held.get(key)},
        "placement_current": bool(current),
        "written_by": f"LangSlice {langslice.__version__}",
    }
    with job.writing():
        now = job.state.by_id(record.id)
        if now is None or (now.image_correction or {}).get("artifact_dir") != artifact_dir:
            return []  # traced again meanwhile: the next export writes the new reply
        try:
            written = formats.write_trace_reply(layout, record.id, borders, reply, info)
        except OSError:
            logger.warning("Could not write the trace reply of %s", record.id, exc_info=True)
            return []
    return [(str(written), "trace_reply")]


def _section_maps(workspace: Workspace, frame: Any, warp: DeformableRecord | None,
                  full_resolution: bool) -> Any:
    """The section's maps (looked up on the core module at call time)."""
    from langslice.core import maps

    return maps.section_maps(workspace, frame, warp, full_resolution=full_resolution)


def _write_exports(job: Job, workspace: Workspace, facts: dict[str, Any],
                   ) -> list[tuple[str, str]]:
    """The stack's exports (every placed section, whatever the call named)
    and ``registration.json``, from the job's state as it stands. Call it
    under the job lock."""
    from langslice.core.maps import placement_problem, residual_markers, section_frame
    from langslice.job import formats
    from langslice.job.checkpoint import write_json_atomic

    state = job.state
    layout = job.layout
    exports_dir = layout.exports_dir
    files: list[tuple[str, str]] = []
    sections_linear: list[dict[str, Any]] = []
    sections_markers: list[dict[str, Any]] = []
    for record in state.in_order():
        if placement_problem(state, record) is not None:
            continue
        try:
            frame = section_frame(state, workspace, record)
            warp, _stored = _warp(job, state, record)
            markers = residual_markers(frame, warp) if warp is not None else []
        except (OSError, ValueError):
            logger.warning("Not exported: %s", record.id, exc_info=True)
            continue
        base = {"filename": record.id, "width": frame.file_size[0],
                "height": frame.file_size[1], "nr": int(record.index_corrected) + 1,
                "pixel_to_atlas_um": frame.pixel_to_atlas_um()}
        sections_linear.append(base)
        sections_markers.append({**base, "markers": markers})
    if sections_linear:
        from langslice.job.quint import job_export

        exports_dir.mkdir(parents=True, exist_ok=True)
        for name, kind, rows in ((formats.QUICKNII_FILE, "quicknii", sections_linear),
                                 (formats.VISUALIGN_FILE, "visualign", sections_markers)):
            try:
                path = exports_dir / name
                write_json_atomic(str(path), job_export(rows, facts))
            except (OSError, ValueError, KeyError):
                logger.warning("Could not write %s", name, exc_info=True)
                continue
            files.append((str(path), kind))
    path = formats.write_registration(layout, state, workspace)
    if path is not None:
        files.append((str(path), "registration"))
    # tool plugins' layers (docs/developers/plugins.md)
    if any(record.layers for record in state.slices):
        from langslice.job.plugin_store import export_layers

        files += [(str(layout.resolve(relative)), "plugin_layer")
                  for relative in export_layers(layout, state)]
    return files
