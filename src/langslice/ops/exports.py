"""Writing a job's derived files: each section's maps and the stack's exports.

``export_maps`` (a verb of the CLI and the library, not of the model-facing
doors) and ``submit`` write, from the stack as it stands:

- per placed section (``sections/<name>/``, :mod:`langslice.job.formats`):
  ``coords.tif``, ``labels.tif``, ``labels_fiji.tif`` + ``labels.csv``,
  ``residual.tif`` (with an applied deformation) and ``maps.json``, on the
  section's working copy (what every picture is drawn from) or with
  *full_resolution* on the file's own pixels (:func:`langslice.core.maps.section_maps`);
- ``exports/quicknii.json`` (every placed section's linear anchoring) and
  ``exports/visualign.json`` (the same, with VisuAlign markers of each
  applied deformation) (:func:`langslice.job.quint.job_export`);
- ``registration.json`` again, so it lists the files just written.

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

from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.deformable.record import DeformableRecord
    from langslice.core.state import SliceState
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


def _warp(job: Job, record: SliceState) -> tuple[DeformableRecord | None, str | None]:
    """The section's applied deformation record (None: linear) and its
    stored folder; ``ValueError`` when one is applied but cannot be read."""
    from langslice.job.formats import applied_deformation

    held = applied_deformation(job.state, record)
    if held is None or "keep_linear" in held:
        return None, None
    loaded = job.deformations.get(record.id, str(held.get("key", "")))
    if loaded is None:
        raise ValueError("its applied deformation record cannot be read")
    return loaded, str(held.get("record") or "") or None


def export_maps(
    job: Job, workspace: Workspace, ids: Sequence[str] | None = None, *,
    full_resolution: bool = False,
) -> Exported:
    """Write the maps of *ids* (every section when None or empty) and the
    stack's exports (see the module text). A section without a placement
    (no position or transform), or whose files cannot be read, is skipped
    with its reason. ``UNKNOWN_SLICE_IDS`` when *ids* names a section the
    job does not have."""
    from langslice.core.maps import placement_problem, residual_markers, section_frame, section_maps
    from langslice.job import formats

    started = time.perf_counter()
    state = job.state
    records = state.in_order()
    if ids:
        unknown = [str(ref) for ref in ids if state.resolve(ref) is None]
        if unknown:
            raise Refused("UNKNOWN_SLICE_IDS", unknown=unknown)
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
    from langslice.core.layers import atlas_facts

    facts = atlas_facts(atlas)
    for record in targets:
        problem = placement_problem(state, record)
        if problem is not None:
            skipped.append({"id": record.id, "reason": problem})
            continue
        try:
            frame = section_frame(state, workspace, record)
            warp, stored = _warp(job, record)
        except (OSError, ValueError) as exc:
            skipped.append({"id": record.id, "reason": str(exc)})
            continue
        if not persist:
            folder = layout.section_dir(record.id)
            names = dict(formats.SECTION_FILES)
            if warp is None:
                names.pop("residual")
            files += [(str(folder / name), kind) for kind, name in names.items()]
            done.append(record.id)
            continue
        entry = formats.section_entry(state, workspace, layout, record, facts)
        try:
            maps = section_maps(workspace, frame, warp, full_resolution=full_resolution)
            written = formats.write_section_maps(
                layout, maps, atlas, parameters_digest=entry["parameters_digest"],
                deformation_record=stored)
        except (OSError, ValueError) as exc:
            logger.warning("Could not write the maps of %s", record.id, exc_info=True)
            skipped.append({"id": record.id, "reason": str(exc)})
            continue
        files += [(str(path), kind) for path, kind in written]
        done.append(record.id)
    # The stack's exports: every placed section, whatever *ids* named.
    exports_dir = layout.exports_dir
    if persist:
        sections_linear: list[dict[str, Any]] = []
        sections_markers: list[dict[str, Any]] = []
        for record in records:
            if placement_problem(state, record) is not None:
                continue
            try:
                frame = section_frame(state, workspace, record)
                warp, _stored = _warp(job, record)
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
                    path = formats.write_json(exports_dir / name, job_export(rows, facts))
                except (OSError, ValueError, KeyError):
                    logger.warning("Could not write %s", name, exc_info=True)
                    continue
                files.append((str(path), kind))
        path = formats.write_registration(layout, state, workspace)
        if path is not None:
            files.append((str(path), "registration"))
    else:
        files += [(str(exports_dir / formats.QUICKNII_FILE), "quicknii"),
                  (str(exports_dir / formats.VISUALIGN_FILE), "visualign"),
                  (str(layout.folder / "registration.json"), "registration")]
    return Exported(files=files, sections=done, skipped=skipped,
                    full_resolution=bool(full_resolution), written=persist,
                    seconds=time.perf_counter() - started)
