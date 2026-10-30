"""Apply a section's landmark pairs: their VisuAlign triangulation, applied.

:func:`landmark_triangulation` builds the warp of the section's saved pairs
(:mod:`langslice.tools.landmarks`) and the four fixed corners, as VisuAlign
builds it, and applies it as one undo step: composed onto the section's
deformation at the first application, replacing the preceding landmark
application after that. It returns :class:`LandmarksApplied`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.ndimage import map_coordinates

from langslice.core.fits import deformation
from langslice.core.fits import landmarks as geometry
from langslice.core.fits.deformable import finish_fit, prepare_warp
from langslice.core.fits.deformable.engines import EngineResult
from langslice.core.fits.deformable.fields import MeshStep, mesh_fields
from langslice.core.fits.deformable.geometry import WorkingGrid
from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.fits.frame import digest
from langslice.core.geometry.triangulation import (
    DegenerateMesh,
    PiecewiseAffine,
    visualign_mesh,
)
from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.deformation import DeformationRef, LandmarkWarp
from langslice.core.stack.landmarks import Pair
from langslice.core.stack.landmarks_json import pair_to_json
from langslice.tools.landmarks import landmark_section
from langslice.tools.look import requested_edge, show_result
from langslice.tools.run import Run, pictured

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


def _sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            map_coordinates(
                field[..., k],
                [points[..., 1], points[..., 0]],
                order=1,
                mode="nearest",
            )
            for k in range(2)
        ],
        -1,
    )


#: What the apply reply's two displacement figures are.
DISPLACEMENT_NOTE_KEY = "displacement_note"
DISPLACEMENT_NOTE = (
    "displacement_mm is the whole deformation now applied to the section (every step "
    "since its linear placement, the landmark warp included); step_displacement_mm, when "
    "present, is only the change this application made over the warp it replaced or "
    "was applied onto."
)
#: Folded triangles a reply lists; the rest are counted.
MAX_FOLDS_LISTED = 8


@dataclass(frozen=True)
class LandmarksApplied:
    """What :func:`landmark_triangulation` applied, or found applied."""

    section: str
    #: False: the section's warp was already the one these pairs make.
    written: bool
    #: The pairs the warp is built from, by id.
    built_from: tuple[str, ...]
    #: The applied deformation's numbers: displacement, folds, flags.
    numbers: JsonObject
    #: The warp's folded triangles (``area_fraction``, ``triangles``);
    #: empty or None when it has none.
    folds: JsonObject | None
    #: Built on the section's earlier deformation (written only).
    composed: bool = False
    #: Replacing the preceding landmark application (written only).
    replaced: bool = False

    def reply(self) -> dict[str, Any]:
        """Return the tool's reply: what was applied and its numbers.

        Returns
        -------
        dict
            ``status``, ``section``, ``written``, ``landmark_count``,
            ``built_from``, ``composed`` and ``replaced`` when written, the
            numbers, the note on the displacement figures, and ``folds``.
        """
        reply: dict[str, Any] = {
            "status": "ok",
            "section": self.section,
            "written": self.written,
            "landmark_count": len(self.built_from),
            "built_from": list(self.built_from),
        }
        if self.written:
            reply["composed"] = self.composed
            reply["replaced"] = self.replaced
        reply.update(self.numbers)
        reply[DISPLACEMENT_NOTE_KEY] = DISPLACEMENT_NOTE
        reply.update(_folds_reply(self.folds))
        return reply


def _degenerate_words(exc: DegenerateMesh, names: Sequence[str]) -> str:
    def ids(group: Sequence[int]) -> str:
        return ", ".join(
            "a fixed corner" if i < 4 else names[i - 4] for i in group
        )

    groups = "; ".join(ids(group) for group in exc.vertices[:6])
    if "Repeated" in str(exc):
        return (
            f"Pairs with the same section point cannot be triangulated ({groups}). "
            "Move one section endpoint of each group, or delete one pair."
        )
    return (
        f"The section points of {groups} lie on one line, a zero-area triangle. "
        "Move one of those section endpoints off the line, or delete one pair."
    )


def _folds(
    mesh: Any, points: np.ndarray, names: Sequence[str]
) -> dict[str, Any]:
    """The warp's folded triangles: the share of the section frame they cover
    and their vertices' pair ids (a fixed corner vertex has none, so a
    triangle at the frame's edge lists fewer than three)."""
    triangles = [
        [names[i - 4] for i in sorted(tri) if i >= 4]
        for tri in mesh.folded_triangles()
    ]
    if not triangles:
        return {}
    return {
        "area_fraction": round(mesh.folded_fraction(points), 5),
        "triangles": triangles,
    }


def _folds_reply(folds: Mapping[str, Any] | None) -> dict[str, Any]:
    """The reply's ``folds``: nothing when the warp has none."""
    if not folds:
        return {}
    triangles = folds["triangles"]
    ids = sorted(
        {i for tri in triangles for i in tri}, key=lambda s: (len(s), s)
    )
    return {
        "folds": {
            "area_fraction": folds["area_fraction"],
            "pairs": ids,
            "triangles": triangles[:MAX_FOLDS_LISTED],
            "triangle_count": len(triangles),
            "note": (
                "Folded triangles are applied as VisuAlign does (outlined in magenta on the "
                "picture). A fold means the section points of these pairs are ordered "
                "differently from their atlas points; fix those pairs' endpoints if it is wrong."
            ),
        }
    }


def _require_applicable(job: Job, record: Section) -> tuple[Pair, ...]:
    """The section's pairs, refusing too few of them or any placed against another plane."""
    pairs = geometry.section_pairs(record)
    if len(pairs) < 3:
        raise Refused(
            RefusalCode.INSUFFICIENT_LANDMARKS,
            message=f"{record.id} has {len(pairs)} saved pair(s); at least three corresponding "
            "pairs are required (set_landmarks adds them).",
        )
    stale_ids = [
        pair.id for pair in pairs if geometry.stale(job.state, record, pair)
    ]
    if stale_ids:
        raise Refused(
            RefusalCode.STALE_LANDMARK_PLANE,
            pairs=stale_ids,
            message=(
                f"Pairs {', '.join(stale_ids)} were placed against an earlier atlas plane of "
                f"{record.id}. Draw its current atlas plane with look, then re-place their atlas "
                "endpoints with set_landmarks (the pair id and an atlas endpoint), or delete them."
            ),
        )
    return pairs


def _mesh(
    names: Sequence[str],
    source: np.ndarray,
    target: np.ndarray,
    size: tuple[int, int],
) -> PiecewiseAffine:
    """The VisuAlign mesh of the pairs and the four fixed corners, as VisuAlign
    builds it; invalid geometry is refused naming the pairs."""
    try:
        return visualign_mesh(source, target, size=size)
    except DegenerateMesh as exc:
        # The mesh's first four vertices are the fixed corners.
        raise Refused(
            RefusalCode.LANDMARK_TRIANGULATION_FAILED,
            pairs=sorted(
                {
                    names[i - 4]
                    for group in exc.vertices
                    for i in group
                    if i >= 4
                }
            ),
            message=_degenerate_words(exc, names),
        ) from exc


def _landmark_reference(
    job: Job,
    section: Section,
    folder: Path,
    key: str,
    result: DeformableRecord,
    numbers: dict[str, Any],
    revision: str,
    pairs: Sequence[Pair],
    folds: dict[str, Any],
    baseline_key: str | None,
    baseline_steps: tuple[JsonObject, ...],
    composed: bool,
) -> DeformationRef:
    """The deformation reference the section holds for a landmark application."""
    step: JsonObject = {
        "method": "landmark_triangulation",
        "start": "current" if composed else "linear",
        "landmark_revision": revision,
        "landmark_count": len(pairs),
    }
    keys = {p.id: geometry.pair_key(p) for p in pairs}
    warp = LandmarkWarp(
        revision=revision,
        pairs=keys,
        folds=folds,
        baseline_key=baseline_key,
        baseline_steps=baseline_steps,
    )
    return DeformationRef(
        key=key,
        linear_key=deformation.linear_key(job.state, section),
        steps=(*baseline_steps, step),
        record=job.layout.relative(folder),
        summary=numbers,
        inverse_source=result.inverse_source,
        landmark=warp,
    )


def _unchanged(record: Section, pairs: Sequence[Pair]) -> LandmarksApplied:
    """Return the answer when the section's warp is the one these pairs make.

    Nothing is written; the numbers are the applied deformation's.
    """
    applied = record.deformation
    summary: JsonObject = {}
    if applied is not None and applied.summary is not None:
        summary = applied.summary
    warp = geometry.landmark_warp(record)
    folds = None if warp is None else to_jsonable(warp.folds)
    return LandmarksApplied(
        section=record.id,
        written=False,
        built_from=tuple(p.id for p in pairs),
        numbers=summary,
        folds=folds,
    )


def landmark_triangulation(
    job: Job, workspace: Workspace, section: str
) -> LandmarksApplied:
    """Apply the complete set, replacing the preceding landmark application."""
    record = landmark_section(job, section)
    pairs = _require_applicable(job, record)
    # The warp is the pairs' and the four corners' mesh alone; a warp saved
    # before that (with automatic holds) has another revision, so the same
    # pairs build it again.
    pair_forms = [pair_to_json(pair) for pair in pairs]
    revision = digest({"pairs": pair_forms, "mesh": "pairs and corners"})
    held = record.deformation
    applied_warp = geometry.landmark_warp(record)
    if applied_warp is not None and applied_warp.revision == revision:
        return _unchanged(record, pairs)
    started = time.monotonic()
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
        try:
            current = job.deformations.current(job.state, record)
        except (
            ValueError
        ):  # its record.json does not load (the error names it)
            logger.warning(
                "Applied deformation of %s unreadable",
                record.id,
                exc_info=True,
            )
            current = None
        if held is not None and current is None:
            raise Refused(
                RefusalCode.LANDMARK_WARP_UNAVAILABLE,
                message="The current deformation cannot be read; undo it first.",
            )
        replacing = (
            current is not None
            and current.engine.get("name") == "landmark_triangulation"
        )
        previous = (
            current.parent if replacing and current is not None else current
        )
        baseline_key: str | None = None
        baseline_steps: tuple[JsonObject, ...] = ()
        if replacing and applied_warp is not None:
            baseline_key = applied_warp.baseline_key
            baseline_steps = applied_warp.baseline_steps
        elif held is not None:
            baseline_key, baseline_steps = held.key, held.steps or ()
        size = grid.image.size
        source = (
            geometry.transform(
                np.linalg.inv(geometry.orientation(record)),
                [p.section_fraction_xy for p in pairs],
            )
            * np.asarray(size)
            - 0.5
        )
        target = geometry.transform(
            grid.placement.atlas_to_section, [p.atlas_native_xy for p in pairs]
        )
        if previous is not None:
            assert previous.inverse_field_mm is not None
            target += (
                _sample(previous.inverse_field_mm, target) / previous.mm_per_px
            )
        names = [p.id for p in pairs]
        prepared = prepare_warp(
            workspace.atlas,
            grid.placement,
            size,
            deformation.DETAIL_LEVEL,
            previous=previous,
        )
        mesh = _mesh(names, source, target, size)
        yy, xx = np.indices((size[1], size[0]), dtype=float)
        points = np.stack([xx, yy], -1)
        mm = grid.placement.section_mm_per_px
        # The fields as loading the record rebuilds them from its mesh.
        step = MeshStep(mesh.source, mesh.target, mesh.simplices)
        forward_mm, inverse_mm = mesh_fields(step, size, mm)
        folds = _folds(mesh, points, names)
        finite = np.isfinite(forward_mm).all()
        finite = finite and np.isfinite(inverse_mm).all()
        if not finite:
            raise ValueError("The mesh does not cover the full section frame")
        prepared.grid = WorkingGrid(size, mm, size)
        result = finish_fit(
            prepared,
            EngineResult(
                field_mm=forward_mm,
                inverse_field_mm=inverse_mm,
                inverse_source="exact inverse on the same landmark triangles",
                native_parameters={
                    "source_xy": mesh.source.tolist(),
                    "target_xy": mesh.target.tolist(),
                    "simplices": mesh.simplices.tolist(),
                    "margin": 0.1,
                    # vertices: 4 fixed corners, then the pairs
                    "pair_count": len(names),
                },
                runtime_s=time.monotonic() - started,
                engine_version=scipy.__version__,
            ),
            landmark_warp=True,
        )
        result.engine["name"] = "landmark_triangulation"
        result.stored_step = step
    except Refused:
        raise
    except (ValueError, OSError) as exc:
        raise Refused(
            RefusalCode.LANDMARK_TRIANGULATION_FAILED, message=str(exc)
        ) from exc
    key = digest(
        {
            "landmarks": revision,
            "linear": deformation.linear_key(job.state, record),
            "previous": baseline_key,
        }
    )
    result.provenance = {
        "tool": "landmark_triangulation",
        "section_id": record.id,
        "landmark_revision": revision,
        "landmark_baseline_key": baseline_key,
        "pairs": pair_forms,
    }
    job.deformations.put(key, result)
    folder = job.deformations.save(record.id, key, result)
    numbers = deformation.summary(result, current)
    before = job.snapshot()
    record.deformation = _landmark_reference(
        job,
        record,
        folder,
        key,
        result,
        numbers,
        revision,
        pairs,
        folds,
        baseline_key,
        baseline_steps,
        previous is not None,
    )
    job.commit(before)
    return LandmarksApplied(
        section=record.id,
        written=True,
        built_from=tuple(p.id for p in pairs),
        numbers=numbers,
        folds=folds,
        composed=previous is not None,
        replaced=replacing,
    )


# --- the tool ----------------------------------------------------------------


def landmark_triangulation_tool(
    run: Run, section: str, *, view: bool = True, resolution: int = 0
) -> dict[str, Any]:
    """Run the ``landmark_triangulation`` tool: apply, answer, picture.

    Parameters
    ----------
    run
        The run the tool is called in.
    section, view, resolution
        The declared arguments as the model sent them
        (``tools/descriptions/landmarks.py``).

    Returns
    -------
    dict
        The reply (:meth:`LandmarksApplied.reply`), with a
        ``resolution_note`` when the edge was clamped, and its picture.
    """
    edge, note = requested_edge(run, resolution)
    applied = landmark_triangulation(run.job, run.workspace, section)
    result = applied.reply()
    if note:
        result["resolution_note"] = note
    if not run.shows(view):
        return result
    shown = show_result(
        run.job,
        run.workspace,
        "landmark_triangulation",
        "overlay",
        [applied.section],
        resolution=edge,
    )
    return pictured(result, shown)
