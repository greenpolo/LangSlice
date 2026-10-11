"""Edit persistent corresponding pairs; explicitly apply their triangulation."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.ndimage import map_coordinates

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.deformable import FitSettings, finish_fit, prepare_fit
from langslice.core.deformable.engines import EngineResult
from langslice.core.deformable.geometry import WorkingGrid
from langslice.core.handoff import digest
from langslice.core.triangulation import visualign_mesh
from langslice.ops.refusal import Refused, check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


def _section(job: Job, section: str) -> SliceState:
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [section])
    # The pairs only serve a deformation: a section whose deformation is frozen
    # takes neither pairs nor a triangulation (FROZEN).
    check_frozen(job, [record.id], "deformation")
    return record


def _endpoint(job: Job, record: SliceState, point: dict[str, Any], role: str) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise Refused("BAD_ARGS", message="Each endpoint must be an object with picture, x and y.")
    try:
        seq, x, y = int(point["picture"]), float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError):
        raise Refused(
            "BAD_ARGS", message="Each endpoint needs picture, x and y in pixels."
        ) from None
    if point.get("picture") != seq or isinstance(point.get("picture"), bool):
        raise Refused("BAD_ARGS", message="picture is a whole saved picture number.")
    if set(point) - {"picture", "x", "y"}:
        raise Refused("BAD_ARGS", message="An endpoint accepts picture, x and y only.")
    picture = job.views.lookup(seq)
    if picture is None:
        raise Refused("UNKNOWN_PICTURE", picture=seq)
    recipe = picture.recipe or {}
    panels = [
        p
        for p in recipe.get("endpoint_panels", [])
        if p.get("role") == role and p.get("section") == record.id
    ]
    if not panels:
        raise Refused(
            "BAD_LANDMARK_PICTURE",
            picture=seq,
            message=(
                f"Picture {seq} has no exact {role} geometry for {record.id}; "
                + ("draw the atlas plane of this section with look." if role == "atlas"
                   else "draw this section with look.")
            ),
        )
    eligible = [
        p
        for p in panels
        if p["bounds"][0] <= x < p["bounds"][2] and p["bounds"][1] <= y < p["bounds"][3]
    ]
    if not eligible or not np.isfinite([x, y]).all():
        raise Refused(
            "BAD_LANDMARK_POINT",
            picture=seq,
            message="The point is outside image content (or in a caption/gutter).",
        )
    panel = eligible[0]
    held = recipe.get("registration_context") or {}
    if role == "atlas":
        if held.get("plane") != geometry.plane_context(job.state, record):
            raise Refused(
                "STALE_LANDMARK_PLANE",
                picture=seq,
                message="This picture's atlas plane differs from the section's current plane.",
            )
        canonical = geometry.transform(panel["pixel_to_native"], [x, y])
        limits = np.asarray(panel["native_size"])
    else:
        warp = None
        key = panel.get("warp_key")
        if key:
            warp = job.deformations.get(record.id, key)
            if warp is None or warp.inverse_field_mm is None:
                raise Refused(
                    "LANDMARK_WARP_UNAVAILABLE",
                    picture=seq,
                    message="The saved warped view needs its stored inverse mapping.",
                )
        canonical = geometry.section_point(panel, [x, y], warp)
        limits = np.ones(2)
    if not np.isfinite(canonical).all() or np.any(canonical < 0) or np.any(canonical >= limits):
        raise Refused(
            "BAD_LANDMARK_POINT",
            picture=seq,
            message=f"The point is outside the {role}'s image. Coordinates are never clamped.",
        )
    return {
        "canonical_xy": canonical.tolist(),
        "picture": seq,
        "pixel_xy": [x, y],
        "registration_context": copy.deepcopy(held),
        "geometry": copy.deepcopy(panel),
    }


def set_landmarks(
    job: Job,
    workspace: Workspace,
    section: str,
    pairs: Sequence[dict[str, Any]] = (),
    delete: Sequence[str] = (),
) -> dict[str, Any]:
    """Atomically add/edit/delete pairs without applying any deformation."""
    del workspace
    if (
        isinstance(pairs, (str, bytes))
        or not isinstance(pairs, Sequence)
        or any(not isinstance(pair, dict) for pair in pairs)
        or isinstance(delete, (str, bytes))
        or not isinstance(delete, Sequence)
        or any(not isinstance(pair_id, str) for pair_id in delete)
    ):
        raise Refused(
            "BAD_ARGS", message="pairs is a list of objects; delete is a list of pair IDs."
        )
    record = _section(job, section)
    current = copy.deepcopy(record.landmarks or {"next_id": 1, "pairs": []})
    by_id = {pair["id"]: pair for pair in current["pairs"]}
    if len(set(delete)) != len(delete) or len({p.get("id") for p in pairs if p.get("id")}) != sum(
        bool(p.get("id")) for p in pairs
    ):
        raise Refused("BAD_ARGS", message="A pair ID may be edited or deleted only once per batch.")
    changed: list[str] = []
    for pair_id in delete:
        if pair_id not in by_id:
            raise Refused("UNKNOWN_LANDMARK", id=pair_id)
        del by_id[pair_id]
    for edit in pairs:
        pair_id = str(edit.get("id") or "")
        if pair_id and pair_id not in by_id:
            raise Refused("UNKNOWN_LANDMARK", id=pair_id)
        if set(edit) - {"id", "atlas", "section"}:
            raise Refused("BAD_ARGS", message="Pair entries accept id, atlas and section only.")
        if not any(role in edit for role in ("atlas", "section")):
            raise Refused("BAD_ARGS", message="An edit needs at least one endpoint.")
        previous = by_id.get(pair_id)
        if not previous and (not edit.get("atlas") or not edit.get("section")):
            raise Refused(
                "BAD_ARGS", message="A new pair needs explicit atlas and section endpoints."
            )
        item = copy.deepcopy(previous or {})
        if not pair_id:
            pair_id = f"L{current['next_id']}"
            current["next_id"] += 1
        item["id"] = pair_id
        for role in ("atlas", "section"):
            if role in edit:
                endpoint = _endpoint(job, record, edit[role], role)
                item[role] = endpoint
                item["atlas_native_xy" if role == "atlas" else "section_fraction_xy"] = endpoint[
                    "canonical_xy"
                ]
        if "atlas" in edit:
            item["plane"] = geometry.plane_context(job.state, record)
        item["stale"] = geometry.stale(job.state, record, item)
        by_id[pair_id] = item
        changed.append(pair_id)
    current["pairs"] = list(by_id.values())
    before = job.snapshot()
    written = bool(pairs or delete) and record.landmarks != current
    if written:
        record.landmarks = current
        job.commit(before)
    return {
        "status": "ok",
        "section": record.id,
        "written": written,
        "changed": changed,
        "deleted": list(delete),
        "landmarks": copy.deepcopy(current["pairs"]),
    }


def _sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            map_coordinates(
                field[..., k], [points[..., 1], points[..., 0]], order=1, mode="nearest"
            )
            for k in range(2)
        ],
        -1,
    )


def landmark_triangulation(job: Job, workspace: Workspace, section: str) -> dict[str, Any]:
    """Apply the complete set, replacing the preceding landmark application."""
    record = _section(job, section)
    pairs = record.landmarks.get("pairs", [])
    if len(pairs) < 3:
        raise Refused(
            "INSUFFICIENT_LANDMARKS", message="At least three corresponding pairs are required."
        )
    if any(geometry.stale(job.state, record, pair) for pair in pairs):
        raise Refused(
            "STALE_LANDMARK_PLANE",
            message="Update the stale atlas endpoints at the current plane first.",
        )
    revision = digest(pairs)
    held = record.deformation or {}
    if held.get("landmark_revision") == revision:
        return {
            "status": "ok",
            "section": record.id,
            "written": False,
            "landmark_count": len(pairs),
            **held.get("summary", {}),
        }
    started = time.monotonic()
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
        current = job.deformations.current(job.state, record)
        if held and "keep_linear" not in held and current is None:
            raise Refused(
                "LANDMARK_WARP_UNAVAILABLE",
                message="The current deformation cannot be read; undo it first.",
            )
        replacing = current is not None and current.engine.get("name") == "landmark_triangulation"
        previous = current.parent if replacing and current is not None else current
        baseline_key = held.get("landmark_baseline_key") if replacing else held.get("key")
        baseline_steps = copy.deepcopy(
            held.get("landmark_baseline_steps", held.get("steps", [])[:-1])
            if replacing else held.get("steps", [])
        )
        if previous is not None and previous.inverse_field_mm is None:
            raise Refused(
                "LANDMARK_WARP_UNAVAILABLE",
                message="Composition needs the baseline deformation's inverse.",
            )
        size = grid.image.size
        source = (
            geometry.transform(
                np.linalg.inv(geometry.orientation(record)),
                [p["section_fraction_xy"] for p in pairs],
            )
            * np.asarray(size)
            - 0.5
        )
        target = geometry.transform(
            grid.placement.atlas_to_section, [p["atlas_native_xy"] for p in pairs]
        )
        if previous is not None:
            assert previous.inverse_field_mm is not None
            target += _sample(previous.inverse_field_mm, target) / previous.mm_per_px
        mesh = visualign_mesh(source, target, size=size)
        yy, xx = np.indices((size[1], size[0]), dtype=float)
        points = np.stack([xx, yy], -1)
        mm = grid.placement.section_mm_per_px
        forward, inverse = mesh.map(points), mesh.inverse(points)
        if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
            raise ValueError("The mesh does not cover the full section frame")
        image, _identity = deformation.stain_image(workspace, job.state, grid, deformation.FIT_LOOK)
        prepared = prepare_fit(
            image, workspace.atlas, grid.placement, FitSettings(), previous=previous
        )
        prepared.grid = WorkingGrid(size, mm, size)
        result = finish_fit(
            prepared,
            EngineResult(
                field_mm=(forward - points) * mm,
                inverse_field_mm=(inverse - points) * mm,
                inverse_source="exact inverse on the same landmark triangles",
                native_parameters={
                    "source_xy": mesh.source.tolist(),
                    "target_xy": mesh.target.tolist(),
                    "simplices": mesh.simplices.tolist(),
                    "margin": 0.1,
                },
                runtime_s=time.monotonic() - started,
                engine_version=scipy.__version__,
            ),
        )
        result.engine["name"] = "landmark_triangulation"
        result.settings = replace(result.settings, engine="landmark_triangulation")
    except Refused:
        raise
    except (ValueError, OSError) as exc:
        raise Refused("LANDMARK_TRIANGULATION_FAILED", message=str(exc)) from exc
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
        "pairs": copy.deepcopy(pairs),
    }
    job.deformations.put(key, result)
    folder = job.deformations.save(record.id, key, result)
    numbers = deformation.summary(result, current)
    before = job.snapshot()
    record.deformation = {
        "key": key,
        "record": job.layout.relative(folder),
        "linear_key": deformation.linear_key(job.state, record),
        "summary": numbers,
        "inverse_source": result.inverse_source,
        "landmark_revision": revision,
        "landmark_baseline_key": baseline_key,
        "landmark_baseline_steps": baseline_steps,
        "steps": [
            *baseline_steps,
            {
                "method": "landmark_triangulation",
                "start": "current" if previous else "linear",
                "landmark_revision": revision,
                "landmark_count": len(pairs),
            },
        ],
    }
    job.commit(before)
    return {
        "status": "ok",
        "section": record.id,
        "written": True,
        "landmark_count": len(pairs),
        "composed": previous is not None,
        "replaced": replacing,
        **numbers,
    }
