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
from langslice.core.triangulation import DegenerateMesh, visualign_mesh
from langslice.ops.refusal import Refused, check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


#: ``delete=[DELETE_ALL]`` removes every pair of the section.
DELETE_ALL = "all"


def _section(job: Job, section: str) -> SliceState:
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [section])
    # The pairs only serve a deformation: a section whose deformation is frozen
    # takes neither pairs nor a triangulation (FROZEN).
    check_frozen(job, [record.id], "deformation")
    return record


def _span(low: float, high: float) -> str:
    return f"{int(np.floor(low))}-{int(np.ceil(high)) - 1}"


def _content(seq: int, panels: Sequence[dict[str, Any]]) -> str:
    """Where picture *seq*'s image content lies, from its panels' bounds."""
    boxes = []
    for panel in panels:
        x0, y0, x1, y1 = panel["bounds"]
        text = f"x {_span(x0, x1)}, y {_span(y0, y1)}"
        if text not in boxes:
            boxes.append(text)
    return f"image content is {' or '.join(boxes)} in picture {seq}"


def _plane_words(plane: dict[str, Any]) -> str:
    angles = plane.get("angles") or [0, 0]
    return f"{plane.get('position_mm')} mm, angles {list(angles)}"


def _endpoint(
    job: Job, record: SliceState, point: dict[str, Any], role: str, label: str = "the pair"
) -> dict[str, Any]:
    """The saved endpoint of *point* (``picture``, ``x``, ``y``) in *role*;
    *label* says which pair of the call it is, for the refusals."""
    here = f"{label}, {role} endpoint"
    if not isinstance(point, dict):
        raise Refused(
            "BAD_ARGS", message=f"{here}: an endpoint is an object with picture, x and y."
        )
    try:
        seq, x, y = int(point["picture"]), float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError):
        raise Refused(
            "BAD_ARGS", message=f"{here}: an endpoint needs picture, x and y in pixels."
        ) from None
    if point.get("picture") != seq or isinstance(point.get("picture"), bool):
        raise Refused("BAD_ARGS", message=f"{here}: picture is a whole saved picture number.")
    if set(point) - {"picture", "x", "y"}:
        raise Refused("BAD_ARGS", message=f"{here}: an endpoint accepts picture, x and y only.")
    picture = job.views.lookup(seq)
    if picture is None:
        raise Refused(
            "UNKNOWN_PICTURE",
            picture=seq,
            message=(
                f"{here}: there is no saved picture {seq}. Use the picture number "
                "a look or zoom reply gave."
            ),
        )
    recipe = picture.recipe or {}
    every = [p for p in recipe.get("endpoint_panels", []) if p.get("role") == role]
    panels = [p for p in every if p.get("section") == record.id]
    if not panels:
        others = sorted({str(p.get("section")) for p in every})
        raise Refused(
            "BAD_LANDMARK_PICTURE",
            picture=seq,
            message=(
                f"{here}: picture {seq} has no {role} geometry for {record.id}"
                + (f" (it has it for {', '.join(others)})" if others else "")
                + "; use a picture "
                + ("of this section's atlas plane or an overlay of it" if role == "atlas"
                   else "of this section or an overlay of it")
                + f" drawn with look (sections=[{record.id!r}])."
            ),
        )
    where = f"({x:g}, {y:g}) in picture {seq}"
    eligible = [
        p
        for p in panels
        if p["bounds"][0] <= x < p["bounds"][2] and p["bounds"][1] <= y < p["bounds"][3]
    ]
    if not eligible or not np.isfinite([x, y]).all():
        raise Refused(
            "BAD_LANDMARK_POINT",
            picture=seq,
            message=(
                f"{here}: the point {where} is outside the image content (in a caption/gutter, "
                f"or off the picture); {_content(seq, panels)}. Choose a point inside it."
            ),
        )
    panel = eligible[0]
    held = recipe.get("registration_context") or {}
    if role == "atlas":
        now = geometry.plane_context(job.state, record)
        if held.get("plane") != now:
            raise Refused(
                "STALE_LANDMARK_PLANE",
                picture=seq,
                message=(
                    f"{here}: picture {seq} was drawn at the atlas plane "
                    f"{_plane_words(held.get('plane') or {})}, but {record.id} is now at "
                    f"{_plane_words(now)}. Draw its atlas plane again with look "
                    f"(sections=[{record.id!r}]) and place the point on that picture."
                ),
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
                    message=f"{here}: the warped view in picture {seq} needs its stored "
                    "inverse mapping, which is gone; draw the section again with look.",
                )
        canonical = geometry.section_point(panel, [x, y], warp)
        limits = np.ones(2)
    if not np.isfinite(canonical).all() or np.any(canonical < 0) or np.any(canonical >= limits):
        raise Refused(
            "BAD_LANDMARK_POINT",
            picture=seq,
            message=(
                f"{here}: the point {where} lies outside the {role} image; "
                f"{_content(seq, [panel])}. Coordinates are never clamped."
            ),
        )
    return {
        "canonical_xy": canonical.tolist(),
        "picture": seq,
        "pixel_xy": [x, y],
        "registration_context": copy.deepcopy(held),
        "geometry": copy.deepcopy(panel),
    }


def _listing(job: Job, record: SliceState) -> list[dict[str, Any]]:
    """One compact record per pair: id, the endpoints as the call gave them
    (picture, x, y) and the state (``applied``, ``pending`` or ``stale``)."""
    states = geometry.pair_states(job.state, record)
    rows = []
    for pair in record.landmarks.get("pairs", []):
        row: dict[str, Any] = {"id": pair["id"]}
        for role in ("atlas", "section"):
            x, y = pair[role]["pixel_xy"]
            row[role] = {"picture": pair[role]["picture"], "x": round(x, 1), "y": round(y, 1)}
        row["state"] = states[pair["id"]]
        rows.append(row)
    return rows


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    states = [row["state"] for row in rows]
    return {"total": len(states), **{k: states.count(k) for k in ("applied", "pending", "stale")}}


def set_landmarks(
    job: Job,
    workspace: Workspace,
    section: str,
    pairs: Sequence[dict[str, Any]] = (),
    delete: Sequence[str] = (),
) -> dict[str, Any]:
    """Atomically add/edit/delete pairs without applying any deformation.

    ``delete=["all"]`` clears the section's pairs (pair IDs are never reused).
    With no pairs and no delete nothing is written and the reply is the
    listing: one record per pair with its state (``applied``: in the landmark
    warp applied now; ``pending``: saved but not applied; ``stale``: placed
    against an atlas plane the section has left).
    """
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
    deleted = list(by_id) if DELETE_ALL in delete else list(delete)
    if DELETE_ALL in delete and len(delete) > 1:
        raise Refused("BAD_ARGS", message=f'delete ["{DELETE_ALL}"] stands alone.')
    for pair_id in deleted:
        if pair_id not in by_id:
            raise Refused(
                "UNKNOWN_LANDMARK",
                id=pair_id,
                message=(
                    f"{record.id} has no pair {pair_id}; its pairs are "
                    f"{', '.join(by_id) or 'none'}."
                ),
            )
        del by_id[pair_id]
    for index, edit in enumerate(pairs):
        pair_id = str(edit.get("id") or "")
        label = f"pairs[{index}] ({pair_id or 'new pair'})"
        if pair_id and pair_id not in by_id:
            raise Refused(
                "UNKNOWN_LANDMARK",
                id=pair_id,
                message=(
                    f"{label}: {record.id} has no pair {pair_id}; its pairs are "
                    f"{', '.join(by_id) or 'none'}. Leave id out to add a pair."
                ),
            )
        if set(edit) - {"id", "atlas", "section"}:
            raise Refused(
                "BAD_ARGS", message=f"{label}: pair entries accept id, atlas and section only."
            )
        if not any(role in edit for role in ("atlas", "section")):
            raise Refused("BAD_ARGS", message=f"{label}: an edit needs at least one endpoint.")
        previous = by_id.get(pair_id)
        if not previous and (not edit.get("atlas") or not edit.get("section")):
            raise Refused(
                "BAD_ARGS",
                message=f"{label}: a new pair needs explicit atlas and section endpoints.",
            )
        item = copy.deepcopy(previous or {})
        if not pair_id:
            pair_id = f"L{current['next_id']}"
            current["next_id"] += 1
        item["id"] = pair_id
        for role in ("atlas", "section"):
            if role in edit:
                endpoint = _endpoint(job, record, edit[role], role, label)
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
    rows = _listing(job, record)
    return {
        "status": "ok",
        "section": record.id,
        "written": written,
        **({} if pairs or delete else {"note": "No pairs or delete given: this is the listing; "
                                                "nothing was written."}),
        "changed": changed,
        "deleted": deleted,
        "counts": _counts(rows),
        "landmarks": rows,
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


#: Folded triangles a reply lists; the rest are counted.
MAX_FOLDS_LISTED = 8


def _degenerate_words(exc: DegenerateMesh, names: Sequence[str]) -> str:
    def ids(group: Sequence[int]) -> str:
        return ", ".join(names[i - 4] if i >= 4 else "a fixed corner" for i in group)

    groups = "; ".join(ids(group) for group in exc.vertices[:6])
    if "Repeated" in str(exc):
        return (f"Pairs with the same section point cannot be triangulated ({groups}). "
                "Move one section endpoint of each group, or delete one pair.")
    return (f"The section points of {groups} lie on one line, a zero-area triangle. "
            "Move one of those section endpoints off the line, or delete one pair.")


def _folds(mesh: Any, points: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """The warp's folded triangles: the share of the section frame they cover
    and their pair ids (a fixed corner vertex has none, so a triangle at the
    frame's edge lists fewer than three)."""
    triangles = [[names[i - 4] for i in sorted(tri) if i >= 4] for tri in mesh.folded_triangles()]
    if not triangles:
        return {}
    return {"area_fraction": round(mesh.folded_fraction(points), 5), "triangles": triangles}


def _folds_reply(folds: dict[str, Any] | None) -> dict[str, Any]:
    """The reply's ``folds``: nothing when the warp has none."""
    if not folds:
        return {}
    triangles = folds["triangles"]
    ids = sorted({i for tri in triangles for i in tri}, key=lambda s: (len(s), s))
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


def landmark_triangulation(job: Job, workspace: Workspace, section: str) -> dict[str, Any]:
    """Apply the complete set, replacing the preceding landmark application."""
    record = _section(job, section)
    pairs = record.landmarks.get("pairs", [])
    if len(pairs) < 3:
        raise Refused(
            "INSUFFICIENT_LANDMARKS",
            message=f"{record.id} has {len(pairs)} saved pair(s); at least three corresponding "
            "pairs are required (set_landmarks adds them).",
        )
    stale_ids = [pair["id"] for pair in pairs if geometry.stale(job.state, record, pair)]
    if stale_ids:
        raise Refused(
            "STALE_LANDMARK_PLANE",
            pairs=stale_ids,
            message=(
                f"Pairs {', '.join(stale_ids)} were placed against an earlier atlas plane of "
                f"{record.id}. Draw its current atlas plane with look, then re-place their atlas "
                "endpoints with set_landmarks (the pair id and an atlas endpoint), or delete them."
            ),
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
            **_folds_reply(held.get("landmark_folds")),
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
        names = [p["id"] for p in pairs]
        try:
            mesh = visualign_mesh(source, target, size=size)
        except DegenerateMesh as exc:
            # The mesh's first four vertices are the fixed corners.
            raise Refused(
                "LANDMARK_TRIANGULATION_FAILED",
                pairs=sorted({names[i - 4] for group in exc.vertices for i in group if i >= 4}),
                message=_degenerate_words(exc, names),
            ) from exc
        yy, xx = np.indices((size[1], size[0]), dtype=float)
        points = np.stack([xx, yy], -1)
        mm = grid.placement.section_mm_per_px
        forward, inverse = mesh.map(points), mesh.inverse(points)
        folds = _folds(mesh, points, names)
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
        "landmark_pairs": {p["id"]: geometry.pair_key(p) for p in pairs},
        "landmark_folds": folds,
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
        **_folds_reply(folds),
    }
