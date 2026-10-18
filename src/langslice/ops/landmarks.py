"""Edit persistent corresponding pairs; explicitly apply their triangulation."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.ndimage import map_coordinates

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.atlas.tree import structure_table
from langslice.core.deformable import finish_fit, prepare_warp
from langslice.core.deformable.engines import EngineResult
from langslice.core.deformable.geometry import WorkingGrid
from langslice.core.handoff import digest
from langslice.core.picture_geometry import source_frame
from langslice.core.triangulation import DegenerateMesh, visualign_mesh
from langslice.ops.refusal import Refused, check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


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


def _content(seq: int, panels: Sequence[dict[str, Any]], frame: Any = None) -> str:
    """Where picture *seq*'s image content lies, from its panels' bounds (on a
    zoom, in the grid's numbers: the whole picture's pixels)."""
    boxes = []
    for panel in panels:
        x0, y0, x1, y1 = panel["bounds"]
        if frame is not None:
            x0, y0 = frame.to_source([x0 - 0.5, y0 - 0.5]) + 0.5
            x1, y1 = frame.to_source([x1 - 0.5, y1 - 0.5]) + 0.5
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
        given = (x, y)
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
                f"{here}: there is no saved picture {seq}"
                + (f" (the newest is {newest.seq})" if (newest := job.views.latest(
                    exclude_tool=None)) else "")
                + ". Use the picture number a look or zoom reply gave."
            ),
        )
    recipe = picture.recipe or {}
    # On a zoom, x and y are the numbers its grid prints: pixels of the whole
    # picture it was cut from. They are the same on the source picture.
    frame = source_frame(recipe)
    if frame is not None and np.isfinite([x, y]).all():
        x, y = (float(v) for v in frame.to_picture([x, y]))
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
    where = f"({given[0]:g}, {given[1]:g}) in picture {seq}"
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
                f"or off the picture); {_content(seq, panels, frame)}. Choose a point inside it."
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
            try:
                warp = job.deformations.get(record.id, key)
            except ValueError:  # its record.json does not load (the error names it)
                logger.warning("Landmark view warp of %s unreadable", record.id, exc_info=True)
                warp = None
            if warp is None:
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
                f"{_content(seq, [panel], frame)}. Coordinates are never clamped."
            ),
        )
    return {
        "canonical_xy": canonical.tolist(),
        "picture": seq,
        "pixel_xy": list(given),
        "registration_context": copy.deepcopy(held),
        "geometry": copy.deepcopy(panel),
    }


def _atlas_regions(
    workspace: Workspace, plane: dict[str, Any], items: Sequence[dict[str, Any]]
) -> tuple[dict[str, str], list[str]]:
    """The region drawn at each of *items*' atlas endpoint on the atlas *plane*
    (``plane_context``), as ``ACRONYM (name)`` by pair id, and the ids whose
    atlas endpoint is outside the atlas."""
    from langslice.ops.atlas import label_at, plane_annotation, region_words

    if not items:
        return {}, []
    annotation = plane_annotation(workspace, plane)
    structures = structure_table(workspace.atlas)
    regions: dict[str, str] = {}
    outside: list[str] = []
    for item in items:
        label = label_at(annotation, item["atlas_native_xy"])
        if not label:
            regions[item["id"]] = "outside the atlas"
            outside.append(item["id"])
            continue
        regions[item["id"]] = region_words(structures, label)
    return regions, outside


def _outside_words(outside: Sequence[str]) -> str:
    one = len(outside) == 1
    return (f"The atlas endpoint{'' if one else 's'} of {', '.join(outside)} "
            f"{'is' if one else 'are'} outside the atlas, where no brain is drawn: "
            f"{'it pairs' if one else 'they pair'} empty atlas with the tissue. For a pair "
            "on the outline, put the atlas endpoint on the drawn border or just inside it; "
            "a point read off a close-up picture lands within a few pixels of what you "
            "meant, a point read off a whole picture can miss by 20 pixels or more.")


def _listing(job: Job, record: SliceState) -> list[dict[str, Any]]:
    """One compact record per pair: id, the endpoints as the call gave them
    (picture, x, y) and the state (``applied``, ``moved``, ``pending`` or
    ``stale``)."""
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
    kinds = ("applied", "moved", "pending", "stale")
    return {"total": len(states), **{k: states.count(k) for k in kinds}}


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
    warp applied now; ``moved``: in it at other endpoints; ``pending``: saved
    but not in it; ``stale``: placed against an atlas plane the section has
    left), and ``warp``: the pairs the current warp was built from, those not
    in it, those deleted but still shaping it.
    """
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
    placed: list[dict[str, Any]] = []
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
            placed.append(item)
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
    regions, outside = _atlas_regions(workspace, geometry.plane_context(job.state, record),
                                      placed)
    return {
        "status": "ok",
        "section": record.id,
        "written": written,
        **({} if pairs or delete else {"note": "No pairs or delete given: this is the listing; "
                                                "nothing was written."}),
        "changed": changed,
        **({"atlas_endpoint_regions": regions} if regions else {}),
        **({"outside_atlas": outside, "warning": _outside_words(outside)} if outside else {}),
        "deleted": deleted,
        "counts": _counts(rows),
        "landmarks": rows,
        **({"warp": report} if (report := geometry.warp_report(job.state, record)) else {}),
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


#: The automatic holds' lattice spacing, as a fraction of the section's longer side.
HOLD_SPACING = 1 / 7
#: A lattice point closer than this many spacings to a pair's section point is
#: dropped: the pair moves its own neighbourhood freely.
HOLD_CLEARANCE = 0.7
#: Fractions of a spacing the lattice is shifted by when a mesh with holds is
#: degenerate (a hold collinear with pair points), tried in turn.
HOLD_SHIFTS = (0.0, 0.11, 0.23, 0.37)
#: How a fold reply names a hold vertex.
HOLD_WORDS = "automatic hold"


def automatic_holds(
    pair_points: np.ndarray, size: tuple[int, int], shift: float = 0.0,
    footprint: np.ndarray | None = None,
) -> np.ndarray:
    """Section points (pixels) of the automatic holds: a regular lattice over
    the section frame, spaced :data:`HOLD_SPACING` of its longer side, without
    the points within :data:`HOLD_CLEARANCE` spacings of any of *pair_points*,
    and, given a *footprint* mask (rows, cols on the frame: the placed atlas
    footprint the deformable fit's fixed mask is), without the points off it.

    Each hold corresponds to itself (it keeps its spot where the baseline
    placement puts it); the pairs are never changed."""
    width, height = size
    spacing = max(width, height) * HOLD_SPACING
    columns, rows = max(1, round(width / spacing)), max(1, round(height / spacing))
    xs = (np.arange(columns) + 0.5) * width / columns - 0.5 + shift * spacing
    ys = (np.arange(rows) + 0.5) * height / rows - 0.5 + shift * spacing
    lattice = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    inside = (lattice[:, 0] < width - 0.5) & (lattice[:, 1] < height - 0.5)
    lattice = lattice[inside]
    if footprint is not None and len(lattice):
        rows, cols = np.rint(lattice[:, 1]).astype(int), np.rint(lattice[:, 0]).astype(int)
        lattice = lattice[footprint[rows.clip(0, height - 1), cols.clip(0, width - 1)]]
    points = np.asarray(pair_points, dtype=float).reshape(-1, 2)
    if not len(points) or not len(lattice):
        return lattice
    gap = np.hypot(*(lattice[:, None, :] - points[None, :, :]).transpose(2, 0, 1))
    return lattice[gap.min(axis=1) > HOLD_CLEARANCE * spacing]


def _vertex_name(index: int, names: Sequence[str]) -> str:
    """The name of mesh vertex *index* (the four fixed corners come first,
    then the pairs, then the holds): a pair id, or a hold marked by its number."""
    k = index - 4
    return names[k] if k < len(names) else f"{geometry.HOLD_PREFIX}{k - len(names)}"


def _degenerate_words(exc: DegenerateMesh, names: Sequence[str]) -> str:
    def ids(group: Sequence[int]) -> str:
        return ", ".join(
            "a fixed corner" if i < 4
            else HOLD_WORDS if i - 4 >= len(names) else names[i - 4]
            for i in group)

    groups = "; ".join(ids(group) for group in exc.vertices[:6])
    if "Repeated" in str(exc):
        return (f"Pairs with the same section point cannot be triangulated ({groups}). "
                "Move one section endpoint of each group, or delete one pair.")
    return (f"The section points of {groups} lie on one line, a zero-area triangle. "
            "Move one of those section endpoints off the line, or delete one pair.")


def _folds(mesh: Any, points: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """The warp's folded triangles: the share of the section frame they cover
    and their vertices (a pair id, or ``~K`` for the K-th automatic hold; a
    fixed corner vertex has none, so a triangle at the frame's edge lists
    fewer than three)."""
    triangles = [[_vertex_name(i, names) for i in sorted(tri) if i >= 4]
                 for tri in mesh.folded_triangles()]
    if not triangles:
        return {}
    return {"area_fraction": round(mesh.folded_fraction(points), 5), "triangles": triangles}


def _folds_reply(folds: dict[str, Any] | None) -> dict[str, Any]:
    """The reply's ``folds``: nothing when the warp has none."""
    if not folds:
        return {}
    triangles = folds["triangles"]
    ids = sorted({i for tri in triangles for i in tri if not i.startswith(geometry.HOLD_PREFIX)},
                 key=lambda s: (len(s), s))
    shown = [[HOLD_WORDS if i.startswith(geometry.HOLD_PREFIX) else i for i in tri]
             for tri in triangles[:MAX_FOLDS_LISTED]]
    with_holds = any(i.startswith(geometry.HOLD_PREFIX) for tri in triangles for i in tri)
    return {
        "folds": {
            "area_fraction": folds["area_fraction"],
            "pairs": ids,
            "triangles": shown,
            "triangle_count": len(triangles),
            "note": (
                "Folded triangles are applied as VisuAlign does (outlined in magenta on the "
                "picture). A fold means the section points of these pairs are ordered "
                "differently from their atlas points; fix those pairs' endpoints if it is wrong."
                + (" An automatic hold in a triangle is a spot kept where it was: the "
                   "fold comes from the pairs beside it, and a pair placed far from its "
                   "neighbours' atlas points drags that spot." if with_holds else "")
            ),
        }
    }


def _holds_reply(count: int) -> dict[str, Any]:
    """The reply's words on the automatic holds."""
    return {
        "automatic_holds": count,
        "holds_note": (
            f"{count} automatic hold{'s were' if count != 1 else ' was'} added between the "
            "pairs, over the placed atlas: spots that keep "
            "their current place, so a pair moves its neighbourhood up to the holds "
            "near it and the section between the pairs stays put. They are not pairs "
            "and cannot be edited; they are recomputed from the pairs on every application."
            if count else
            "No automatic holds were added: every spot of the hold lattice over the placed atlas "
            "is near a pair."
        ),
    }


def _require_applicable(job: Job, record: SliceState) -> list[dict[str, Any]]:
    """The section's pairs, refusing too few of them or any placed against another plane."""
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
    return pairs


def _mesh_with_holds(
    names: Sequence[str], source: np.ndarray, target: np.ndarray, size: tuple[int, int],
    footprint: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """The VisuAlign mesh of the pairs plus automatic holds, each hold
    corresponding to itself against the baseline (the pair targets are already
    in the baseline's section coordinates, so a hold's target is its own spot).
    Only this verb adds holds: the mesh is exact for any point set."""
    for shift in HOLD_SHIFTS:
        holds = automatic_holds(source, size, shift, footprint)
        try:
            mesh = visualign_mesh(
                np.vstack([source, holds]), np.vstack([target, holds]), size=size)
            return mesh, holds
        except DegenerateMesh as exc:
            bare = {i for group in exc.vertices for i in group if i >= 4 + len(names)}
            if not bare or shift == HOLD_SHIFTS[-1]:
                # The mesh's first four vertices are the fixed corners.
                raise Refused(
                    "LANDMARK_TRIANGULATION_FAILED",
                    pairs=sorted({names[i - 4] for group in exc.vertices
                                  for i in group if 4 <= i < 4 + len(names)}),
                    message=_degenerate_words(exc, names),
                ) from exc
    raise AssertionError("HOLD_SHIFTS is not empty")


def _landmark_reference(
    job: Job, record: SliceState, folder: Any, key: str, result: Any, numbers: dict[str, Any],
    revision: str, pairs: list[dict[str, Any]], folds: dict[str, Any],
    hold_fractions: list[Any], baseline_key: str | None, baseline_steps: list[Any],
    composed: bool,
) -> dict[str, Any]:
    """The deformation reference the section holds for a landmark application."""
    return {
        "key": key,
        "record": job.layout.relative(folder),
        "linear_key": deformation.linear_key(job.state, record),
        "summary": numbers,
        "inverse_source": result.inverse_source,
        "landmark_revision": revision,
        "landmark_pairs": {p["id"]: geometry.pair_key(p) for p in pairs},
        "landmark_folds": folds,
        "landmark_holds": hold_fractions,
        "landmark_baseline_key": baseline_key,
        "landmark_baseline_steps": baseline_steps,
        "steps": [
            *baseline_steps,
            {
                "method": "landmark_triangulation",
                "start": "current" if composed else "linear",
                "landmark_revision": revision,
                "landmark_count": len(pairs),
            },
        ],
    }


def _unchanged_reply(record: SliceState, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The reply when the section's warp is already the one these pairs make."""
    held = record.deformation or {}
    return {
        "status": "ok",
        "section": record.id,
        "written": False,
        "landmark_count": len(pairs),
        "built_from": [p["id"] for p in pairs],
        **_holds_reply(len(held.get("landmark_holds", []))),
        **held.get("summary", {}),
        DISPLACEMENT_NOTE_KEY: DISPLACEMENT_NOTE,
        **_folds_reply(held.get("landmark_folds")),
    }


def landmark_triangulation(job: Job, workspace: Workspace, section: str) -> dict[str, Any]:
    """Apply the complete set, replacing the preceding landmark application."""
    record = _section(job, section)
    pairs = _require_applicable(job, record)
    revision = digest(pairs)
    held = record.deformation or {}
    if held.get("landmark_revision") == revision:
        return _unchanged_reply(record, pairs)
    started = time.monotonic()
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
        try:
            current = job.deformations.current(job.state, record)
        except ValueError:  # its record.json does not load (the error names it)
            logger.warning("Applied deformation of %s unreadable", record.id, exc_info=True)
            current = None
        if held and current is None:
            raise Refused(
                "LANDMARK_WARP_UNAVAILABLE",
                message="The current deformation cannot be read; undo it first.",
            )
        replacing = current is not None and current.engine.get("name") == "landmark_triangulation"
        previous = current.parent if replacing and current is not None else current
        baseline_key = held.get("landmark_baseline_key") if replacing else held.get("key")
        baseline_steps = copy.deepcopy(
            held["landmark_baseline_steps"] if replacing else held.get("steps", [])
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
        prepared = prepare_warp(workspace.atlas, grid.placement, size, deformation.DETAIL_LEVEL,
                                previous=previous)
        mesh, holds = _mesh_with_holds(names, source, target, size, prepared.footprint)
        yy, xx = np.indices((size[1], size[0]), dtype=float)
        points = np.stack([xx, yy], -1)
        mm = grid.placement.section_mm_per_px
        forward, inverse = mesh.map(points), mesh.inverse(points)
        folds = _folds(mesh, points, names)
        if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
            raise ValueError("The mesh does not cover the full section frame")
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
                    # vertices: 4 fixed corners, then the pairs, then the holds
                    "pair_count": len(names),
                    "hold_count": len(holds),
                },
                runtime_s=time.monotonic() - started,
                engine_version=scipy.__version__,
            ),
            landmark_warp=True,
        )
        result.engine["name"] = "landmark_triangulation"
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
        "automatic_holds_xy": holds.tolist(),
        "hold_spacing_px": max(size) * HOLD_SPACING,
        "hold_clearance_spacings": HOLD_CLEARANCE,
    }
    job.deformations.put(key, result)
    folder = job.deformations.save(record.id, key, result)
    numbers = deformation.summary(result, current)
    before = job.snapshot()
    # The holds in the section file's edge fractions, as the pairs' section
    # points are, so a picture of any orientation can place them.
    hold_fractions = geometry.transform(
        geometry.orientation(record), (holds + 0.5) / np.asarray(size)).tolist()
    record.deformation = _landmark_reference(
        job, record, folder, key, result, numbers, revision, pairs, folds, hold_fractions,
        baseline_key, baseline_steps, previous is not None)
    job.commit(before)
    return {
        "status": "ok",
        "section": record.id,
        "written": True,
        "landmark_count": len(pairs),
        "built_from": [p["id"] for p in pairs],
        "composed": previous is not None,
        "replaced": replacing,
        **_holds_reply(len(holds)),
        **numbers,
        DISPLACEMENT_NOTE_KEY: DISPLACEMENT_NOTE,
        **_folds_reply(folds),
    }
