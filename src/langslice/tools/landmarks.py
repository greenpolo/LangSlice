"""Edit persistent corresponding pairs; explicitly apply their triangulation."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.atlas.tree import structure_table
from langslice.core.fits import landmarks as geometry
from langslice.core.geometry.pixel_frames import source_frame
from langslice.core.jsonable import JsonObject
from langslice.core.pictures.drawn import collecting
from langslice.core.pictures.landmarks import pair_picture
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.landmarks import Endpoint, Landmarks, Pair
from langslice.tools.look import Looked, save_pictures
from langslice.tools.refusal import check_frozen, unknown_sections
from langslice.tools.run import Run, pictured

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


#: ``delete=[DELETE_ALL]`` removes every pair of the section.
DELETE_ALL = "all"


def landmark_section(job: Job, section: str) -> Section:
    """Return the section a landmark call names.

    Parameters
    ----------
    job
        The job whose stack holds the section.
    section
        The section's filename, or its stem.

    Returns
    -------
    Section
        The section, which may take pairs and a landmark warp.

    Raises
    ------
    Refused
        ``UNKNOWN_SLICE_IDS``; ``SECTION_BUSY`` or ``FROZEN`` (its
        deformation), as :func:`langslice.tools.refusal.check_frozen`.
    """
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
    job: Job, record: Section, point: dict[str, Any], role: str, label: str = "the pair"
) -> Endpoint:
    """The saved endpoint of *point* (``picture``, ``x``, ``y``) in *role*;
    *label* says which pair of the call it is, for the refusals."""
    here = f"{label}, {role} endpoint"
    if not isinstance(point, dict):
        raise Refused(
            RefusalCode.BAD_ARGS, message=f"{here}: an endpoint is an object with picture, x and y."
        )
    try:
        seq, x, y = int(point["picture"]), float(point["x"]), float(point["y"])
        given = (x, y)
    except (KeyError, TypeError, ValueError):
        raise Refused(
            RefusalCode.BAD_ARGS, message=f"{here}: an endpoint needs picture, x and y in pixels."
        ) from None
    if point.get("picture") != seq or isinstance(point.get("picture"), bool):
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"{here}: picture is a whole saved picture number.")
    if set(point) - {"picture", "x", "y"}:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"{here}: an endpoint accepts picture, x and y only.")
    picture = job.views.lookup(seq)
    if picture is None:
        raise Refused(
            RefusalCode.UNKNOWN_PICTURE,
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
            RefusalCode.BAD_LANDMARK_PICTURE,
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
            RefusalCode.BAD_LANDMARK_POINT,
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
                RefusalCode.STALE_LANDMARK_PLANE,
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
                    RefusalCode.LANDMARK_WARP_UNAVAILABLE,
                    picture=seq,
                    message=f"{here}: the warped view in picture {seq} needs its stored "
                    "inverse mapping, which is gone; draw the section again with look.",
                )
        canonical = geometry.section_point(panel, [x, y], warp)
        limits = np.ones(2)
    if not np.isfinite(canonical).all() or np.any(canonical < 0) or np.any(canonical >= limits):
        raise Refused(
            RefusalCode.BAD_LANDMARK_POINT,
            picture=seq,
            message=(
                f"{here}: the point {where} lies outside the {role} image; "
                f"{_content(seq, [panel], frame)}. Coordinates are never clamped."
            ),
        )
    canonical_x, canonical_y = cast("list[float]", canonical.tolist())
    # the picture's record and panel, read back from its saved recipe
    context = copy.deepcopy(cast("JsonObject", held))
    panel_facts = copy.deepcopy(cast("JsonObject", panel))
    return Endpoint(
        canonical_xy=(canonical_x, canonical_y),
        picture=seq,
        pixel_xy=given,
        registration_context=context,
        geometry=panel_facts,
    )


def _atlas_regions(
    workspace: Workspace, plane: dict[str, Any], items: Sequence[Pair]
) -> dict[str, str]:
    """The region drawn at each of *items*' atlas endpoint on the atlas *plane*
    (``plane_context``), as ``ACRONYM (name)`` by pair id."""
    from langslice.tools.atlas import label_at, plane_annotation, region_words

    if not items:
        return {}
    annotation = plane_annotation(workspace, plane)
    structures = structure_table(workspace.atlas)
    return {item.id: region_words(structures, label_at(annotation,
                                                       item.atlas_native_xy) or 0)
            for item in items}


#: How deep inside the atlas (native pixels from its outline) the point a
#: refusal proposes lies, so that the rounding of picture pixels keeps it inside.
INSIDE_MARGIN_PX = 1.5


def _refuse_outside(job: Job, workspace: Workspace, plane: dict[str, Any],
                    placed: Sequence[tuple[str, Pair]]) -> None:
    """Refuse the call (``ATLAS_ENDPOINT_OUTSIDE``, nothing written) when an
    atlas endpoint of *placed* (``(label, pair)``) lies outside the atlas on
    *plane*, naming for each the nearest point inside it in the numbers it
    was given in (a zoom's: the whole picture's)."""
    from scipy.ndimage import distance_transform_edt

    from langslice.tools.atlas import label_at, plane_annotation

    if not placed:
        return
    annotation = plane_annotation(workspace, plane)
    outside = [(label, pair) for label, pair in placed
               if not label_at(annotation, pair.atlas_native_xy)]
    if not outside:
        return
    inside = annotation != 0
    depth = np.asarray(distance_transform_edt(inside))
    deep = depth >= INSIDE_MARGIN_PX
    target = deep if deep.any() else inside
    # The index of the nearest target pixel, per pixel: (2, rows, cols).
    nearest = np.asarray(distance_transform_edt(~target, return_distances=False,
                                                return_indices=True))
    sentences = []
    for label, pair in outside:
        endpoint = pair.atlas
        seq = endpoint.picture
        x, y = endpoint.pixel_xy
        sentence = (f"{label}, atlas endpoint: the point ({x:g}, {y:g}) in picture {seq} is "
                    "outside the atlas, where no brain is drawn, so it would pair empty atlas "
                    "with the tissue")
        c, r = (int(np.clip(np.rint(v), 0, n - 1)) for v, n in zip(
            pair.atlas_native_xy, (annotation.shape[1], annotation.shape[0]), strict=True))
        if target.any():
            row, col = int(nearest[0][r, c]), int(nearest[1][r, c])
            matrix = np.asarray(endpoint.geometry["pixel_to_native"], dtype=float)
            full = np.vstack([matrix[:2], [0.0, 0.0, 1.0]])
            px, py = geometry.transform(np.linalg.inv(full)[:2], [float(col), float(row)])
            picture = job.views.lookup(seq)
            frame = source_frame((picture.recipe or {}) if picture is not None else None)
            if frame is not None:
                px, py = (float(v) for v in frame.to_source([px, py]))
            sentence += (f"; the nearest point inside it is ({px:.0f}, {py:.0f}) in picture "
                         f"{seq}")
        sentences.append(sentence + ".")
    raise Refused(
        RefusalCode.ATLAS_ENDPOINT_OUTSIDE,
        pairs=[label for label, _pair in outside],
        message=" ".join(sentences) + " Nothing was written. For a pair on "
        "the outline, put the atlas endpoint on the drawn border or just "
        "inside it.")


def _listing(job: Job, record: Section) -> list[dict[str, Any]]:
    """One compact record per pair: id, the endpoints as the call gave them
    (picture, x, y) and the state (``applied``, ``moved``, ``pending`` or
    ``stale``)."""
    states = geometry.pair_states(job.state, record)
    rows: list[dict[str, Any]] = []
    for pair in geometry.section_pairs(record):
        row: dict[str, Any] = {"id": pair.id}
        for role, endpoint in (("atlas", pair.atlas), ("section", pair.section)):
            x, y = endpoint.pixel_xy
            row[role] = {"picture": endpoint.picture, "x": round(x, 1),
                         "y": round(y, 1)}
        row["state"] = states[pair.id]
        rows.append(row)
    return rows


def _edited_pair(
    job: Job, section: Section, previous: Pair | None, pair_id: str,
    edit: JsonObject, label: str,
) -> Pair:
    """Return pair *pair_id* after *edit* (a new pair without *previous*).

    Each endpoint the edit gives is read off its picture and the others
    kept; the atlas plane is the section's now when the atlas end moved.
    """
    atlas_end = None if previous is None else previous.atlas
    section_end = None if previous is None else previous.section
    # an endpoint as the call gave it, checked by _endpoint
    if "atlas" in edit:
        given = cast("dict[str, Any]", edit["atlas"])
        atlas_end = _endpoint(job, section, given, "atlas", label)
    if "section" in edit:
        given = cast("dict[str, Any]", edit["section"])
        section_end = _endpoint(job, section, given, "section", label)
    # a new pair gives both endpoints (refused above otherwise)
    assert atlas_end is not None and section_end is not None
    plane: JsonObject = {} if previous is None else previous.plane
    if "atlas" in edit:
        plane = geometry.plane_context(job.state, section)
    item = Pair(
        id=pair_id, atlas=atlas_end, atlas_native_xy=atlas_end.canonical_xy,
        section=section_end, section_fraction_xy=section_end.canonical_xy,
        plane=plane, stale=False)
    if previous is not None:
        item = replace(
            previous, atlas=item.atlas, atlas_native_xy=item.atlas_native_xy,
            section=item.section, section_fraction_xy=item.section_fraction_xy,
            plane=plane)
    return replace(item, stale=geometry.stale(job.state, section, item))


@dataclass(frozen=True)
class LandmarksEdited:
    """What :func:`set_landmarks` wrote, and the pairs as they are now."""

    section: str
    written: bool
    #: The call named no pairs and no delete: it only lists them.
    listing: bool
    #: The pairs added or edited, by id, in call order.
    changed: tuple[str, ...]
    deleted: tuple[str, ...]
    #: The region each new atlas endpoint lies in, by pair id.
    atlas_endpoint_regions: dict[str, str]
    #: One compact record per pair: id, endpoints, state.
    landmarks: Sequence[JsonObject]
    #: Which pairs the current warp was built from (empty: no warp).
    warp: JsonObject

    def reply(self) -> dict[str, Any]:
        """Return the tool's reply: the write, the pairs and their states.

        Returns
        -------
        dict
            ``status``, ``section``, ``written``, a ``note`` for a listing,
            ``changed``, ``atlas_endpoint_regions``, ``deleted``, ``counts``
            (pairs per state), ``landmarks`` and ``warp``; the empty ones
            left out.
        """
        reply: dict[str, Any] = {
            "status": "ok", "section": self.section, "written": self.written}
        if self.listing:
            reply["note"] = ("No pairs or delete given: this is the listing; "
                             "nothing was written.")
        reply["changed"] = list(self.changed)
        if self.atlas_endpoint_regions:
            reply["atlas_endpoint_regions"] = self.atlas_endpoint_regions
        reply["deleted"] = list(self.deleted)
        states = [row["state"] for row in self.landmarks]
        reply["counts"] = {"total": len(states)} | {
            kind: states.count(kind) for kind in _PAIR_STATES}
        reply["landmarks"] = list(self.landmarks)
        if self.warp:
            reply["warp"] = self.warp
        return reply


#: The states a pair can be in, as the reply counts them.
_PAIR_STATES = ("applied", "moved", "pending", "stale")


def set_landmarks(
    job: Job,
    workspace: Workspace,
    section: str,
    pairs: Sequence[dict[str, Any]] = (),
    delete: Sequence[str] = (),
) -> LandmarksEdited:
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
            RefusalCode.BAD_ARGS,
            message="pairs is a list of objects; delete is a list of pair IDs.",
        )
    record = landmark_section(job, section)
    saved = record.landmarks or Landmarks()
    next_id = saved.next_id
    by_id = {pair.id: pair for pair in saved.pairs}
    if len(set(delete)) != len(delete) or len({p.get("id") for p in pairs if p.get("id")}) != sum(
        bool(p.get("id")) for p in pairs
    ):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="A pair ID may be edited or deleted only once per batch.")
    changed: list[str] = []
    placed: list[Pair] = []
    labels: list[str] = []
    deleted = list(by_id) if DELETE_ALL in delete else list(delete)
    if DELETE_ALL in delete and len(delete) > 1:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f'delete ["{DELETE_ALL}"] stands alone.')
    for pair_id in deleted:
        if pair_id not in by_id:
            raise Refused(
                RefusalCode.UNKNOWN_LANDMARK,
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
                RefusalCode.UNKNOWN_LANDMARK,
                id=pair_id,
                message=(
                    f"{label}: {record.id} has no pair {pair_id}; its pairs are "
                    f"{', '.join(by_id) or 'none'}. Leave id out to add a pair."
                ),
            )
        if set(edit) - {"id", "atlas", "section"}:
            raise Refused(
                RefusalCode.BAD_ARGS,
                message=f"{label}: pair entries accept id, atlas and section only.",
            )
        if not any(role in edit for role in ("atlas", "section")):
            raise Refused(RefusalCode.BAD_ARGS,
                          message=f"{label}: an edit needs at least one endpoint.")
        previous = by_id.get(pair_id)
        if not previous and (not edit.get("atlas") or not edit.get("section")):
            raise Refused(
                RefusalCode.BAD_ARGS,
                message=f"{label}: a new pair needs explicit atlas and section endpoints.",
            )
        if not pair_id:
            pair_id = f"L{next_id}"
            next_id += 1
        item = _edited_pair(job, record, previous, pair_id, edit, label)
        if "atlas" in edit:
            placed.append(item)
            labels.append(label)
        by_id[pair_id] = item
        changed.append(pair_id)
    _refuse_outside(job, workspace, geometry.plane_context(job.state, record),
                    list(zip(labels, placed, strict=True)))
    current = Landmarks(next_id=next_id, pairs=tuple(by_id.values()))
    before = job.snapshot()
    written = bool(pairs or delete) and record.landmarks != current
    if written:
        record.landmarks = current
        job.commit(before)
    return LandmarksEdited(
        section=record.id, written=written, listing=not (pairs or delete),
        changed=tuple(changed), deleted=tuple(deleted),
        atlas_endpoint_regions=_atlas_regions(
            workspace, geometry.plane_context(job.state, record), placed),
        landmarks=_listing(job, record),
        warp=geometry.warp_report(job.state, record))


# --- the tool ----------------------------------------------------------------

#: The most pairs one call pictures (one picture per pair).
MAX_PAIR_PICTURES = 4


def set_landmarks_tool(
    run: Run,
    section: str,
    pairs: list[dict[str, Any]],
    delete: list[str],
    *,
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``set_landmarks`` tool: write the pairs, answer, picture.

    With pictures, a call takes at most :data:`MAX_PAIR_PICTURES` pairs
    (``TOO_MANY_LANDMARK_VIEWS``, nothing written) and answers with one
    picture per pair it added or edited. A picture that fails leaves the
    write standing (``render_failed``).

    Parameters
    ----------
    run
        The run the tool belongs to.
    section, pairs, delete
        The call's arguments, as :func:`set_landmarks` takes them.
    view
        Whether to answer with the pairs' pictures.

    Returns
    -------
    dict
        The reply (:meth:`LandmarksEdited.reply`), with its pictures.
    """
    job, workspace = run.job, run.workspace
    if run.shows(view) and len(pairs) > MAX_PAIR_PICTURES:
        raise Refused(
            RefusalCode.TOO_MANY_LANDMARK_VIEWS,
            f"You sent {len(pairs)} pairs with view true, and a call draws one "
            "picture per pair for at most four. Nothing was written. Send "
            f"the same {len(pairs)} pairs again with view false (no pair "
            "pictures; the pairs are listed in the reply), or send them in "
            "calls of four or fewer.")
    edited = set_landmarks(job, workspace, section, pairs, delete)
    result = edited.reply()
    if not run.shows(view) or not edited.changed:
        return result

    record = job.state.by_id(edited.section)
    assert record is not None
    saved = () if record.landmarks is None else record.landmarks.pairs
    try:
        with collecting() as notes:
            pictures = [
                pair_picture(workspace, job.state, record, pair,
                             store=job.deformations)
                for pair in saved if pair.id in edited.changed]
        entries = save_pictures(
            job, workspace, "set_landmarks", pictures, notes, {})
        looked = Looked(pictures=[p.image for p in pictures], entries=entries)
        return pictured(result, looked)
    except Exception as exc:  # noqa: BLE001 - the write stands
        result["render_failed"] = str(exc)
        return result
