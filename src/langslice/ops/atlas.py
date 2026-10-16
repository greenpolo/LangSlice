"""Atlas lookups: the region hierarchy, searched like text. Reads, never writes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core import look as looks
from langslice.core import zoom as zooms
from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.atlas.tree import ancestry, path, structure_table
from langslice.core.atlas_fetch import atlas_section
from langslice.core.atlas_grep import GREP_ATLAS_LIMIT, grep_structures, plane_structure_ids
from langslice.core.captions import angles_label, caption, view_angles_label
from langslice.core.display import (
    atlas_caption,
    default_options,
    framed_atlas,
    regions_in_plane,
)
from langslice.core.layers import collecting, note
from langslice.core.positioning import ATLAS_UPSAMPLE
from langslice.core.sizes import picture_edge
from langslice.core.state import Angles, StackState
from langslice.ops.look import not_shown, save_pictures, split_shown
from langslice.ops.refusal import Refused, sides_refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: The recipe renderer of :func:`grep_atlas_view`'s pictures.
REGIONS_RENDERER = "grep_atlas_view"


#: Points one ``grep_atlas`` call looks up (as many as it lists region rows).
MAX_POINTS = GREP_ATLAS_LIMIT
#: Containing regions named per point, nearest first (the root is left out).
MAX_CHAIN = 6


def grep_atlas(job: Job, workspace: Workspace, query: str = "", section: object = "",
               points: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Regions whose acronym or name matches *query*
    (:func:`langslice.core.atlas_grep.grep_structures`).

    With *section* (the filename of a section with a
    position), each row says whether the region or a descendant appears in
    the atlas plane at that placement; a section without a position gets a
    ``note`` instead. Returns ``query``, ``matches`` (the total), ``rows``
    (at most :data:`~langslice.core.atlas_grep.GREP_ATLAS_LIMIT`), ``more``
    when some were left out. Refused: ``BAD_ARGS`` (empty query),
    ``NO_STRUCTURES``, ``UNKNOWN_SLICE_IDS``.

    With *points* (``{picture, x, y}`` of saved pictures; x, y as everywhere:
    on a zoom, the numbers its grid prints), ``points`` in the result says, per
    point, the region drawn there and the regions containing it
    (:func:`region_at_points`); *query* may then be empty.
    """
    text = str(query).strip()
    if not text and not points:
        raise Refused("BAD_ARGS", message="Empty query: give a query, points, or both.")
    found = region_at_points(job, workspace, points) if points else None
    if not text:
        return {"points": found}
    result = _grep_regions(job, workspace, text, section)
    if found is not None:
        result["points"] = found
    return result


def _grep_regions(job: Job, workspace: Workspace, text: str, section: object) -> dict[str, Any]:
    entries = list(structure_table(workspace.atlas).values())
    if not entries:
        raise Refused("NO_STRUCTURES", message="This atlas has no region hierarchy.")
    state = job.state
    present: set[int] | None = None
    note = ""
    if section != "":
        record = state.resolve(section)
        if record is None:
            raise unknown_sections(state, [section])
        if record.position_mm is None:
            note = f"{record.id} has no position yet, so in_section is omitted."
        else:
            present = plane_structure_ids(state, workspace, record.position_mm,
                                          record.angles)
    rows, total = grep_structures(entries, text, present, limit=GREP_ATLAS_LIMIT)
    result: dict[str, Any] = {"query": text, "matches": total, "rows": rows}
    if total > len(rows):
        result["more"] = total - len(rows)
    if note:
        result["note"] = note
    return result


def _label_layer(job: Job, record: Any) -> Any:
    """The atlas ids a saved picture drew, per picture pixel (its ``labels.tif``),
    or None when it saved none."""
    import tifffile

    if not record.layers or not record.path:
        return None
    path = job.layout.resolve(record.path) / "labels.tif"
    try:
        return tifffile.imread(path)
    except (OSError, ValueError):
        return None


def plane_annotation(workspace: Workspace, plane: dict[str, Any]) -> np.ndarray:
    """The annotation slice of a section's atlas *plane* (``position_mm``,
    ``plane``, optional ``angles``: ``landmarks.plane_context``), in atlas-native pixels."""
    from langslice.core.atlas.render import annotation_slice

    angles = plane.get("angles") or [0, 0]
    return annotation_slice(workspace.atlas, float(plane["position_mm"]),
                            plane=plane["plane"], pitch_deg=float(angles[0]),
                            yaw_deg=float(angles[1]))


def label_at(annotation: np.ndarray, native_xy: Sequence[float] | np.ndarray) -> int | None:
    """The annotation id at atlas-native point *native_xy*; None off the slice."""
    c, r = int(np.rint(native_xy[0])), int(np.rint(native_xy[1]))
    if 0 <= r < annotation.shape[0] and 0 <= c < annotation.shape[1]:
        return int(annotation[r, c])
    return None


def region_words(structures: dict[int, Any], label: int) -> str:
    """``ACRONYM (name)`` of annotation id *label*."""
    entry = structures.get(label)
    return f"{entry['acronym']} ({entry['name']})" if entry is not None else f"region {label}"


def _chain(table: dict[int, Any], region_id: int) -> list[str]:
    """The regions containing *region_id*, nearest first, as ``acronym (name)``."""
    entry = table.get(region_id)
    if entry is None:
        return []
    out = []
    for parent in reversed(ancestry(entry)):
        held = table.get(parent)
        if held is None or len(path(held)) <= 1:
            continue  # the root says nothing
        out.append(region_words(table, parent))
    return out[:MAX_CHAIN]


def region_at_points(job: Job, workspace: Workspace, points: Sequence[dict[str, Any]]
                     ) -> list[dict[str, Any]]:
    """The atlas region drawn at each of *points* (``{picture, x, y}``) of saved
    pictures, one row each: ``region`` (acronym, name, id) and ``within`` (the
    regions containing it, nearest first), or ``region`` null with the reason
    (outside the atlas, or outside the picture's image). Reads what the
    picture drew: its saved ``labels.tif`` for a section on its atlas (the
    drawn atlas, whatever deformation the section carries), the section's atlas
    plane for an atlas picture drawn for a section, the source picture of a crop.
    Refused: ``BAD_ARGS``, ``UNKNOWN_PICTURE``, ``NO_STRUCTURES``,
    ``NO_ATLAS_IN_PICTURE`` (a picture that draws no atlas)."""
    from langslice.core.picture_geometry import source_frame

    structures = structure_table(workspace.atlas)
    if not structures:
        raise Refused("NO_STRUCTURES", message="This atlas has no region hierarchy.")
    if (isinstance(points, (str, bytes)) or not isinstance(points, Sequence)
            or any(not isinstance(p, dict) for p in points)):
        raise Refused("BAD_ARGS", message="points is a list of {picture, x, y} objects.")
    if len(points) > MAX_POINTS:
        raise Refused("BAD_ARGS", message=f"At most {MAX_POINTS} points per call; you sent "
                      f"{len(points)}. Send them in calls of {MAX_POINTS} or fewer.")
    job.views.flush()
    layers: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        here = f"points[{index}]"
        try:
            seq, x, y = int(point["picture"]), float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            raise Refused("BAD_ARGS", message=f"{here}: a point needs picture (a saved picture "
                          "number), x and y in pixels.") from None
        if set(point) - {"picture", "x", "y"} or isinstance(point["picture"], bool):
            raise Refused("BAD_ARGS", message=f"{here}: a point is picture, x and y only.")
        record = job.views.lookup(seq)
        if record is None:
            raise Refused("UNKNOWN_PICTURE", picture=seq,
                          message=f"{here}: there is no saved picture {seq}. Use the picture "
                          "number a look or zoom reply gave.")
        recipe = record.recipe or {}
        row: dict[str, Any] = {"picture": seq, "x": x, "y": y}
        if recipe.get("renderer") == zooms.CROP_RENDERER:
            # A crop of a saved image: its grid numbers are the source picture's pixels.
            source = job.views.lookup(int(recipe.get("source") or 0))
            if source is not None:
                record, recipe = source, source.recipe or {}
        frame = source_frame(recipe)
        px, py = (float(v) for v in frame.to_picture([x, y])) if frame is not None else (x, y)
        panels = recipe.get("endpoint_panels") or []
        if record.seq not in layers:
            layers[record.seq] = _label_layer(job, record)
        labels = layers[record.seq]
        label = None
        if labels is not None:
            r, c = int(np.rint(py)), int(np.rint(px))
            on_image = 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] and (
                not panels or any(b[0] <= px < b[2] and b[1] <= py < b[3]
                                  for b in (p["bounds"] for p in panels)))
            label = int(labels[r, c]) if on_image else None
            if not on_image:
                row["region"] = None
                row["note"] = (f"({x:g}, {y:g}) is outside the image content of picture {seq} "
                               "(a caption, a margin or off the picture).")
        else:
            label = _plane_label(workspace, recipe, px, py, seq, record.mode)
        if "region" not in row:
            entry = structures.get(label) if label else None
            if not label:
                row["region"] = None
                row["note"] = "Outside the atlas: no brain is drawn there."
            elif entry is None:
                row["region"] = {"id": label}
            else:
                row["region"] = {"acronym": str(entry["acronym"]), "name": str(entry["name"]),
                                 "id": int(label)}
                row["within"] = _chain(structures, int(label))
        rows.append(row)
    return rows


def _plane_label(workspace: Workspace, recipe: dict[str, Any], px: float, py: float,
                 seq: int, mode: str | None) -> int | None:
    """The annotation id at picture pixel (*px*, *py*) of an atlas picture drawn
    for a section (its endpoint panel gives the plane's pixels); refused for a
    picture that draws no atlas or no section's plane."""
    from langslice.core.landmarks import transform

    panels = [p for p in recipe.get("endpoint_panels") or [] if p.get("role") == "atlas"]
    held = (recipe.get("registration_context") or {}).get("plane")
    if not panels or not held:
        sectionless = mode == "section"
        raise Refused(
            "NO_ATLAS_IN_PICTURE", picture=seq,
            message=(
                f"Picture {seq} shows the section alone, with no atlas drawn on it, so no "
                "region is drawn at its points. Use an overlay picture of the section (its "
                "atlas is drawn on it), or an atlas picture drawn for the section."
                if sectionless else
                f"Picture {seq} has no atlas plane tied to a section, so no region is drawn "
                "at its points. Use an overlay picture of the section, or an atlas picture "
                "drawn for the section."))
    panel = panels[0]
    native = transform(panel["pixel_to_native"], [px, py])
    label = label_at(plane_annotation(workspace, held), native)
    inside = all(b[0] <= px < b[2] and b[1] <= py < b[3] for b in [panel["bounds"]])
    return label if inside else None


# --- grep_atlas_view ---------------------------------------------------------------------


@dataclass(frozen=True)
class RegionsView:
    """What :func:`grep_atlas_view` showed (the shape of
    :class:`langslice.ops.look.Looked`, with the regions)."""

    pictures: list[Image.Image] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    not_shown: list[dict[str, Any]] = field(default_factory=list)
    #: The regions highlighted, as named.
    regions: list[str] = field(default_factory=list)
    #: ``{"<mm:.2f>": [regions]}``: the named regions with no pixel in that plane.
    regions_not_in_plane: dict[str, list[str]] = field(default_factory=dict)
    #: ``(asked, used)`` positions moved into the atlas range.
    clamped: list[tuple[float, float]] = field(default_factory=list)


def _resolved(workspace: Workspace, state: StackState, regions: Any,
              ) -> list[tuple[str, frozenset[int]]]:
    """``(name as asked, ids with descendants)`` per region. Refused:
    ``BAD_ARGS``, ``NO_STRUCTURES``, ``UNKNOWN_REGIONS``, ``NO_SIDES``."""
    from langslice.core.atlas.sides import has_sides
    from langslice.core.damage import normalized_entries
    from langslice.core.deformable.plane_images import resolve_entries

    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise Refused("BAD_ARGS", message="regions is a list of acronyms, names or ids.")
    try:
        names = normalized_entries(str(entry) for entry in regions)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    if not names:
        raise Refused("BAD_ARGS", message="Name at least one region.")
    if not structure_table(workspace.atlas):
        raise Refused("NO_STRUCTURES", message="This atlas has no region hierarchy.")
    try:
        found = resolve_entries(workspace.atlas, names)
    except ValueError as exc:
        raise Refused("UNKNOWN_REGIONS", message=str(exc)) from exc
    if state.plane == "sagittal" and has_sides(names):
        raise Refused("NO_SIDES", message="A sagittal section lies within one hemisphere, so a "
                      "region cannot be limited to one side.")
    return [(name, ids) for name, (_region, _side, ids) in zip(names, found, strict=True)]


def _zoom_pixels(window: Sequence[float], size: tuple[float, float]) -> tuple[float, ...]:
    if not window:
        return ()
    return (window[0] * size[0], window[1] * size[1], window[2] * size[0], window[3] * size[1])


def _region_picture(
    workspace: Workspace, state: StackState, regions: list[tuple[str, frozenset[int]]],
    position_mm: float, long_edge: int, angles: Angles, *, window: Sequence[float] = (),
    base: tuple[int, int] | None = None, base_um: float | None = None,
) -> looks.LookPicture:
    """The atlas template at *position_mm* and the stack's view *angles*, the
    only *regions* outlined; at *window* (fractions of the
    unzoomed picture) when given."""
    options = replace(default_options("template", atlas_channels=("template",),
                                      long_edge=long_edge), regions=tuple(regions))
    voxel = atlas_um_per_px(workspace.atlas)
    if base is None or base_um is None:
        whole = framed_atlas(workspace, state, position_mm, options, angles=angles)
        native = atlas_section(workspace, state, position_mm, frame=True, angles=angles)
        base, base_um = whole.size, voxel * max(native.size) / max(whole.size)
    tag = ""
    if window:
        extent = max((window[2] - window[0]) * base[0], (window[3] - window[1]) * base[1])
        um = max(base_um * extent / long_edge, voxel / ATLAS_UPSAMPLE)
        at_um = (base[0] * base_um / um, base[1] * base_um / um)
        options = replace(options, zoom=_zoom_pixels(window, at_um))
        picture = framed_atlas(workspace, state, position_mm, options, angles=angles,
                               um_per_px=um)
        tag = f"zoom x{max(picture.size) / max(extent, 1e-9):.1f}  "
    else:
        picture = framed_atlas(workspace, state, position_mm, options, angles=angles)
        um = base_um
    names = ", ".join(name for name, _ids in regions)
    median = state.drawn_at_median(angles)
    plane = (view_angles_label(angles, median=True) if median
             else f"{angles_label(angles)} (the stack's cutting angles)")
    border_word = picture.info.get("border_word")
    text = (f"atlas at {position_mm:.2f} mm{plane}; "
            f"{um:.1f} um/px; regions highlighted: {names}"
            + (f"; borders {border_word}" if border_word else ""))
    image = caption(picture, tag + atlas_caption(state, position_mm, options, angles=angles,
                                                  median=median))
    args = {"mode": "atlas", "regions": [name for name, _ids in regions],
            "positions_mm": [float(position_mm)], "long_edge": int(long_edge),
            "zoom": [float(v) for v in window]}
    recipe = {"renderer": REGIONS_RENDERER, "args": args,
              "state": looks.snapshot(state, [], view_angles=True),
              "base": [int(base[0]), int(base[1])],
              "shown": [int(picture.width), int(picture.height)], "um_per_px": round(um, 4)}
    note(image, mode="atlas", recipe=recipe, caption=text,
         extra={"position_mm": float(position_mm)})
    return looks.LookPicture(image=image, caption=text, recipe=recipe, sections=(), mode="atlas",
                             um_per_px=um, extra={"position_mm": float(position_mm)})


def _redraw_regions(
    recipe: dict[str, Any], window: Sequence[float], workspace: Workspace, state: StackState,
    store: Any, zoom_edge: int | None,
) -> looks.LookPicture:
    """:func:`langslice.core.zoom.redraw`'s redrawer for a ``grep_atlas_view``
    recipe: the picture drawn again at *window*, from the stack as it was."""
    args = recipe["args"]
    held = recipe.get("state") or {}
    stale = bool(held) and looks.changed_since(state, held)
    drawn = looks.restored(state, held) if stale else state
    angles = held.get("view_angles")
    base, shown, um = recipe.get("base"), recipe.get("shown"), recipe.get("um_per_px")
    base_um = None
    if um is not None and base and shown:
        earlier = args.get("zoom") or []
        extent = (max((earlier[2] - earlier[0]) * base[0], (earlier[3] - earlier[1]) * base[1])
                  if earlier else max(shown))
        base_um = float(um) * extent / max(max(shown), 1)
    regions = _resolved(workspace, drawn, args["regions"])
    position = float(args["positions_mm"][0])
    picture = _region_picture(
        workspace, drawn, regions, position, int(args.get("long_edge") or picture_edge(workspace)),
        (float(angles[0]), float(angles[1])) if angles else drawn.view_angles, window=window,
        base=(int(base[0]), int(base[1])) if base else None, base_um=base_um)
    return replace(picture, stale=stale)


zooms.RENDERERS[REGIONS_RENDERER] = _redraw_regions


def grep_atlas_view(
    job: Job, workspace: Workspace, regions: Sequence[str], positions_mm: Sequence[float],
) -> RegionsView:
    """The atlas template at each of *positions_mm* (clamped into the atlas
    range), at the stack's cutting angles (``StackState.view_angles``), with
    only the borders of *regions* drawn.
    A region may name a side (``"CTX:left"``, as the section is displayed).

    Each picture is saved like a ``look`` picture (recipe, caption, number),
    so it can be zoomed. At most ``Job.picture_cap`` (None: all)
    are shown; the rest are ``not_shown``. ``regions_not_in_plane`` names, per
    position, the regions with no pixel in that plane. Refused: ``BAD_ARGS``,
    ``NO_STRUCTURES``, ``UNKNOWN_REGIONS``, ``NO_SIDES``, ``WRONG_SIDE_PAIR`` (a side
    the picture does not have: its message names the pair to use), ``SIDES_AMBIGUOUS``.
    """
    state = job.state
    resolved = _resolved(workspace, state, regions)
    if isinstance(positions_mm, (str, bytes)) or not isinstance(positions_mm, Sequence):
        raise Refused("BAD_ARGS", message="positions_mm is a list of numbers.")
    try:
        asked = [float(v) for v in positions_mm]
    except (TypeError, ValueError):
        raise Refused("BAD_ARGS", message="positions_mm must be numbers.") from None
    if not asked:
        raise Refused("BAD_ARGS", message="Give at least one position.")
    low, high = workspace.position_range
    used = [min(max(v, low), high) for v in asked]
    angles = state.view_angles
    edge = picture_edge(workspace)
    with sides_refused(), collecting() as notes:
        drawn = [_region_picture(workspace, state, resolved, mm, edge, angles)
                 for mm in used]
    shown, rest = split_shown(job, drawn)
    names = [name for name, _ids in resolved]
    entries = save_pictures(job, workspace, "grep_atlas_view", shown, notes,
                            {"regions": names, "positions_mm": asked})
    options = replace(default_options("template", atlas_channels=("template", "borders")),
                      regions=tuple(resolved))
    absent: dict[str, list[str]] = {}
    for mm in used[:len(shown)]:
        with sides_refused():
            present = regions_in_plane(workspace, state, mm, options, angles=angles)
        missing = [name for name in names if name not in present]
        if missing:
            absent[f"{mm:.2f}"] = missing
    return RegionsView(
        pictures=[picture.image for picture in shown], entries=entries,
        not_shown=not_shown(rest, "atlas"), regions=names, regions_not_in_plane=absent,
        clamped=[(a, u) for a, u in zip(asked, used, strict=True) if a != u])
