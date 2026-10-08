"""Atlas lookups: the region hierarchy, searched like text. Reads, never writes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core import look as looks
from langslice.core import zoom as zooms
from langslice.core.atlas.render import atlas_um_per_px
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
from langslice.ops.look import MAX_LOOK_PICTURES, not_shown, save_pictures
from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: The recipe renderer of :func:`grep_atlas_view`'s pictures.
REGIONS_RENDERER = "grep_atlas_view"


def grep_atlas(job: Job, workspace: Workspace, query: str, section: object = "") -> dict[str, Any]:
    """Regions whose acronym or name matches *query*
    (:func:`langslice.core.atlas_grep.grep_structures`).

    With *section* (the filename of a section with a
    position), each row says whether the region or a descendant appears in
    the atlas plane at that placement; a section without a position gets a
    ``note`` instead. Returns ``query``, ``matches`` (the total), ``rows``
    (at most :data:`~langslice.core.atlas_grep.GREP_ATLAS_LIMIT`), ``more``
    when some were left out. Refused: ``BAD_ARGS`` (empty query),
    ``NO_STRUCTURES``, ``UNKNOWN_SLICE_IDS``.
    """
    text = str(query).strip()
    if not text:
        raise Refused("BAD_ARGS", message="Empty query.")
    structures = getattr(workspace.atlas, "structures", None)
    entries = list(structures.values()) if structures else []
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
    from langslice.core.deformable.atlas_images import resolve_entries

    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise Refused("BAD_ARGS", message="regions is a list of acronyms, names or ids.")
    try:
        names = normalized_entries(str(entry) for entry in regions)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    if not names:
        raise Refused("BAD_ARGS", message="Name at least one region.")
    if not getattr(workspace.atlas, "structures", None):
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
    text = (f"atlas at {position_mm:.2f} mm{plane}; "
            f"{um:.1f} um/px; regions highlighted: {names}")
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
    so it can be zoomed. At most :data:`~langslice.ops.look.MAX_LOOK_PICTURES`
    are shown; the rest are ``not_shown``. ``regions_not_in_plane`` names, per
    position, the regions with no pixel in that plane. Refused: ``BAD_ARGS``,
    ``NO_STRUCTURES``, ``UNKNOWN_REGIONS``, ``NO_SIDES``.
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
    with collecting() as notes:
        drawn = [_region_picture(workspace, state, resolved, mm, edge, angles) for mm in used]
    shown, rest = drawn[:MAX_LOOK_PICTURES], drawn[MAX_LOOK_PICTURES:]
    names = [name for name, _ids in resolved]
    entries = save_pictures(job, workspace, "grep_atlas_view", shown, notes,
                            {"regions": names, "positions_mm": asked})
    options = replace(default_options("template", atlas_channels=("template", "borders")),
                      regions=tuple(resolved))
    absent: dict[str, list[str]] = {}
    for mm in used[:MAX_LOOK_PICTURES]:
        present = regions_in_plane(workspace, state, mm, options, angles=angles)
        missing = [name for name in names if name not in present]
        if missing:
            absent[f"{mm:.2f}"] = missing
    return RegionsView(
        pictures=[picture.image for picture in shown], entries=entries,
        not_shown=not_shown(rest, "atlas"), regions=names, regions_not_in_plane=absent,
        clamped=[(a, u) for a, u in zip(asked, used, strict=True) if a != u])
