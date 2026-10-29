"""Persistent landmark geometry and display annotations, independent of the image drawn."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.fits.deformation import linear_key
from langslice.core.jsonable import JsonObject
from langslice.core.sections.native import unoriented_fractions
from langslice.core.stack.deformation import LandmarkWarp
from langslice.core.stack.landmarks import Pair
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack


def affine_from_points(points: np.ndarray) -> np.ndarray:
    """Affine whose images of (0,0), (1,0), (0,1) are *points*."""
    return np.array(
        [[*(points[1] - points[0]), 0], [*(points[2] - points[0]), 0], [*points[0], 1]], dtype=float
    ).T


def transform(matrix: Any, points: Any) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return (
        np.concatenate([p, np.ones((*p.shape[:-1], 1))], axis=-1)
        @ np.asarray(matrix, dtype=float).T
    )[..., :2]


def orientation(section: Section) -> np.ndarray:
    """Oriented edge fractions -> acquisition edge fractions."""
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x, y = unoriented_fractions(p[:, 0], p[:, 1], section.rotation_deg, section.flip)
    return affine_from_points(np.stack([x, y], -1))


def plane_context(state: Stack, section: Section) -> dict[str, Any]:
    return {
        "atlas": state.atlas,
        "plane": state.plane,
        "position_mm": section.position_mm,
        "angles": list(section.angles),
    }


def stale(state: Stack, section: Section, pair: Pair) -> bool:
    """Return whether a pair was placed against another plane than now.

    Parameters
    ----------
    state
        The stack the section is in.
    section
        The section the pair is on.
    pair
        One of the section's landmark pairs.

    Returns
    -------
    bool
        True when the atlas plane the pair was placed against is not the
        section's now.
    """
    return pair.plane != plane_context(state, section)


def pair_key(pair: Pair) -> str:
    """What of a pair the landmark warp is made from: its two endpoints."""
    from langslice.core.fits.frame import digest

    return digest({"atlas": pair.atlas_native_xy, "section": pair.section_fraction_xy})


def section_pairs(section: Section) -> tuple[Pair, ...]:
    """Return a section's landmark pairs.

    Parameters
    ----------
    section
        The section whose pairs are read.

    Returns
    -------
    tuple of Pair
        Its pairs, in the order they were made; none before its first.
    """
    return () if section.landmarks is None else section.landmarks.pairs


def landmark_warp(section: Section) -> LandmarkWarp | None:
    """Return what the landmark warp a section holds was built from.

    Parameters
    ----------
    section
        The section whose deformation is read.

    Returns
    -------
    LandmarkWarp or None
        The pairs and baseline of its warp; None when its deformation is
        not a landmark warp.
    """
    if section.deformation is None:
        return None
    return section.deformation.landmark


def applied_pairs(section: Section) -> dict[str, str] | None:
    """The pairs the landmark warp that is the section's current deformation
    was built from (pair id -> :func:`pair_key`); None when the current
    deformation is not a landmark warp."""
    warp = landmark_warp(section)
    if warp is None or not warp.revision:
        return None
    return dict(warp.pairs)


def pair_states(state: Stack, section: Section) -> dict[str, str]:
    """Each pair's state: ``stale`` (placed against another atlas plane than
    the section's now), ``applied`` (an endpoint-for-endpoint member of the
    landmark warp that is the section's current deformation), ``moved`` (in
    that warp, but with other endpoints now) or ``pending`` (not in it)."""
    built = applied_pairs(section) or {}
    out: dict[str, str] = {}
    for pair in section_pairs(section):
        if stale(state, section, pair):
            out[pair.id] = "stale"
        elif pair.id not in built:
            out[pair.id] = "pending"
        else:
            out[pair.id] = "applied" if built[pair.id] == pair_key(pair) else "moved"
    return out


def warp_report(state: Stack, section: Section) -> dict[str, Any]:
    """Which pairs the section's current deformation was built from
    (``built_from``), which pairs are not in it as they are now
    (``not_applied``), which were deleted but still shape it
    (``deleted_still_in_warp``), in a ``note``; empty when the section has no
    pairs and no landmark warp."""
    pairs = [pair.id for pair in section_pairs(section)]
    built = applied_pairs(section)
    if built is None:
        if not pairs:
            return {}
        return {"built_from": [], "not_applied": pairs,
                "note": "No landmark warp is applied to this section yet; "
                        "landmark_triangulation applies the pairs."}
    states = pair_states(state, section)
    new = [i for i in pairs if states[i] == "pending"]
    moved = [i for i in pairs if states[i] == "moved"]
    stale_ids = [i for i in pairs if states[i] == "stale"]
    gone = [i for i in built if i not in pairs]
    made_of = sorted(built, key=_order)
    report: dict[str, Any] = {"built_from": made_of}
    if not (new or moved or stale_ids or gone):
        report["note"] = "The current warp was built from exactly these pairs."
        return report
    report["not_applied"] = [*new, *moved, *stale_ids]
    if gone:
        report["deleted_still_in_warp"] = gone
    parts = [f"The current warp was built from {', '.join(made_of) or 'no pair'}."]
    if new:
        parts.append(f"Added, not in it yet: {', '.join(new)}.")
    if moved:
        parts.append(f"Moved since, so it still holds their earlier endpoints: "
                     f"{', '.join(moved)}.")
    if stale_ids:
        parts.append(f"Placed against an earlier atlas plane: {', '.join(stale_ids)}.")
    if gone:
        parts.append(f"Deleted, but still shaping it: {', '.join(gone)}.")
    parts.append("Run landmark_triangulation again to apply the set as it stands now.")
    report["note"] = " ".join(parts)
    return report


def _order(pair_id: str) -> tuple[int, str]:
    return (len(pair_id), pair_id)


def landmark_counts(state: Stack, section: Section) -> dict[str, Any]:
    """``{"total", "pending", "stale", "warp"}`` of the section's pairs
    (``pending``: not in the applied landmark warp as they are now; ``warp``
    in words: whether that warp matches the pairs); empty when it has none."""
    states = list(pair_states(state, section).values())
    if not states and applied_pairs(section) is None:
        return {}
    report = warp_report(state, section)
    matches = bool(report) and "not_applied" not in report
    return {"total": len(states),
            "pending": states.count("pending") + states.count("moved"),
            "stale": states.count("stale"),
            "warp": "matches the pairs" if matches else
            "none applied" if applied_pairs(section) is None
            else "differs from the pairs; run landmark_triangulation again"}


def section_panel(
    section: Section,
    size: tuple[int, int],
    *,
    pixel_to_render: Any,
    bounds: Any,
    warp_key: str | None = None,
    role: str = "section",
) -> dict[str, Any]:
    return {
        "role": role,
        "section": section.id,
        "bounds": list(bounds),
        "render_size": list(size),
        "pixel_to_render": np.asarray(pixel_to_render).tolist(),
        "orientation": orientation(section).tolist(),
        "warp_key": warp_key,
    }


def section_point(panel: dict[str, Any], xy: Any, warp: Any = None) -> np.ndarray:
    if "pixel_to_file_fraction" in panel:
        return transform(panel["pixel_to_file_fraction"], xy)
    p = transform(panel["pixel_to_render"], xy)
    width, height = panel["render_size"]
    if warp is not None:
        from langslice.core.fits.deformable.render import warp_source_points

        x, y = warp_source_points(warp, (width, height), np.array([[p[0]]]), np.array([[p[1]]]))
        p = np.array([x.item(), y.item()])
    fractions = (p + 0.5) / np.array([width, height])
    return transform(panel["orientation"], fractions)


def section_picture_point(panel: dict[str, Any], fractions: Any, warp: Any = None) -> np.ndarray:
    if "pixel_to_file_fraction" in panel:
        return transform(np.linalg.inv(panel["pixel_to_file_fraction"]), fractions)
    oriented = transform(np.linalg.inv(panel["orientation"]), fractions)
    size = np.asarray(panel["render_size"])
    p = oriented * size - 0.5
    if warp is not None:
        from scipy.ndimage import map_coordinates

        wp = (p + 0.5) * np.asarray(warp.section_size) / size - 0.5
        shift = np.array(
            [
                map_coordinates(
                    warp.field_mm[..., k], [[wp[1]], [wp[0]]], order=1, mode="nearest"
                ).item()
                for k in range(2)
            ]
        )
        p += shift / warp.mm_per_px * size / np.asarray(warp.section_size)
    return transform(np.linalg.inv(panel["pixel_to_render"]), p)


def glyph(image: Image.Image, xy: Any, label: str, role: str, *, old: bool = False,
          taken: list[tuple[float, float, float, float]] | None = None) -> None:
    """Open-centre circle (section) / diamond (atlas), dual contrast outlines.

    *taken* lists the label boxes already drawn on the picture: a label that
    would cover one is moved down until it is free, and its box is added."""
    x, y = (float(v) for v in xy)
    draw = ImageDraw.Draw(image)
    radius = 7
    for colour, width in (("black", 5), ("white", 3), ("#ffaa00" if old else "#00ffff", 1)):
        if role == "atlas":
            draw.line(
                [
                    (x, y - radius),
                    (x + radius, y),
                    (x, y + radius),
                    (x - radius, y),
                    (x, y - radius),
                ],
                fill=colour,
                width=width,
            )
        else:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), outline=colour, width=width
            )
    if not label:
        return
    text = label + (" stale" if old else "")
    top = y - 7
    box = draw.textbbox((x + 10, top), text)
    for _ in range(12):
        padded = (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)
        if taken is None or not any(
            padded[0] < t[2] and t[0] < padded[2] and padded[1] < t[3] and t[1] < padded[3]
            for t in taken
        ):
            break
        top += box[3] - box[1] + 4
        box = draw.textbbox((x + 10, top), text)
    padded = (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)
    if taken is not None:
        taken.append(padded)
    draw.rectangle(padded, fill="black", outline="white")
    draw.text((x + 10, top), text, fill="white")


#: Markers of one kind closer than this (picture pixels) share one label.
MERGE_PX = 12.0


#: The colour folded triangles are outlined in on a warped overlay.
FOLD_COLOUR = "#ff2bd6"


def draw_folds(
    image: Image.Image, panel: dict[str, Any], section: Section, *,
    warp: DeformableRecord | None = None,
) -> None:
    """Outline the folded triangles of the applied landmark warp, between the
    section endpoints of their pairs (a fixed corner vertex has none; a vertex
    a warp saved by an earlier version names otherwise is skipped)."""
    warp_facts = landmark_warp(section)
    folds: JsonObject = {} if warp_facts is None else warp_facts.folds
    # the folded triangles, each the ids of its corner pairs
    triangles = cast("list[list[str]]", folds.get("triangles", []))
    by_id = {pair.id: pair.section_fraction_xy for pair in section_pairs(section)}
    draw = ImageDraw.Draw(image)
    for ids in triangles:
        points = [
            tuple(float(v) for v in section_picture_point(panel, by_id[i], warp))
            for i in ids if i in by_id
        ]
        if len(points) < 2:
            continue
        path = points + ([points[0]] if len(points) > 2 else [])
        draw.line(path, fill="black", width=5)
        draw.line(path, fill=FOLD_COLOUR, width=3)


def draw_landmarks(
    image: Image.Image,
    panels: list[dict[str, Any]],
    state: Stack,
    section: Section,
    *,
    warp: DeformableRecord | None = None,
) -> None:
    for panel in panels:
        if panel["role"] == "section" and panel.get("warp_key"):
            draw_folds(image, panel, section, warp=warp)
    placed: list[tuple[str, np.ndarray, str, bool]] = []
    for pair in section_pairs(section):
        for panel in panels:
            role = panel["role"]
            xy = (
                transform(np.linalg.inv(panel["pixel_to_native"]), pair.atlas_native_xy)
                if role == "atlas"
                else section_picture_point(
                    panel, pair.section_fraction_xy, warp if panel.get("warp_key") else None
                )
            )
            x0, y0, x1, y1 = panel["bounds"]
            if x0 <= xy[0] < x1 and y0 <= xy[1] < y1:
                placed.append((role, np.asarray(xy, dtype=float), pair.id,
                               stale(state, section, pair)))
    # Markers of one kind at the same spot, or near enough for their labels to
    # collide, are one group with one label naming every id; labels of
    # different groups never cover each other.
    groups: list[list[tuple[str, np.ndarray, str, bool]]] = []
    for item in placed:
        for group in groups:
            if group[0][0] == item[0] and any(
                np.hypot(*(member[1] - item[1])) <= MERGE_PX for member in group
            ):
                group.append(item)
                break
        else:
            groups.append([item])
    taken: list[tuple[float, float, float, float]] = []
    labelled: list[list[tuple[str, np.ndarray, str, bool]]] = []
    for group in groups:
        first = group[0]
        label = ", ".join(member[2] for member in group)
        # A pair whose two endpoints meet (its warp applied) is labelled once:
        # the other kind's group of the same ids at that spot already says it.
        if any(other[0][0] != first[0] and {m[2] for m in other} == {m[2] for m in group}
               and any(np.hypot(*(a[1] - b[1])) <= MERGE_PX for a in other for b in group)
               for other in labelled):
            label = ""
        else:
            labelled.append(group)
        for member in group[1:]:
            glyph(image, member[1], "", member[0], old=member[3])
        glyph(image, first[1], label, first[0], old=all(m[3] for m in group), taken=taken)


def context(state: Stack, section: Section) -> dict[str, Any]:
    return {
        "plane": plane_context(state, section),
        "linear_key": linear_key(state, section),
        "orientation": {"rotate_quarter": section.rotation_deg, "flip": section.flip},
        "deformation_key": section.deformation_key,
    }
