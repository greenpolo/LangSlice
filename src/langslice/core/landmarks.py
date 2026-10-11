"""Persistent landmark geometry and display annotations, independent of appearance."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from langslice.core.deformation import linear_key
from langslice.core.native import unoriented_fractions
from langslice.core.state import SliceState, StackState


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


def orientation(record: SliceState) -> np.ndarray:
    """Oriented edge fractions -> acquisition edge fractions."""
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x, y = unoriented_fractions(p[:, 0], p[:, 1], record.rotation_deg, record.flip)
    return affine_from_points(np.stack([x, y], -1))


def plane_context(state: StackState, record: SliceState) -> dict[str, Any]:
    return {
        "atlas": state.atlas,
        "plane": state.plane,
        "position_mm": record.position_mm,
        "angles": list(record.angles),
    }


def stale(state: StackState, record: SliceState, pair: dict[str, Any]) -> bool:
    return pair.get("plane") != plane_context(state, record)


def pair_key(pair: dict[str, Any]) -> str:
    """What of a pair the landmark warp is made from: its two endpoints."""
    from langslice.core.handoff import digest

    return digest({"atlas": pair["atlas_native_xy"], "section": pair["section_fraction_xy"]})


def applied_pairs(record: SliceState) -> dict[str, str] | None:
    """The pairs the landmark warp that is the section's current deformation
    was built from (pair id -> :func:`pair_key`); None when the current
    deformation is not a landmark warp."""
    held = record.deformation or {}
    if not held.get("landmark_revision"):
        return None
    built = held.get("landmark_pairs") or {}
    if not built:  # a record from before the pairs were listed in it
        from langslice.core.handoff import digest

        pairs = record.landmarks.get("pairs", [])
        if digest(pairs) == held["landmark_revision"]:
            built = {pair["id"]: pair_key(pair) for pair in pairs}
    return dict(built)


def pair_states(state: StackState, record: SliceState) -> dict[str, str]:
    """Each pair's state: ``stale`` (placed against another atlas plane than
    the section's now), ``applied`` (an endpoint-for-endpoint member of the
    landmark warp that is the section's current deformation), ``moved`` (in
    that warp, but with other endpoints now) or ``pending`` (not in it)."""
    built = applied_pairs(record) or {}
    out: dict[str, str] = {}
    for pair in record.landmarks.get("pairs", []):
        if stale(state, record, pair):
            out[pair["id"]] = "stale"
        elif pair["id"] not in built:
            out[pair["id"]] = "pending"
        else:
            out[pair["id"]] = "applied" if built[pair["id"]] == pair_key(pair) else "moved"
    return out


def warp_report(state: StackState, record: SliceState) -> dict[str, Any]:
    """Which pairs the section's current deformation was built from
    (``built_from``), which pairs are not in it as they are now
    (``not_applied``), which were deleted but still shape it
    (``deleted_still_in_warp``), in a ``note``; empty when the section has no
    pairs and no landmark warp."""
    pairs = [pair["id"] for pair in record.landmarks.get("pairs", [])]
    built = applied_pairs(record)
    if built is None:
        if not pairs:
            return {}
        return {"built_from": [], "not_applied": pairs,
                "note": "No landmark warp is applied to this section yet; "
                        "landmark_triangulation applies the pairs."}
    states = pair_states(state, record)
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


def landmark_counts(state: StackState, record: SliceState) -> dict[str, Any]:
    """``{"total", "pending", "stale", "warp"}`` of the section's pairs
    (``pending``: not in the applied landmark warp as they are now; ``warp``
    in words: whether that warp matches the pairs); empty when it has none."""
    states = list(pair_states(state, record).values())
    if not states and applied_pairs(record) is None:
        return {}
    report = warp_report(state, record)
    matches = bool(report) and "not_applied" not in report
    return {"total": len(states),
            "pending": states.count("pending") + states.count("moved"),
            "stale": states.count("stale"),
            "warp": "matches the pairs" if matches else
            "none applied" if applied_pairs(record) is None
            else "differs from the pairs; run landmark_triangulation again"}


def section_panel(
    record: SliceState,
    size: tuple[int, int],
    *,
    pixel_to_render: Any,
    bounds: Any,
    warp_key: str | None = None,
    role: str = "section",
) -> dict[str, Any]:
    return {
        "role": role,
        "section": record.id,
        "bounds": list(bounds),
        "render_size": list(size),
        "pixel_to_render": np.asarray(pixel_to_render).tolist(),
        "orientation": orientation(record).tolist(),
        "warp_key": warp_key,
    }


def section_point(panel: dict[str, Any], xy: Any, warp: Any = None) -> np.ndarray:
    if "pixel_to_file_fraction" in panel:
        return transform(panel["pixel_to_file_fraction"], xy)
    p = transform(panel["pixel_to_render"], xy)
    width, height = panel["render_size"]
    if warp is not None:
        from langslice.core.deformable.render import warp_source_points

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
    image: Image.Image, panel: dict[str, Any], record: SliceState, *, warp: Any = None
) -> None:
    """Outline the folded triangles of the applied landmark warp, between the
    section endpoints of their pairs (a fixed corner vertex has none)."""
    folds = (record.deformation or {}).get("landmark_folds") or {}
    by_id = {pair["id"]: pair for pair in record.landmarks.get("pairs", [])}
    draw = ImageDraw.Draw(image)
    for ids in folds.get("triangles", []):
        points = [
            tuple(float(v) for v in section_picture_point(
                panel, by_id[i]["section_fraction_xy"], warp))
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
    state: StackState,
    record: SliceState,
    *,
    warp: Any = None,
) -> None:
    for panel in panels:
        if panel["role"] == "section" and panel.get("warp_key"):
            draw_folds(image, panel, record, warp=warp)
    placed: list[tuple[str, np.ndarray, str, bool]] = []
    for pair in record.landmarks.get("pairs", []):
        for panel in panels:
            role = panel["role"]
            xy = (
                transform(np.linalg.inv(panel["pixel_to_native"]), pair["atlas_native_xy"])
                if role == "atlas"
                else section_picture_point(
                    panel, pair["section_fraction_xy"], warp if panel.get("warp_key") else None
                )
            )
            x0, y0, x1, y1 = panel["bounds"]
            if x0 <= xy[0] < x1 and y0 <= xy[1] < y1:
                placed.append((role, np.asarray(xy, dtype=float), pair["id"],
                               stale(state, record, pair)))
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
    for group in groups:
        first = group[0]
        label = ", ".join(member[2] for member in group)
        for member in group[1:]:
            glyph(image, member[1], "", member[0], old=member[3])
        glyph(image, first[1], label, first[0], old=all(m[3] for m in group), taken=taken)


def context(state: StackState, record: SliceState) -> dict[str, Any]:
    return {
        "plane": plane_context(state, record),
        "linear_key": linear_key(state, record),
        "orientation": {"rotate_quarter": record.rotation_deg, "flip": record.flip},
        "deformation_key": (record.deformation or {}).get("key"),
    }
