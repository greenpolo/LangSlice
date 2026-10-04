"""The status table: one row per section, as data for the doors.

Split out of ``linear/render.py`` (layered refactor, phase 3d). Not a
picture: the rows every write answers with, ``status`` returns whole, and
the opening and MCP briefing print as text (:func:`status_text`).
"""

from __future__ import annotations

from typing import Any

from langslice.core.state import SliceState, StackState


def status_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order. The ``ls`` of the environment.

    ``delta_to_next_mm`` is the SIGNED distance to the next section in
    corrected order that carries a position, and null when this section has
    none or no placed section follows it. ``transform_iou`` and
    ``transform_mirrored`` come off the recorded transform. Data only: no
    comparison against the nominal interval, no verdict.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        here = record.position_mm
        delta: float | None = None
        if here is not None:
            following = next(
                (
                    value
                    for r in ordered[index + 1 :]
                    if (value := r.position_mm) is not None
                ),
                None,
            )
            if following is not None:
                delta = round(following - here, 3)
        transform = record.transform or {}
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "delta_to_next_mm": delta,
                "flip": record.flip,
                "rotation_deg": record.rotation_deg,
                "damaged": record.damaged,
                "damage_note": record.damage_note,
                "transform": transform.get("kind"),
                **({"transform_model": (
                    "elastix_bspline" if transform["spline"].get("backend") == "elastix"
                    else "thin_plate_spline"),
                    "landmarks": len(transform["spline"]["source"])}
                   if transform.get("spline") else {}),
                "transform_iou": transform.get("iou"),
                "transform_mirrored": transform.get("mirrored"),
                **({"keep_linear": record.deformation["keep_linear"]}
                   if record.deformation and "keep_linear" in record.deformation
                   else {"deformation_steps": len(record.deformation.get("steps") or [])}
                   if record.deformation else {}),
                "caveats": list(record.caveats),
            }
        )
    return rows


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows without their null and empty fields, for a tool payload.

    A position-only run carried null transform fields and empty caveat lists
    on every row of every result, a third of the text the model paid for
    (run 5, 2026-09-09). Absent means null; ``status_text`` keeps the full
    rows.
    """
    return [
        {k: v for k, v in row.items() if v is not None and v != [] and v != ""}
        for row in rows
    ]


def status_text(state: StackState) -> str:
    """The status rows as one line per section, for a text message."""
    lines = [
        "index  id  position_mm  delta_to_next_mm  flags  "
        "transform(kind, iou, mirrored)"
    ]
    for row in status_rows(state):
        flags: list[str] = []
        if row["flip"]:
            flags.append("flipped")
        if row["rotation_deg"]:
            flags.append(f"rotated {row['rotation_deg']}")
        if row["damaged"]:
            note = row["damage_note"]
            flags.append(f"damaged: {note}" if note else "damaged")
        flags.extend(row["caveats"])
        position = (
            "unplaced" if row["position_mm"] is None else f"{row['position_mm']:.3f} mm"
        )
        delta = (
            "-" if row["delta_to_next_mm"] is None else f"{row['delta_to_next_mm']:.3f}"
        )
        transform = ""
        if row["transform"]:
            transform = f"  transform={row['transform']}"
            if row["transform_iou"] is not None:
                transform += f" iou={float(row['transform_iou']):.3f}"
            if row["transform_mirrored"] is not None:
                transform += f" mirrored={bool(row['transform_mirrored'])}"
        if row.get("deformation_steps"):
            transform += f"  deformation={row['deformation_steps']} step(s)"
        lines.append(
            f"{row['index']:>3}  {row['id']}  {position}  {delta}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
            + transform
        )
    return "\n".join(lines)


def slice_flags(record: SliceState) -> list[str]:
    """The section's current corrections, as short human-readable flags."""
    flags: list[str] = []
    if record.rotation_deg:
        flags.append(f"rotated {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append(
            f"damaged: {record.damage_note}" if record.damage_note else "damaged"
        )
    return flags
