"""The status table: one row per section, as data for the doors.

Not a picture: the rows every write answers with, ``status`` returns whole, and
the opening and MCP briefing print as text (:func:`status_text`).
"""

from __future__ import annotations

from typing import Any

from langslice.core.state import SliceState, StackState

#: What a status reply says under ``cutting_angles_deg`` when the sections'
#: angles differ: each row carries its own.
PER_SECTION = "per section"


def angles_dict(record: SliceState) -> dict[str, float]:
    """The section's ``{"pitch", "yaw"}`` in degrees, as floats."""
    pitch, yaw = record.angles
    return {"pitch": pitch, "yaw": yaw}


def stack_angles_entry(state: StackState) -> dict[str, float] | str:
    """What a status reply says under ``cutting_angles_deg``: the stack's
    ``{"pitch", "yaw"}`` (a copy of the stored angles), or
    :data:`PER_SECTION` when the sections differ (their rows carry them)."""
    return PER_SECTION if state.mixed_angles else dict(state.cutting_angles_deg)


def status_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order. The ``ls`` of the environment.

    ``delta_to_next_mm`` is the SIGNED distance to the next section in
    corrected order that carries a position, and null when this section has
    none or no placed section follows it. ``transform_iou`` and
    ``transform_mirrored`` come off the recorded transform. Data only: no
    comparison against the nominal interval, no verdict. When the sections'
    cutting angles differ (a registration supplied per section), each row
    also carries its own ``cutting_angles_deg``; a single-angle stack's rows
    do not (the stack's angle is reported once, beside them).
    """
    ordered = state.in_order()
    mixed = state.mixed_angles
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
                **({"cutting_angles_deg": angles_dict(record)} if mixed else {}),
                "damaged": record.damaged,
                "damage_note": record.damage_note,
                "transform": transform.get("kind"),
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


#: Every field a status row can carry, in order. A model gets rows without
#: their null and empty fields (:func:`compact_rows`); a script or a coding
#: agent gets every row with all of them (:func:`uniform_rows`).
ROW_FIELDS: tuple[str, ...] = (
    "index", "id", "position_mm", "delta_to_next_mm", "flip", "rotation_deg",
    "cutting_angles_deg", "damaged", "damage_note", "transform", "transform_iou",
    "transform_mirrored", "keep_linear", "deformation_steps", "caveats",
    "locked", "damage_by_user",
)
#: What :func:`uniform_rows` fills in where a row has no value (else null):
#: ``locked`` and ``damage_by_user`` (``ops.views.status``) are set only
#: when true.
ROW_DEFAULTS: dict[str, Any] = {"caveats": [], "locked": False, "damage_by_user": False}
#: The reply keys that hold status rows (``status``, ``view_stack``: the
#: whole table; a write: the rows it changed).
ROW_KEYS: tuple[str, ...] = ("rows", "changed")


def uniform_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows (whole or compacted) with every field of :data:`ROW_FIELDS`:
    null where a row has no value (or its :data:`ROW_DEFAULTS` value), then
    any other field a row carries. One schema for every row, so a script can
    read any field of any row (the last placed section's
    ``delta_to_next_mm`` is null, not missing)."""
    extra = list(dict.fromkeys(key for row in rows for key in row if key not in ROW_FIELDS))
    def filled(row: dict[str, Any], key: str) -> Any:
        if key in row:
            return row[key]
        default = ROW_DEFAULTS.get(key)
        return list(default) if isinstance(default, list) else default

    return [{**{key: filled(row, key) for key in ROW_FIELDS},
             **{key: row.get(key) for key in extra}} for row in rows]


def with_uniform_rows(reply: Any) -> Any:
    """A tool reply with its status rows (:data:`ROW_KEYS`) made uniform
    (:func:`uniform_rows`); anything else as it is. For the doors a script
    reads (the agent CLI, the library)."""
    if not isinstance(reply, dict):
        return reply
    return {key: uniform_rows(value) if key in ROW_KEYS and isinstance(value, list)
            and all(isinstance(row, dict) for row in value) else value
            for key, value in reply.items()}


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows without their null and empty fields, for a tool payload.

    Null transform fields and empty lists on every row of every result
    would be a large share of the text a model pays for. Absent means null;
    ``status_text`` keeps the full rows.
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
        if "cutting_angles_deg" in row:
            angles = row["cutting_angles_deg"]
            transform += f"  angles=pitch {angles['pitch']:.2f} yaw {angles['yaw']:.2f}"
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
