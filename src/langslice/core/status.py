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


def mask_rows(state: StackState, record: SliceState) -> list[dict[str, Any]]:
    """``[{name, kind, stale}]`` of *record*'s masks."""
    from langslice.core.section_masks import mask_rows as rows

    return rows(state, record)


#: Step fields that say how a step was made, not with which settings.
_STEP_FACTS = ("method", "start", "note", "trace", "settings", "include", "exclude")
#: A native step's stored setting names as the tools name them.
_STEP_NAMES = {"fit_section": "section_image", "fit_atlas": "atlas_image"}


def step_method(step: dict[str, Any]) -> str:
    """The method a deformation step names (a step saved before steps named
    one: ``trace_borders`` for a traced fit, else ``fireants_syn``)."""
    if step.get("method"):
        return str(step["method"])
    return "trace_borders" if step.get("trace") else "fireants_syn"


def step_settings(step: dict[str, Any]) -> dict[str, Any]:
    """The settings a deformation step was made with, by the tools' names: a
    plugin's or a script's ``settings`` as it gave them, a native fit's own
    (``section_image``, ``atlas_image``, ``mask``, ``smoothness_um``, ...,
    and ``restrict_to`` when it fitted by regions)."""
    if isinstance(step.get("settings"), dict):
        return dict(step["settings"])
    out = {_STEP_NAMES.get(key, key): value for key, value in step.items()
           if key not in _STEP_FACTS}
    if step.get("include"):
        out["restrict_to"] = list(step["include"])
    return out


def deformation_entry(deformation: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a status row and a registration listing say of a deformation:
    ``method`` (its last step's), ``steps`` (every step's method, first
    first) and ``settings`` (the last step's, :func:`step_settings`); None
    without one."""
    if not deformation:
        return None
    steps = [step for step in deformation.get("steps") or [] if isinstance(step, dict)]
    if not steps:
        return {"method": "unknown", "steps": [], "settings": {}}
    return {"method": step_method(steps[-1]), "steps": [step_method(s) for s in steps],
            "settings": step_settings(steps[-1])}


def transform_entry(transform: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a status row and a registration listing say of a transform: its
    ``kind``, the knobs ``interactive_transform`` takes (``rotation_deg``,
    ``scale_x``, ``scale_y``, ``shear``, ``translate_x_mm``, ``translate_y_mm``
    and their ``pivot``) and ``mirrored``; None without one."""
    if not transform:
        return None
    knobs = transform.get("physical") or {}
    return {"kind": transform.get("kind"), **knobs,
            **({"mirrored": transform["mirrored"]} if "mirrored" in transform else {})}


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
    none or no placed section follows it. ``mirrored`` (in ``transform``) comes off
    the recorded transform. Data only: no comparison against the nominal
    interval, no verdict. ``transform`` is the section's transform
    (:func:`transform_entry`: its kind and knobs) and ``deformation`` the
    deformation it holds (:func:`deformation_entry`: its method, its steps'
    methods and its settings). ``damaged_regions``
    are the section's marked regions (``damaged``: it has some); a
    ``damage_note`` may stand alone (a host's note). When the sections'
    cutting angles differ (a registration supplied per section), each row
    also carries its own ``cutting_angles_deg``; a single-angle stack's rows
    do not (the stack's angle is reported once, beside them). A section with
    landmark pairs carries ``landmarks``: their ``total``, ``pending`` (not in
    the landmark warp applied now) and ``stale`` (placed against another atlas
    plane) counts. A section tool plugins left layers on carries ``layers``
    (name, kind, frame and tool of each); other sections do not.
    """
    from langslice.core.landmarks import landmark_counts

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
        rows.append(
            {
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "delta_to_next_mm": delta,
                "flip": record.flip,
                "rotation_deg": record.rotation_deg,
                **({"cutting_angles_deg": angles_dict(record)} if mixed else {}),
                "damaged": record.damaged,
                "damaged_regions": list(record.damaged_regions),
                "damage_note": record.damage_note,
                "transform": transform_entry(record.transform),
                **({"deformation": deformation_entry(record.deformation)}
                   if record.deformation else {}),
                **({"landmarks": counts} if (counts := landmark_counts(state, record)) else {}),
                **({"layers": [{"name": ref["name"], "kind": ref["kind"],
                                "frame": ref["frame"], "tool": ref["tool"]}
                               for ref in record.layers]} if record.layers else {}),
                **({"masks": mask_rows(state, record)} if record.masks else {}),
                "caveats": list(record.caveats),
            }
        )
    return rows


#: Every field a status row can carry, in order. A model gets rows without
#: their null and empty fields (:func:`compact_rows`); a script or a coding
#: agent gets every row with all of them (:func:`uniform_rows`).
ROW_FIELDS: tuple[str, ...] = (
    "id", "position_mm", "delta_to_next_mm", "flip",
    "rotation_deg",
    "cutting_angles_deg", "damaged", "damaged_regions", "damage_note", "transform",
    "deformation", "landmarks", "caveats",
    "frozen", "damage_by_user", "registration",
)
#: What :func:`uniform_rows` fills in where a row has no value (else null):
#: ``frozen`` (the properties the user froze on the section) and
#: ``damage_by_user`` (``ops.views.status``) are set only when there are any.
ROW_DEFAULTS: dict[str, Any] = {"caveats": [], "damaged_regions": [], "frozen": [],
                                "damage_by_user": False}
#: The reply keys that hold status rows (``status``: the whole table; a
#: write: the rows it changed).
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
    return {key: uniform_rows(value) if key in ROW_KEYS and _status_rows(value) else value
            for key, value in reply.items()}


def _status_rows(value: Any) -> bool:
    """*value* is a list of status rows: dicts with no field outside
    :data:`ROW_FIELDS` (``grep_atlas``'s region rows, under the same key, are
    not)."""
    # ``layers``, ``masks``: only a section with plugin layers / masks has them
    fields = {*ROW_FIELDS, "layers", "masks"}
    return isinstance(value, list) and all(
        isinstance(row, dict) and set(row) <= fields for row in value)


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
        "filename  position_mm  delta_to_next_mm  flags  "
        "transform(kind, mirrored)  deformation(methods)"
    ]
    for row in status_rows(state):
        flags: list[str] = []
        if row["flip"]:
            flags.append("flipped")
        if row["rotation_deg"]:
            flags.append(f"rotated {row['rotation_deg']}")
        if row["damaged"]:
            flags.append(_damage_flag(row["damaged_regions"], row["damage_note"]))
        elif row["damage_note"]:
            flags.append(f"damage note: {row['damage_note']}")
        flags.extend(row["caveats"])
        position = (
            "unplaced" if row["position_mm"] is None else f"{row['position_mm']:.3f} mm"
        )
        delta = (
            "-" if row["delta_to_next_mm"] is None else f"{row['delta_to_next_mm']:.3f}"
        )
        transform = ""
        held = row["transform"]
        if held:
            transform = f"  transform={held.get('kind')}"
            if held.get("mirrored") is not None:
                transform += f" mirrored={bool(held['mirrored'])}"
        if row.get("deformation"):
            transform += f"  deformation={' + '.join(row['deformation']['steps'])}"
        if "cutting_angles_deg" in row:
            angles = row["cutting_angles_deg"]
            transform += f"  angles=pitch {angles['pitch']:.2f} yaw {angles['yaw']:.2f}"
        lines.append(
            f"{row['id']}  {position}  {delta}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
            + transform
        )
    return "\n".join(lines)


def _damage_flag(regions: list[str], note: str) -> str:
    """``damaged``, its regions in brackets, then its note: ``damaged
    [CTX:left, OB]: torn``."""
    head = "damaged" + (f" [{', '.join(regions)}]" if regions else "")
    return f"{head}: {note}" if note else head
