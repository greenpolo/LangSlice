"""What a host supplies for a job: its inputs, applied to a new stack.

A host configures a job with ``JobSpec.inputs`` (what is already known per
section) and ``JobSpec.frozen`` (what the agent may not change).
:func:`apply_host_inputs` writes them into a freshly ingested stack, and
:func:`changed_inputs` tells a resume that was made from other inputs.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any, cast

from langslice.core.jsonable import Json, JsonObject, to_jsonable
from langslice.core.stack.affine import IDENTITY_PHYSICAL, StoredAffine
from langslice.core.stack.codec import affine_from_json
from langslice.core.stack.section import ROTATIONS, Section
from langslice.core.stack.stack import Stack
from langslice.job.settings import frozen as freeze
from langslice.job.settings.spec import JobSpec, supplied_angles

#: The transform a section with a frozen affine and none supplied carries: the
#: host's snapshot is already registered in-plane, so the identity IS its
#: alignment. ``kind`` ``"host"`` counts as a transform at submit and is never
#: written back to the host.
HOST_TRANSFORM_KIND = "host"


def _named(spec: JobSpec, key: str) -> set[str]:
    """The section filenames ``spec.inputs[key]`` names (a list, or a mapping's keys)."""
    return {str(name) for name in (spec.inputs or {}).get(key) or []}


def host_damaged_ids(spec: JobSpec) -> set[str]:
    """Sections the host gave a damage note (``inputs.damaged``); the note
    stays first in the section's ``damage_note``."""
    return _named(spec, "damaged")


def host_transform() -> StoredAffine:
    """The identity transform a section with a frozen affine and none supplied
    carries: the host's snapshot is already aligned in-plane."""
    return StoredAffine(
        kind=HOST_TRANSFORM_KIND, params=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        physical=IDENTITY_PHYSICAL, mirrored=False)


def _section(state: Stack, key: str, name: object) -> Section:
    """The section *name* of ``inputs.<key>``; ``ValueError`` for one not here."""
    section = state.by_id(str(name))
    if section is None:
        raise ValueError(f"inputs.{key} names an unknown section: {name!r}")
    return section


def _host_affine(given: object) -> StoredAffine:
    """Return a copy of a host's transform dictionary as the stored affine."""
    return affine_from_json(copy.deepcopy(cast("JsonObject", given)))


def apply_host_inputs(state: Stack, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}`` for the whole stack or
    a filename -> ``{"pitch": deg, "yaw": deg}`` mapping, each section's own
    plane kept as supplied
    (:func:`langslice.job.settings.spec.supplied_angles`; a section it does
    not name keeps the flat plane), orientation as a
    filename -> ``{"flip": bool, "rotation_deg": 0|90|180|270}`` mapping (a
    missing key leaves that part as it is), damage as a filename -> note
    mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host supplies for a task that IS on is applied too — it is
    a starting point, not a constraint. A ``damaged`` note becomes the
    section's ``damage_note`` (shown in status, so the agent can mark the
    regions it names); it does not make the section damaged, and it stays
    first in the note whatever the agent marks. ``spec.frozen`` is the
    constraint (:mod:`langslice.job.settings.frozen`): it must name sections
    that are here, a frozen position must have been supplied, and a section
    with a frozen affine and no supplied transform carries the ``"host"``
    identity (:func:`host_transform`), because its snapshot is already
    aligned. The other "none" values (the flat plane, no flip and no turn,
    no deformation) are what a section starts with.
    """
    inputs = spec.inputs or {}

    order = inputs.get("order") or []
    if order:
        known = [state.by_id(str(name)) for name in order]
        missing = [str(name) for name, hit in zip(order, known, strict=True) if hit is None]
        if missing:
            raise ValueError(f"inputs.order names sections that are not here: {missing}")
        tail = [s for s in state.in_order() if s.id not in {str(name) for name in order}]
        for index, record in enumerate([r for r in known if r is not None] + tail):
            record.index_corrected = index
        state.notes.append(f"inputs: order set by the host ({len(order)} sections)")

    positions = inputs.get("positions") or {}
    if positions:
        applied = 0
        for name, value in positions.items():
            record = _section(state, "positions", cast("object", name))
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    stack_angles, section_angles = supplied_angles(inputs.get("angles"))
    if stack_angles is not None:
        pitch, yaw = stack_angles
        state.set_angles((pitch, yaw))
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {pitch:.2f}, yaw {yaw:.2f})"
        )
    if section_angles:
        for name, (pitch, yaw) in section_angles.items():
            record = _section(state, "angles", name)
            record.angles = (pitch, yaw)
        state.notes.append(
            f"inputs: cutting angles of {len(section_angles)} section(s) set by the host"
        )

    orientation = inputs.get("orientation") or {}
    if orientation:
        if not isinstance(orientation, dict):
            raise ValueError("inputs.orientation must map section filenames to "
                             "{flip, rotation_deg}")
        for name, value in orientation.items():
            record = _section(state, "orientation", cast("object", name))
            if not isinstance(value, dict) or set(value) - {"flip", "rotation_deg"}:
                raise ValueError(f"inputs.orientation[{name!r}] must be "
                                 "{flip: bool, rotation_deg: 0|90|180|270}")
            flip = value.get("flip")
            if flip is not None:
                if not isinstance(flip, bool):
                    raise ValueError(f"inputs.orientation[{name!r}].flip must be true or false")
                record.flip = flip
            rotation = value.get("rotation_deg")
            if rotation is not None:
                if isinstance(rotation, bool) or not isinstance(rotation, (int, float)) or (
                        rotation % 360 not in ROTATIONS):
                    raise ValueError(f"inputs.orientation[{name!r}].rotation_deg must be one "
                                     f"of {list(ROTATIONS)}")
                record.rotation_deg = int(rotation % 360)
        state.notes.append(f"inputs: {len(orientation)} orientation(s) set by the host")

    transforms = inputs.get("transforms") or {}
    if transforms:
        for name, value in transforms.items():
            record = _section(state, "transforms", cast("object", name))
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            record.transform = _host_affine(cast("object", value))
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # A host's word on a section's damage (a note, no regions): the agent
        # reads it in status and marks the regions it names.
        for name, note in damaged.items():
            record = _section(state, "damaged", cast("object", name))
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} damage note(s) from the host")

    frozen = spec.frozen
    if frozen:
        unknown = freeze.unknown_names(frozen, {record.id for record in state.slices})
        if unknown:
            raise ValueError(f"frozen names sections that are not here: {unknown}")
        held = freeze.sections_of(frozen, [record.id for record in state.in_order()])
        unplaced = [name for name, props in held.items()
                    if "position" in props and (rec := state.by_id(name)) is not None
                    and rec.position_mm is None]
        if unplaced:
            raise ValueError(
                f"frozen: the position of {unplaced} is frozen, and a position has no "
                "default: supply it in inputs.positions")
        for name, props in held.items():
            record = _section(state, "frozen", name)
            if "affine" in props and record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(held)} section(s) with a frozen property "
            "(the agent may not change it)")


class InputsChanged(ValueError):
    """A job folder's checkpoint was made from other supplied inputs
    (``JobSpec.inputs``) than the spec opening it: resuming would keep the
    old ones and drop the new without a word (:meth:`Job.open`)."""


#: How a caller starts over instead of resuming (the CLI flag and the spec field).
START_FRESH = "start a fresh job instead (`--fresh` on the command line, resume=False in a spec)"


def changed_inputs(saved_spec: Any, spec: JobSpec) -> list[str]:
    """The ``inputs`` keys, and ``frozen``, whose values differ between a
    checkpoint's spec (*saved_spec*, the dict it stores) and *spec*; none
    when the checkpoint records no spec inputs."""
    if not isinstance(saved_spec, dict) or "inputs" not in saved_spec:
        return []

    # a checkpoint's spec, read as JSON
    saved = cast("dict[str, object]", saved_spec)
    inputs = saved.get("inputs")
    saved_inputs: Mapping[str, Json] = {}
    if isinstance(inputs, dict):
        saved_inputs = cast("dict[str, Json]", inputs)
    old = to_jsonable(saved_inputs)
    new = to_jsonable(spec.inputs or {})
    keys = set(old) | set(new)
    changed = [key for key in keys if old.get(key) != new.get(key)]
    if freeze.clean(saved.get("frozen")) != freeze.clean(spec.frozen):
        changed.append("frozen")
    return sorted(changed)


def inputs_changed(folder: str | os.PathLike[str], changed: list[str]) -> InputsChanged:
    """The refusal for resuming the job in *folder* with other *changed* inputs."""
    return InputsChanged(
        f"The job in {folder} was made from other supplied inputs "
        f"({', '.join(changed)} differ); resuming it would ignore the new "
        f"ones. Open it with the inputs it was made from, or {START_FRESH}."
    )
