"""The stack state: the checkpoint, the result, and what every tool writes.

One object, one JSON shape, three uses. Corrections are DATA — ``flip``,
``rotation_deg`` and ``index_corrected`` describe a view over the discovered
stack, and the user's image files are never modified.

Order and position are separate fields that must agree at submit: reordering
changes only ``index_corrected``, and ``submit`` refuses positions that are not
monotone along the corrected order. Direction is not the agent's concern: the
sections of a stack cut posterior-first are identical to one cut
anterior-first, so at submit :func:`normalize_to_atlas_order` reverses a
descending stack, and the emitted corrected order always runs the atlas way
(positions increasing from the atlas origin end).
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from langslice.core.stack.section import (
    ANGLES_KEY,
    Angles,
    MixedAngles,
    SliceState,
    angles_tuple,
    flat_angles,
)


def plane_angles(state: StackState, angles: Angles | None) -> Angles:
    """*angles* as floats, or, when None, the stack's one angle
    (:attr:`StackState.stack_angles`, which refuses a stack whose sections
    differ). The rule every atlas-plane picture takes its plane by: a
    section's own angles for a picture of that section, the stack's view
    angles for one without."""
    return state.stack_angles if angles is None else (float(angles[0]), float(angles[1]))


@dataclass
class StackState:
    """The whole stack: inputs, per-section records, and run bookkeeping."""

    image_folder: str = ""
    atlas: str = ""
    plane: str = "coronal"
    #: Optional protocol values; None means not supplied.
    interval_mm: float | None = None
    thickness_mm: float | None = None
    #: The job's settings as run, as a plain mapping.
    spec: dict[str, Any] = field(default_factory=dict)
    #: Corrected indices of the sections AFTER a gap the agent concluded is real.
    interval_breaks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slices: list[SliceState] = field(default_factory=list)
    submitted: bool = False
    #: The agent's post-submit debrief (what it reached for that was not
    #: there), verbatim. Data for the environment's builders, not for the run.
    debrief: str = ""

    # --- views -----------------------------------------------------------

    def in_order(self) -> list[SliceState]:
        """The stack as corrected: sections sorted by ``index_corrected``."""
        return sorted(self.slices, key=lambda s: s.index_corrected)

    def by_id(self, slice_id: str) -> SliceState | None:
        return next((s for s in self.slices if s.id == slice_id), None)

    def resolve(self, ref: object) -> SliceState | None:
        """The section *ref* names: its filename, or the filename without its
        extension when exactly one section has that stem. Never a number:
        sections are named by filename only (:func:`unknown_sections`)."""
        text = str(ref).strip()
        hit = self.by_id(text)
        if hit is not None:
            return hit
        stems = [record for record in self.slices if filename_stem(record.id) == text]
        return stems[0] if len(stems) == 1 else None

    # --- cutting angles ---------------------------------------------------
    #
    # Each section carries its own plane (``SliceState.cutting_angles_deg``).
    # Everything LangSlice angles itself is one plane for the whole stack;
    # a registration supplied per section may differ section by section.

    @property
    def mixed_angles(self) -> bool:
        """Whether the sections' cutting angles differ."""
        return len({record.angles for record in self.slices}) > 1

    @property
    def stack_angles(self) -> Angles:
        """The one ``(pitch, yaw)`` every section shares (flat for an empty
        stack); :class:`MixedAngles` when they differ, so a reader that
        should use each section's own angle cannot take one silently."""
        planes = {record.angles for record in self.slices}
        if len(planes) > 1:
            raise MixedAngles(
                "The sections of this stack have different cutting angles; "
                "read each section's own (SliceState.angles).")
        return next(iter(planes), (0.0, 0.0))

    @property
    def view_angles(self) -> Angles:
        """The plane an atlas picture without a section is drawn at
        (``look``'s atlas, the opening's atlas reference): the stack's one
        angle, or, when the sections differ, the median pitch and the median
        yaw of its sections."""
        if not self.mixed_angles:
            return self.stack_angles
        pitches = [record.pitch_deg for record in self.slices]
        yaws = [record.yaw_deg for record in self.slices]
        return float(statistics.median(pitches)), float(statistics.median(yaws))

    def drawn_at_median(self, angles: Angles) -> bool:
        """Whether an atlas picture without a section drawn at *angles* is
        drawn at the median of differing sections' angles (:attr:`view_angles`),
        which its caption must say."""
        return self.mixed_angles and (float(angles[0]), float(angles[1])) == self.view_angles

    @property
    def cutting_angles_deg(self) -> dict[str, float]:
        """The stack-wide angles as ``{"pitch", "yaw"}`` (a copy; assign to
        change them); :class:`MixedAngles` when the sections differ."""
        self.stack_angles  # noqa: B018 - raises for a mixed stack
        if not self.slices:
            return flat_angles()
        return dict(self.slices[0].cutting_angles_deg)

    @cutting_angles_deg.setter
    def cutting_angles_deg(self, angles: dict[str, Any]) -> None:
        """Set every section to *angles* (``position_sections``): a stack
        whose sections differed now has one plane."""
        pitch, yaw = angles_tuple(angles)
        for record in self.slices:
            record.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}

    @property
    def pitch_deg(self) -> float:
        """The stack-wide pitch (:attr:`stack_angles`)."""
        return self.stack_angles[0]

    @property
    def yaw_deg(self) -> float:
        """The stack-wide yaw (:attr:`stack_angles`)."""
        return self.stack_angles[1]

    @property
    def is_oblique(self) -> bool:
        """Whether any section's plane is angled."""
        return any(record.is_oblique for record in self.slices)

    # --- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict. Same shape for checkpoint and hand-back.

        The cutting angles are written once, on the stack
        (``"cutting_angles_deg": {"pitch", "yaw"}``), when every section
        shares them, so a single-angle state reads as it always has; when
        they differ (state format 3), the stack's is ``null`` and each
        section row carries its own.
        """
        data = asdict(self)
        rows = data["slices"]
        if self.mixed_angles:
            data[ANGLES_KEY] = None
        else:
            data[ANGLES_KEY] = (dict(rows[0][ANGLES_KEY]) if rows else flat_angles())
            for row in rows:
                row.pop(ANGLES_KEY, None)
        for row in rows:
            for optional in ("layers", "masks", "images"):
                if not row.get(optional):  # most jobs have none: the state reads as it always has
                    row.pop(optional, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackState:
        """Rebuild from :meth:`to_dict` output; ``ValueError`` naming any
        unknown key (a section row's as ``slices.key``).

        A section row without its own angles carries the stack's
        (``cutting_angles_deg`` on the stack, flat without), which is how a
        single-angle state is written.
        """
        slice_fields = SliceState.__dataclass_fields__
        stack_fields = {k for k in cls.__dataclass_fields__ if k != "slices"}
        unknown = {k for k in data if k not in stack_fields | {"slices", ANGLES_KEY}}
        unknown |= {f"slices.{k}" for row in data.get("slices", []) for k in row
                    if k not in slice_fields}
        if unknown:
            raise ValueError(f"Unknown stack state key(s) {sorted(unknown)}")
        stack = data.get(ANGLES_KEY)
        slices = []
        for row in data.get("slices", []):
            kwargs = dict(row)
            kwargs["damaged_regions"] = [str(name) for name in
                                         kwargs.get("damaged_regions") or ()]
            own = kwargs.get(ANGLES_KEY)
            kwargs[ANGLES_KEY] = dict(own if isinstance(own, dict) else
                                      stack if isinstance(stack, dict) else flat_angles())
            slices.append(SliceState(**kwargs))
        kwargs = {k: v for k, v in data.items() if k in stack_fields}
        return cls(slices=slices, **kwargs)

    def restore(self, data: dict[str, Any]) -> None:
        """Overwrite this stack's contents from *data*, in place.

        Tools close over one state object, so undo/redo has to refill it
        rather than swap it.
        """
        other = StackState.from_dict(data)
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(other, name))


def normalize_to_atlas_order(state: StackState) -> bool:
    """Make the corrected order run the atlas way (positions increasing).

    Called at submit after the monotonicity gate has passed. When the placed
    positions decrease along the corrected order, every ``index_corrected``
    is mirrored (``n - 1 - i``) and each recorded interval break, the index of
    the section after a gap, is mapped to the section after that same gap in
    the new order (``n - i``). Returns True when the stack was reversed.
    """
    ordered = state.in_order()
    placed = [s for s in ordered if s.position_mm is not None]
    if len(placed) < 2:
        return False
    first = float(placed[0].position_mm)  # type: ignore[arg-type]
    last = float(placed[-1].position_mm)  # type: ignore[arg-type]
    if last >= first:
        return False
    n = len(ordered)
    for record in ordered:
        record.index_corrected = n - 1 - record.index_corrected
    state.interval_breaks = sorted(n - i for i in state.interval_breaks if 0 < i < n)
    return True


# --- naming sections ----------------------------------------------------------------
#
# A section is named by its filename everywhere a model, a script or a user
# names one (:meth:`StackState.resolve`); the stack order is internal.

#: How many filenames a refusal lists before it says how many it left out.
NAMES_LISTED = 40


def filename_stem(name: str) -> str:
    """*name* without its extension (``s01.tif`` -> ``s01``)."""
    base, dot, _extension = name.rpartition(".")
    return base if dot and base else name


def break_ids(state: StackState) -> list[str]:
    """The interval breaks as the filenames of the sections after each gap
    (stored as corrected indices, ``StackState.interval_breaks``)."""
    by_index = {record.index_corrected: record.id for record in state.slices}
    return [by_index[index] for index in state.interval_breaks if index in by_index]


def unknown_sections(state: StackState, refs: Any) -> dict[str, Any]:
    """The facts of a refusal for section names *refs* that name no section:
    ``unknown`` (the refs as text), ``filenames`` (the valid names, the
    first :data:`NAMES_LISTED`) and ``message``, which says that sections
    are named by filename (a number names none; a stem two files share
    needs the whole filename)."""
    unknown = [str(ref) for ref in (refs if isinstance(refs, (list, tuple)) else [refs])]
    names = sorted(record.id for record in state.slices)
    listed = names[:NAMES_LISTED]
    numbers = [ref for ref in unknown if ref.strip().lstrip("-").isdigit()]
    shared = [ref for ref in unknown
              if sum(filename_stem(name) == ref.strip() for name in names) > 1]
    words = [f"No section is named {', '.join(repr(ref) for ref in unknown)}. Sections are "
             "named by filename (or the filename without its extension, when no other "
             "section shares it)."]
    if numbers:
        words.append("Sections have no numbers to name them by.")
    if shared:
        words.append(f"{', '.join(repr(ref) for ref in shared)}: more than one file has "
                     "this name without its extension; give the whole filename.")
    words.append("Filenames: " + ", ".join(listed)
                 + (f" and {len(names) - len(listed)} more (status lists every section)."
                    if len(names) > len(listed) else "."))
    return {"unknown": unknown, "filenames": listed, "message": " ".join(words)}
