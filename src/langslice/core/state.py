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

#: Quarter-turns the render applies before the flip.
ROTATIONS = (0, 90, 180, 270)

#: The five physical knobs of the transform every section starts from, and
#: the B side of an A/B preview when a section carries nothing yet. (The
#: identity's six normalized numbers are ``core.affine.IDENTITY_PARAMS``.)
IDENTITY_KNOBS: dict[str, float] = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}

#: A plane's cutting angles, ``(pitch_deg, yaw_deg)``.
Angles = tuple[float, float]

#: The key cutting angles are stored under, on a section and (in the
#: serialized state, when every section shares one) on the stack.
ANGLES_KEY = "cutting_angles_deg"


def flat_angles() -> dict[str, float]:
    """The flat plane, ``{"pitch": 0.0, "yaw": 0.0}`` (a new dict)."""
    return {"pitch": 0.0, "yaw": 0.0}


def angles_tuple(angles: dict[str, Any] | None) -> Angles:
    """``(pitch, yaw)`` of a stored angles dict (missing parts are 0)."""
    angles = angles or {}
    return float(angles.get("pitch", 0.0)), float(angles.get("yaw", 0.0))


def plane_angles(state: StackState, angles: Angles | None) -> Angles:
    """*angles* as floats, or, when None, the stack's one angle
    (:attr:`StackState.stack_angles`, which refuses a stack whose sections
    differ). The rule every atlas-plane picture takes its plane by: a
    section's own angles for a picture of that section, the stack's view
    angles for one without."""
    return state.stack_angles if angles is None else (float(angles[0]), float(angles[1]))


def serialized_mixed_angles(data: dict[str, Any]) -> bool:
    """Whether a serialized state (:meth:`StackState.to_dict`, a checkpoint's
    fields at format 3) has sections whose cutting angles differ: the stack's
    ``cutting_angles_deg`` is written null exactly then."""
    return ANGLES_KEY in data and data[ANGLES_KEY] is None


class MixedAngles(ValueError):
    """The stack's sections have different cutting angles, so there is no
    one stack-wide angle to read (:attr:`StackState.stack_angles`)."""


@dataclass
class SliceState:
    """Everything the run knows or proposes about one section image.

    ``transform`` is one dict, whatever produced it::

        {"kind": "silhouette" | "elastix" | "interactive",
         "params": [a, b, tx, c, d, ty],   # normalized 2x3, see langslice.core.affine
         "physical": {rotation_deg, scale_x, scale_y, shear,
                      translate_x_mm, translate_y_mm, pivot},
         "calibration": {section_um_per_px, source},
         "iou": float | None,              # fits only
         "mirrored": bool,                 # det of the 2x2 is negative
         "note": str}                      # interactive only

    ``iou`` is always tissue-silhouette overlap, not anatomical quality.

    For an affine, ``physical`` is the ONE representation it carries, whatever
    made it: the five knobs the alignment tools take (plus the ``shear`` an
    affine can have and they cannot), about a pivot given as canvas
    fractions.

    Coordinates are NORMALIZED — x as a fraction of image width, y as a
    fraction of image height — so the same six numbers apply at any resolution
    (:func:`langslice.core.affine.normalized_affine`). The transform describes the
    section as the run sees it, i.e. AFTER ``rotation_deg`` and ``flip``.
    """

    id: str
    index_original: int
    index_corrected: int
    #: Mirror left-right. Applied AFTER ``rotation_deg``.
    flip: bool = False
    #: Quarter-turn applied before the flip; one of :data:`ROTATIONS`.
    rotation_deg: int = 0
    #: The atlas regions this section is missing, or has so badly displaced
    #: that they would wreck a fit (``mark_damage``): acronyms or ids, each
    #: optionally one side (``"CTX:left"``, :mod:`langslice.core.atlas.sides`).
    #: Every fit and the image model's trace leave them out
    #: (:func:`langslice.core.damage.exclusions`).
    damaged_regions: list[str] = field(default_factory=list)
    #: A damage mark that names no regions: the host's ``inputs.damaged``
    #: (``{filename: note}``, which the agent cannot remove) or a
    #: ``mark_damaged`` flag. Regions may be added beside it.
    damage_marked: bool = False
    damage_note: str = ""
    position_mm: float | None = None
    #: ``"default"`` while the position is the evenly spaced starting one
    #: the job gave the section at ingest (``job.job.default_positions``);
    #: empty once anything wrote it, or when it was supplied.
    position_source: str = ""
    transform: dict[str, Any] | None = None
    #: Raw image-model correction and artifacts; does not replace the transform.
    image_correction: dict[str, Any] | None = None
    #: The applied deformable fit (``fit_deformable``): a reference to its
    #: record on disk (``record``: the ``DeformableRecord`` directory, parent
    #: steps inside it) plus a summary. It lives on top of the linear
    #: placement it was fitted at (``linear_key``); any change to position,
    #: orientation, cutting angles or transform clears it.
    deformation: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)
    #: This section's own cutting angles, ``{"pitch": deg, "yaw": deg}``:
    #: the atlas plane every picture, fit and map of it is drawn at. Equal
    #: on every section of a stack LangSlice angled itself
    #: (``set_cutting_angles`` sets them all); a registration supplied per
    #: section keeps each its own (``inputs.angles``). Serialized on the
    #: stack when all sections share one (:meth:`StackState.to_dict`).
    cutting_angles_deg: dict[str, float] = field(default_factory=flat_angles)

    @property
    def damaged(self) -> bool:
        """Whether the section is marked damaged: it has marked regions, or a
        mark that names none (:attr:`damage_marked`). Read-only."""
        return bool(self.damaged_regions) or bool(self.damage_marked)

    @property
    def angles(self) -> Angles:
        """``(pitch_deg, yaw_deg)`` of this section's plane."""
        return angles_tuple(self.cutting_angles_deg)

    @property
    def pitch_deg(self) -> float:
        return self.angles[0]

    @property
    def yaw_deg(self) -> float:
        return self.angles[1]

    @property
    def is_oblique(self) -> bool:
        return bool(self.pitch_deg or self.yaw_deg)


@dataclass
class StackState:
    """The whole stack: inputs, per-section records, and run bookkeeping."""

    image_folder: str = ""
    atlas: str = ""
    plane: str = "coronal"
    interval_mm: float = 0.0
    thickness_mm: float = 0.0
    #: The JobSpec as run (:meth:`langslice.core.spec.JobSpec.to_dict`).
    spec: dict[str, Any] = field(default_factory=dict)
    #: Corrected indices of the sections AFTER a gap the agent concluded is real.
    interval_breaks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slices: list[SliceState] = field(default_factory=list)
    #: The agent's appearance settings (``preprocess`` tool), per target:
    #: ``{"view"|"fit": {"stack": settings, "sections": {id: settings}}}``.
    #: Empty means the default appearance everywhere. Undone and checkpointed
    #: with everything else (:mod:`langslice.core.appearance`).
    appearance: dict[str, Any] = field(default_factory=dict)
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
        """A section by filename or by corrected index (the two addresses)."""
        text = str(ref).strip()
        hit = self.by_id(text)
        if hit is not None:
            return hit
        try:
            index = int(text)
        except (TypeError, ValueError):
            return None
        return next((s for s in self.slices if s.index_corrected == index), None)

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
        (``view_atlas``, the opening's atlas reference): the stack's one
        angle, or, when the sections differ, the median pitch and the median
        yaw of its sections."""
        if not self.mixed_angles:
            return self.stack_angles
        pitches = [record.pitch_deg for record in self.slices]
        yaws = [record.yaw_deg for record in self.slices]
        return float(statistics.median(pitches)), float(statistics.median(yaws))

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
        """Set every section to *angles* (``set_cutting_angles``): a stack
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
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackState:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys.

        A section row without its own angles carries the stack's
        (``cutting_angles_deg`` on the stack, flat without), which is how a
        single-angle state and every state before format 3 read. A row
        saved with the older ``damaged`` flag and no ``damage_marked``
        reads as a mark that names no regions (its note kept).
        """
        slice_fields = SliceState.__dataclass_fields__
        stack = data.get(ANGLES_KEY)
        slices = []
        for row in data.get("slices", []):
            kwargs = {k: v for k, v in row.items() if k in slice_fields}
            if "damage_marked" not in row and row.get("damaged"):
                kwargs["damage_marked"] = True
            kwargs["damaged_regions"] = [str(name) for name in
                                         kwargs.get("damaged_regions") or ()]
            own = kwargs.get(ANGLES_KEY)
            kwargs[ANGLES_KEY] = dict(own if isinstance(own, dict) else
                                      stack if isinstance(stack, dict) else flat_angles())
            slices.append(SliceState(**kwargs))
        stack_fields = {k for k in cls.__dataclass_fields__ if k != "slices"}
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

