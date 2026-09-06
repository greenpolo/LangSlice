"""The stack state: the checkpoint, the result, and what every tool writes.

One object, one JSON shape, three uses. Corrections are DATA — ``flip``,
``rotation_deg`` and ``index_corrected`` describe a view over the discovered
stack, and the user's image files are never modified.

Order and position are separate fields that must agree at submit: reordering a
section that already has a position clears that position (it was assigned under
the wrong neighbours), and ``submit`` refuses positions that are not monotone
along the corrected order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CONFIDENCE_LEVELS = ("", "low", "medium", "high")

#: Quarter-turns the render applies before the flip.
ROTATIONS = (0, 90, 180, 270)


@dataclass
class SliceState:
    """Everything the run knows or proposes about one section image.

    ``transform`` is one dict, whatever produced it::

        {"kind": "silhouette" | "elastix" | "interactive",
         "params": [a, b, tx, c, d, ty],   # normalized 2x3, see langslice.affine
         "iou": float | None,              # fits only
         "note": str}                      # interactive only

    Coordinates are NORMALIZED — x as a fraction of image width, y as a
    fraction of image height — so the same six numbers apply at any resolution
    (:func:`langslice.affine.normalized_affine`). The transform describes the
    section as the run sees it, i.e. AFTER ``rotation_deg`` and ``flip``.
    """

    id: str
    index_original: int
    index_corrected: int
    #: Mirror left-right. Applied AFTER ``rotation_deg``.
    flip: bool = False
    #: Quarter-turn applied before the flip; one of :data:`ROTATIONS`.
    rotation_deg: int = 0
    #: Agent-internal: excludes the section from DeepSlice and the automatic
    #: affine. Never a user option.
    damaged: bool = False
    damage_note: str = ""
    position_mm: float | None = None
    confidence: str = ""
    transform: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)


def apply_confidence(record: SliceState, value: object) -> bool:
    """Set *record*'s confidence from model output; ignore anything else.

    "The model left it out" must not read as "the model said no confidence":
    an omitted or unrecognised value leaves the current one standing. Returns
    whether it was applied.
    """
    level = str(value or "").strip().lower()
    if not level or level not in CONFIDENCE_LEVELS:
        return False
    record.confidence = level
    return True


def add_caveat(record: SliceState, caveat: str) -> None:
    """Append *caveat* to a section once."""
    if caveat not in record.caveats:
        record.caveats.append(caveat)


@dataclass
class StackState:
    """The whole stack: inputs, per-section records, and run bookkeeping."""

    image_folder: str = ""
    atlas: str = ""
    plane: str = "coronal"
    interval_mm: float = 0.0
    thickness_mm: float = 0.0
    #: The JobSpec as run (:meth:`langslice.linear.spec.JobSpec.to_dict`).
    spec: dict[str, Any] = field(default_factory=dict)
    #: Stack-wide cutting angles; 0/0 means a flat plane.
    cutting_angles_deg: dict[str, float] = field(
        default_factory=lambda: {"pitch": 0.0, "yaw": 0.0}
    )
    #: Corrected indices of the sections AFTER a gap the agent concluded is real.
    interval_breaks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slices: list[SliceState] = field(default_factory=list)
    submitted: bool = False

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

    @property
    def pitch_deg(self) -> float:
        return float(self.cutting_angles_deg.get("pitch", 0.0))

    @property
    def yaw_deg(self) -> float:
        return float(self.cutting_angles_deg.get("yaw", 0.0))

    @property
    def is_oblique(self) -> bool:
        return bool(self.pitch_deg or self.yaw_deg)

    # --- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict. Same shape for checkpoint and hand-back."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackState:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        slice_fields = SliceState.__dataclass_fields__
        slices = [
            SliceState(**{k: v for k, v in row.items() if k in slice_fields})
            for row in data.get("slices", [])
        ]
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
