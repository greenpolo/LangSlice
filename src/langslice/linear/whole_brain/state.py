"""Shared state for whole-brain estimation: the config, the stack, and its slices.

The engine is a sequence of nodes over one :class:`StackState`. Every node
reads and writes this object; the checkpoint file is its JSON serialization,
and the hand-back results file is the *same* shape, so hosts read one schema.

Correction principle: flips and reorders are DATA. The user's image files are
never modified. ``index_corrected`` and ``flip`` describe a view over the
discovered stack, and every later step is expected to look through that view
(see :meth:`StackState.in_order`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# position_source values, in rough order of authority.
POSITION_SOURCES = ("", "survey", "deepslice", "anchor", "refined")
CONFIDENCE_LEVELS = ("", "low", "medium", "high")


@dataclass
class BrainConfig:
    """Host-agnostic request for one whole-brain run.

    The CLI fills this from flags; other hosts (ABBA and friends) fill the
    same shape over the wire.
    """

    image_folder: str
    atlas: str = "allen_mouse_25um"
    plane: str = "coronal"
    thickness_um: int = 50
    interval_um: int = 200
    keep_order: bool = True
    model: str | None = None
    out: str | None = None
    resume: bool = True

    @property
    def thickness_mm(self) -> float:
        return self.thickness_um / 1000.0

    @property
    def interval_mm(self) -> float:
        return self.interval_um / 1000.0


@dataclass
class SliceState:
    """Everything the engine knows or proposes about one slice image.

    Nothing here is applied to the image on disk — positions, flips, angles
    and transforms are proposals the host applies with its own machinery.
    """

    id: str
    index_original: int
    index_corrected: int
    flip: bool = False
    damaged: bool = False
    damage_note: str = ""
    position_mm: float | None = None
    position_source: str = ""
    affine: list[float] | None = None
    interactive_transform: dict[str, float] | None = None
    confidence: str = ""
    caveats: list[str] = field(default_factory=list)


@dataclass
class StackState:
    """The whole stack: inputs, per-slice records, and engine bookkeeping."""

    image_folder: str = ""
    atlas: str = ""
    plane: str = "coronal"
    # Per-axis direction of the *stack* (e.g. {"ap": "anterior_to_posterior"}).
    # Populated by the survey step; empty means "not determined yet".
    axis_directions: dict[str, str] = field(default_factory=dict)
    interval_mm: float = 0.0
    thickness_mm: float = 0.0
    keep_order: bool = True
    oblique_angles_deg: dict[str, float] | None = None
    # Corrected indices where the observed spacing breaks the nominal interval.
    interval_breaks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    contact_sheet: str = ""
    slices: list[SliceState] = field(default_factory=list)
    completed_nodes: list[str] = field(default_factory=list)
    node_cycles: dict[str, int] = field(default_factory=dict)

    # --- views -----------------------------------------------------------

    def in_order(self) -> list[SliceState]:
        """The stack as corrected: slices sorted by ``index_corrected``."""
        return sorted(self.slices, key=lambda s: s.index_corrected)

    def by_id(self, slice_id: str) -> SliceState | None:
        return next((s for s in self.slices if s.id == slice_id), None)

    # --- bookkeeping -----------------------------------------------------

    def mark_complete(self, node: str) -> None:
        if node not in self.completed_nodes:
            self.completed_nodes.append(node)

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
