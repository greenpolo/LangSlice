"""Data classes for whole-brain AP estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrainEstimationConfig:
    """User-provided configuration for a whole-brain estimation run."""

    image_folder: str
    atlas_name: str
    thickness_um: int
    interval_um: int
    n_anchors: int
    ordering: str  # "strict" | "loose" | "none"
    refinement: bool
    max_parallel: int
    z_axis: str  # "AP" | "PA"

    @property
    def thickness_mm(self) -> float:
        return self.thickness_um / 1000.0

    @property
    def interval_mm(self) -> float:
        return self.interval_um / 1000.0


@dataclass
class SlicePosition:
    """Position state for a single slice."""

    filename: str
    index: int
    position_mm: float
    source: str  # "anchor", "interpolated", "extrapolated", "*+refined"
    locked: bool


@dataclass
class BrainEstimationSummary:
    """Summary statistics for a completed run."""

    mean_interval_mm: float
    std_interval_mm: float
    n_slices: int
    n_anchors: int
    n_refined: int
    n_skipped: int


@dataclass
class BrainEstimationResult:
    """Complete result of a whole-brain estimation run."""

    config: BrainEstimationConfig
    slices: list[SlicePosition]
    summary: BrainEstimationSummary

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "atlas": self.config.atlas_name,
            "thickness_um": self.config.thickness_um,
            "interval_um": self.config.interval_um,
            "ordering_mode": self.config.ordering,
            "z_axis": self.config.z_axis,
            "slices": [
                {
                    "filename": s.filename,
                    "position_mm": round(s.position_mm, 4),
                    "source": s.source,
                }
                for s in self.slices
            ],
        }
