"""Provider-agnostic result types for AP estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class APResult:
    position_mm: float
    reasoning: str
    debug_dir: str | None = None


@dataclass
class MultiSliceResult:
    """Result from multi-slice group estimation."""

    positions: list[APResult]
    group_reasoning: str
    debug_dir: str | None = None
