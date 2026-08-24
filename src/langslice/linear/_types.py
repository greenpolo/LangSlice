"""Provider-agnostic result types for position estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionResult:
    """Result from a single-slice position estimate.

    ``position_mm`` is measured along the slice-normal axis (AP for coronal,
    ML for sagittal, DV for horizontal) in atlas-native millimetres.
    """

    position_mm: float
    reasoning: str
    debug_dir: str | None = None


# Shared alias used by whole-brain estimation code.
APResult = PositionResult
