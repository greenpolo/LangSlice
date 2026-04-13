"""Abstract base adapter for AP estimation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAdapter(ABC):
    """Interface that all model adapters must implement.

    An adapter wraps a specific model backend (Gemini API, local Gemma,
    etc.) and exposes a uniform estimation interface to the benchmark.
    """

    name: str  # Human-readable model identifier

    @abstractmethod
    async def estimate_positions(
        self,
        images: list[Path],
        atlas: str,
    ) -> dict[str, float]:
        """Estimate AP positions for a batch of slice images.

        Args:
            images: Paths to histological slice images (ordered anterior→posterior).
            atlas: BrainGlobe atlas name (e.g. "allen_mouse_25um").

        Returns:
            Mapping of {filename: estimated_ap_mm} for each image.
        """
        ...
