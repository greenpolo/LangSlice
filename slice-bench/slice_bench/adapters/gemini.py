"""Gemini API adapter — runs the existing LangSlice whole-brain pipeline."""

from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """Adapter that delegates to the LangSlice whole-brain pipeline via Gemini."""

    name = "gemini-langslice"

    def __init__(self, coarse_model: str | None = None, fine_model: str | None = None):
        self.coarse_model = coarse_model
        self.fine_model = fine_model

    async def estimate_positions(
        self,
        images: list[Path],
        atlas: str,
    ) -> dict[str, float]:
        from langslice.whole_brain.pipeline import run_brain_estimation
        from langslice.whole_brain.types import BrainEstimationConfig

        config = BrainEstimationConfig(
            image_folder=str(images[0].parent),
            atlas_name=atlas,
            thickness_um=50,
            interval_um=200,
            n_anchors=4,
            max_parallel=10,
            z_axis="AP",
            coarse_model=self.coarse_model,
            fine_model=self.fine_model,
        )

        result = await run_brain_estimation(config)
        return {s.filename: s.position_mm for s in result.slices}
