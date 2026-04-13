"""Local Gemma adapter — runs fine-tuned Gemma for AP estimation."""

from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter


class GemmaLocalAdapter(BaseAdapter):
    """Adapter for locally-hosted Gemma 4 models (base or fine-tuned).

    Loads the model via transformers/Unsloth and runs inference locally.
    """

    name = "gemma-local"

    def __init__(self, model_path: str, quantization: str = "4bit"):
        self.model_path = model_path
        self.quantization = quantization
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        # TODO: Implement model loading via transformers or Unsloth
        raise NotImplementedError(
            "Gemma local inference not yet implemented. "
            "Will load from: {self.model_path}"
        )

    async def estimate_positions(
        self,
        images: list[Path],
        atlas: str,
    ) -> dict[str, float]:
        # TODO: Implement local Gemma inference for AP estimation
        # This will:
        # 1. Load atlas reference slices from BrainGlobe
        # 2. For each input image, construct comparison prompts
        # 3. Run inference through the local model
        # 4. Parse AP estimates from model output
        raise NotImplementedError("Gemma local adapter pending langslice-gemma-4-31b training")
