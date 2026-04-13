"""Run AP estimation inference with a fine-tuned langslice-gemma-4-31b model.

Usage:
    python -m langslice_gemma_4_31b.inference.predict \
        --model-path langslice-gemma-4-31b/checkpoints/best \
        --image slice_001.png \
        --atlas allen_mouse_25um
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict AP position with langslice-gemma-4-31b")
    p.add_argument("--model-path", required=True, help="Path to fine-tuned model/adapter")
    p.add_argument("--image", required=True, help="Input slice image")
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument("--show-reasoning", action="store_true", help="Print CoT reasoning")
    return p.parse_args()


def predict(model_path: str, image_path: str, atlas: str, show_reasoning: bool = False):
    """Run single-image AP estimation with the fine-tuned model.

    Pipeline:
    1. Load atlas reference slices from BrainGlobe
    2. Select bracketing references (coarse → fine)
    3. Construct comparison prompt with query + references
    4. Run inference through fine-tuned Gemma
    5. Parse AP estimate from model output
    """
    # TODO: Implement inference pipeline
    raise NotImplementedError("Inference pending model training")


if __name__ == "__main__":
    args = _parse_args()
    predict(
        model_path=args.model_path,
        image_path=args.image,
        atlas=args.atlas,
        show_reasoning=args.show_reasoning,
    )
