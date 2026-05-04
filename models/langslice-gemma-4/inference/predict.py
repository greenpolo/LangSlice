"""Run position-estimation inference with a fine-tuned Gemma 4 E4B model.

Usage:
    python models/langslice-gemma-4/inference/predict.py \
        --model-path models/langslice-gemma-4/checkpoints/best \
        --image slice_001.png \
        --atlas allen_mouse_25um
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict slice position with fine-tuned Gemma 4 E4B")
    p.add_argument("--model-path", required=True, help="Path to fine-tuned model/adapter")
    p.add_argument("--image", required=True, help="Input slice image")
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--show-rationale",
        action="store_true",
        help="Print optional visible rationale if enabled",
    )
    return p.parse_args()


def predict(model_path: str, image_path: str, atlas: str, show_rationale: bool = False):
    """Run single-image position estimation with the fine-tuned model.

    Pipeline:
    1. Load target image and tool definitions
    2. Run the constrained fetch-atlas / submit-estimate loop
    3. Validate tool calls and final position output

    Thinking is off by default. Optional rationale display is only for fallback
    experiments, not the primary deployment path.
    """
    # TODO: Implement inference pipeline
    raise NotImplementedError("Inference pending model training")


if __name__ == "__main__":
    args = _parse_args()
    predict(
        model_path=args.model_path,
        image_path=args.image,
        atlas=args.atlas,
        show_rationale=args.show_rationale,
    )
