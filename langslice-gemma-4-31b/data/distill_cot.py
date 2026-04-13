"""Distill chain-of-thought reasoning from Gemini for training examples.

Takes comparison triplets and asks Gemini to produce anatomical reasoning
explaining why the query image matches a particular AP position relative
to the references.

Usage:
    python -m langslice_gemma_4_31b.data.distill_cot \
        --triplets langslice-gemma-4-31b/data/triplets.jsonl \
        --output langslice-gemma-4-31b/data/triplets_with_cot.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill CoT reasoning from Gemini")
    p.add_argument("--triplets", required=True, help="Input JSONL with triplets")
    p.add_argument("--output", required=True, help="Output JSONL with CoT added")
    p.add_argument("--model", default="gemini-2.5-pro-preview-05-06", help="Gemini model for distillation")
    p.add_argument("--max-parallel", type=int, default=5, help="Parallel API calls")
    return p.parse_args()


async def distill_reasoning(
    triplets_path: Path,
    output_path: Path,
    model: str,
    max_parallel: int = 5,
):
    """Generate CoT reasoning for each triplet via Gemini.

    Prompt strategy:
    - Show Gemini the query image + two reference images with their AP positions
    - Ask it to explain which anatomical features indicate where the query
      falls relative to the references
    - Extract the reasoning as the SFT target for Gemma fine-tuning

    The reasoning should be species-aware and reference visible structures,
    their relative sizes/shapes, and how they change along the AP axis.
    """
    # TODO: Implement Gemini distillation
    # 1. Load triplets from JSONL
    # 2. For each triplet, construct a comparison prompt with images
    # 3. Call Gemini API to get reasoning
    # 4. Write triplet + reasoning to output JSONL
    raise NotImplementedError("Gemini CoT distillation pending implementation")


if __name__ == "__main__":
    import asyncio

    args = _parse_args()
    asyncio.run(
        distill_reasoning(
            triplets_path=Path(args.triplets),
            output_path=Path(args.output),
            model=args.model,
            max_parallel=args.max_parallel,
        )
    )
