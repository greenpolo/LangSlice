"""Legacy scaffold for Gemini reasoning distillation.

The active SFT plan no longer trains full chain-of-thought as the default
target. Gemini rationale summaries are collected by the trace recorder and may
be converted into compact auxiliary captions or fallback rationale traces if
thinking-off deployment SFT fails.

Usage:
    python models/langslice-gemma-4/data/distill_cot.py \
        --triplets models/langslice-gemma-4/data/triplets.jsonl \
        --output models/langslice-gemma-4/data/triplets_with_cot.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legacy Gemini rationale distillation scaffold")
    p.add_argument("--triplets", required=True, help="Input JSONL with triplets")
    p.add_argument("--output", required=True, help="Output JSONL with rationale/caption text added")
    p.add_argument(
        "--model",
        default="gemini-2.5-pro-preview-05-06",
        help="Gemini model for distillation",
    )
    p.add_argument("--max-parallel", type=int, default=5, help="Parallel API calls")
    return p.parse_args()


async def distill_reasoning(
    triplets_path: Path,
    output_path: Path,
    model: str,
    max_parallel: int = 5,
):
    """Generate compact rationale/caption text for each triplet via Gemini.

    Prompt strategy:
    - Show Gemini the query image + references with their positions
    - Ask for short visible anatomical cues relevant to position estimation
    - Use the result only as auxiliary caption/rationale data, not the default
      deployment SFT target

    The reasoning should be species-aware and reference visible structures,
    their relative sizes/shapes, and how they change along the active axis.
    """
    # TODO: Implement Gemini distillation
    # 1. Load triplets from JSONL
    # 2. For each triplet, construct a comparison prompt with images
    # 3. Call Gemini API to get compact rationale/caption text
    # 4. Write triplet + reasoning to output JSONL
    raise NotImplementedError("Gemini rationale distillation pending implementation")


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
