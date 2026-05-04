"""Legacy scaffold for comparison-triplet training data.

The active SFT plan has moved to deployment-shaped tool traces. If this scaffold
is revived, treat triplets as optional auxiliary data, not as the primary
Gemma 4 E4B training format.

Each legacy example is a comparison task:
  - Query image (atlas slice or augmented "synthetic histology")
  - Reference A (atlas slice at known position)
  - Reference B (atlas slice at known position)
  - Expected output: compact position estimate, not full reasoning

Usage:
    python models/langslice-gemma-4/data/build_triplets.py \
        --atlas-slices models/langslice-gemma-4/data/atlas_slices \
        --output models/langslice-gemma-4/data/triplets.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build comparison triplets for training")
    p.add_argument("--atlas-slices", required=True, help="Directory of extracted atlas slices")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument(
        "--triplets-per-atlas",
        type=int,
        default=500,
        help="Triplets to generate per atlas",
    )
    p.add_argument(
        "--max-ref-distance-mm",
        type=float,
        default=2.0,
        help="Max distance between references",
    )
    return p.parse_args()


def build_triplets(
    atlas_slices_dir: Path,
    output_path: Path,
    triplets_per_atlas: int = 500,
    max_ref_distance_mm: float = 2.0,
):
    """Generate comparison triplets from atlas slices.

    Strategy:
    - For each atlas/plane, sample query slices across the valid position range
    - For each query, pick two reference slices that bracket it
    - Vary reference spacing (easy: 2mm apart, hard: 0.3mm apart)
    - Apply augmentations to query images to simulate histological variation

    Augmentations (simulate real tissue):
    - Gaussian noise, contrast jitter, color shifts
    - Small affine warps (simulate tissue deformation)
    - Random masking (simulate tears/damage)
    - CLAHE / histogram normalization variation
    """
    # TODO: Implement triplet generation
    # Output format (JSONL):
    # {
    #   "atlas": "allen_mouse_25um",
    #   "query_image": "path/to/query.png",
    #   "query_position_mm": 5.2,
    #   "ref_a_image": "path/to/ref_a.png",
    #   "ref_a_position_mm": 5.0,
    #   "ref_b_image": "path/to/ref_b.png",
    #   "ref_b_position_mm": 5.5,
    #   "rationale": null,  # Optional compact caption/rationale
    #   "difficulty": "medium"
    # }
    raise NotImplementedError("Triplet generation pending implementation")


if __name__ == "__main__":
    args = _parse_args()
    build_triplets(
        atlas_slices_dir=Path(args.atlas_slices),
        output_path=Path(args.output),
        triplets_per_atlas=args.triplets_per_atlas,
        max_ref_distance_mm=args.max_ref_distance_mm,
    )
