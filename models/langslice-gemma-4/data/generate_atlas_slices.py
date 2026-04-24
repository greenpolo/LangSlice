"""Extract coronal slice images from all BrainGlobe atlases.

Iterates over available BrainGlobe atlases and extracts coronal sections
at regular AP intervals, saving them as images for training data generation.

Usage:
    python models/langslice-gemma-4/data/generate_atlas_slices.py \
        --output-dir models/langslice-gemma-4/data/atlas_slices \
        --step-mm 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract atlas slices from BrainGlobe")
    p.add_argument("--output-dir", required=True, help="Output directory for slice images")
    p.add_argument("--step-mm", type=float, default=0.1, help="AP step size in mm")
    p.add_argument("--atlases", nargs="*", default=None, help="Specific atlases (default: all available)")
    return p.parse_args()


def generate_slices(output_dir: Path, step_mm: float, atlas_names: list[str] | None = None):
    """Extract coronal slices from BrainGlobe atlases.

    For each atlas, generates:
    - Colored region slices (RGB, for visual comparison)
    - Annotation/boundary slices (for region identification)
    - Metadata JSON with AP coordinates and region labels per slice
    """
    # TODO: Implement atlas slice extraction
    # 1. List available BrainGlobe atlases (or use provided list)
    # 2. For each atlas:
    #    a. Load via bg_atlasapi
    #    b. Iterate AP range at step_mm intervals
    #    c. Extract colored region slice via langslice.atlas
    #    d. Save image + metadata (AP mm, visible regions, areas)
    raise NotImplementedError("Atlas slice extraction pending implementation")


if __name__ == "__main__":
    args = _parse_args()
    generate_slices(
        output_dir=Path(args.output_dir),
        step_mm=args.step_mm,
        atlas_names=args.atlases,
    )
