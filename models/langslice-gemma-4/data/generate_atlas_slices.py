"""Legacy scaffold for extracting slice images from BrainGlobe atlases.

The active SFT plan covers coronal, sagittal, and horizontal position
estimation. Any revived extractor should sample the requested plane's
atlas-native position axis, not AP-only coronal sections.

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
    p.add_argument("--step-mm", type=float, default=0.1, help="Position step size in mm")
    p.add_argument(
        "--atlases",
        nargs="*",
        default=None,
        help="Specific atlases (default: all available)",
    )
    return p.parse_args()


def generate_slices(output_dir: Path, step_mm: float, atlas_names: list[str] | None = None):
    """Extract slices from BrainGlobe atlases.

    For each atlas, generates:
    - Colored region slices (RGB, for visual comparison)
    - Annotation/boundary slices (for region identification)
    - Metadata JSON with plane, position coordinates, and region labels per slice
    """
    # TODO: Implement atlas slice extraction
    # 1. List available BrainGlobe atlases (or use provided list)
    # 2. For each atlas:
    #    a. Load via bg_atlasapi
    #    b. Iterate the plane's position range at step_mm intervals
    #    c. Extract colored region slice via langslice.atlas
    #    d. Save image + metadata (position mm, visible regions, areas)
    raise NotImplementedError("Atlas slice extraction pending implementation")


if __name__ == "__main__":
    args = _parse_args()
    generate_slices(
        output_dir=Path(args.output_dir),
        step_mm=args.step_mm,
        atlas_names=args.atlases,
    )
