"""Render every atlas reference JPG at each atlas's native voxel step.

Sister script to ``render_atlas_slates_from_cache.py``: instead of reading
positions out of an existing embedding cache, this enumerates positions at
the atlas's *own* native voxel resolution (the slicing-axis spacing of the
BrainGlobe volume) and renders every one. The result is a complete on-disk
slate that ``embeddings.precompute`` can then walk to build a 100%-cache-hit
SigLIP embedding store — at training time every curriculum-picked section
hits the cache regardless of position.

Native step per atlas comes from ``atlas.resolution`` (BrainGlobe), keyed on
the slicing axis:

    coronal   → resolution[0] (AP)
    horizontal→ resolution[1] (DV)
    sagittal  → resolution[2] (ML)

JPGs are saved to ``data/atlas/<atlas>/<plane>/<X.XX>mm.jpg`` using the same
``{position:.2f}mm.jpg`` filename convention the trainer / cache expect.

Usage::

    python -m embeddings.render_slates_native_step \\
        --manifest-root data/manifest \\
        --output-root data/atlas

By default the script discovers (atlas, plane) pairs from the manifest
shards. Pass ``--pairs allen_mouse_25um:coronal,whs_sd_rat_39um:sagittal``
to restrict. Existing JPGs are skipped unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Path bootstrap: rlvr lives under models/langslice-gemma-4/training/ which
# isn't a normal Python package (the dir name has a hyphen). This script
# lives under models/langslice-gemma-4/training/embeddings/, so the training
# root is __file__.parent.parent.
_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from rlvr.atlas_grid import get_reference_slice, load_atlas  # noqa: E402

logger = logging.getLogger(__name__)

_AXIS_INDEX = {"coronal": 0, "horizontal": 1, "sagittal": 2}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifest"),
        help="Manifest root for discovering (atlas, plane) pairs (default: data/manifest).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/atlas"),
        help="Where to write <atlas>/<plane>/<X.XX>mm.jpg (default: data/atlas).",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated 'atlas:plane' filter (default: every pair found in manifest).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render JPGs that already exist on disk (default: skip).",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality (1-100, default 92).",
    )
    return p.parse_args(argv)


def _discover_pairs_from_manifest(manifest_root: Path) -> list[tuple[str, str]]:
    """Return sorted unique (atlas, plane) pairs found in shards/*/*.jsonl."""
    pairs: set[tuple[str, str]] = set()
    shards_dir = manifest_root / "shards"
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"manifest shards dir not found: {shards_dir}")
    for shard in shards_dir.glob("*/*.jsonl"):
        if ".bak" in shard.name:
            continue
        with shard.open() as fh:
            for line in fh:
                row = json.loads(line)
                atlas = row.get("atlas")
                plane = row.get("orientation")
                if atlas and plane in _AXIS_INDEX:
                    pairs.add((atlas, plane))
    return sorted(pairs)


def _parse_pairs_filter(raw: str | None) -> set[tuple[str, str]] | None:
    if raw is None:
        return None
    out: set[tuple[str, str]] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--pairs entry must be 'atlas:plane', got {chunk!r}")
        atlas, plane = chunk.split(":", 1)
        plane = plane.strip()
        if plane not in _AXIS_INDEX:
            raise ValueError(
                f"--pairs plane must be coronal/sagittal/horizontal, got {plane!r}"
            )
        out.add((atlas.strip(), plane))
    return out or None


def _enumerate_native_positions(atlas, plane: str) -> list[float]:
    """Return every voxel-aligned position (in mm) along the slicing axis.

    The slicing axis is determined by ``plane`` per BrainGlobe's
    (z=AP, y=DV, x=ML) convention. Native step = resolution[axis] / 1000;
    extent = step * shape[axis].
    """
    axis = _AXIS_INDEX[plane]
    step_um = float(atlas.resolution[axis])
    n = int(atlas.shape[axis])
    step_mm = step_um / 1000.0
    return [round(i * step_mm, 6) for i in range(n)]


def _render_one_pair(
    atlas_name: str,
    plane: str,
    output_root: Path,
    *,
    overwrite: bool,
    quality: int,
) -> tuple[int, int, int]:
    """Render every native-step JPG for one (atlas, plane) pair.

    Returns (rendered, skipped_existing, skipped_failed).
    """
    out_dir = output_root / atlas_name / plane
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = load_atlas(atlas_name)
    positions = _enumerate_native_positions(atlas, plane)

    rendered = 0
    skipped_existing = 0
    skipped_failed = 0
    n = len(positions)
    t0 = time.monotonic()

    # Track filenames we've already written this run so we don't render
    # twice when two adjacent native positions snap to the same %.2f name.
    seen_names: set[str] = set()

    for i, position_mm in enumerate(positions):
        name = f"{position_mm:.2f}mm.jpg"
        if name in seen_names:
            # Two consecutive native positions collapsed into the same
            # 0.01mm-precision filename. Skip the duplicate.
            continue
        seen_names.add(name)

        out_path = out_dir / name
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            img = get_reference_slice(atlas, position_mm, plane=plane)
        except Exception as exc:
            logger.warning("render failed for %s/%s @ %.4fmm: %s", atlas_name, plane, position_mm, exc)
            skipped_failed += 1
            continue

        img.convert("RGB").save(out_path, format="JPEG", quality=quality, optimize=True)
        rendered += 1

        if rendered and rendered % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = rendered / elapsed if elapsed > 0 else 0
            done = i + 1
            remaining_est = (n - done) / rate if rate > 0 else 0
            logger.info(
                "  %s/%s: %d rendered (%d skipped existing); %.1f img/s; ETA %.0fs",
                atlas_name, plane, rendered, skipped_existing, rate, remaining_est,
            )

    return rendered, skipped_existing, skipped_failed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    pairs_filter = _parse_pairs_filter(args.pairs)

    discovered = _discover_pairs_from_manifest(args.manifest_root)
    if pairs_filter is not None:
        pairs = [p for p in discovered if p in pairs_filter]
    else:
        pairs = discovered

    if not pairs:
        logger.error("no (atlas, plane) pairs to render (filter: %s)", args.pairs)
        return 2

    logger.info("rendering %d (atlas, plane) pairs to %s", len(pairs), args.output_root)
    logger.info("overwrite existing: %s", args.overwrite)

    grand_t0 = time.monotonic()
    grand_rendered = 0
    grand_skipped_existing = 0
    grand_skipped_failed = 0

    for i, (atlas_name, plane) in enumerate(pairs):
        logger.info("[%d/%d] %s/%s", i + 1, len(pairs), atlas_name, plane)
        try:
            r, se, sf = _render_one_pair(
                atlas_name, plane, args.output_root,
                overwrite=args.overwrite, quality=args.quality,
            )
        except Exception as exc:
            logger.error("pair %s/%s failed: %s", atlas_name, plane, exc)
            continue
        grand_rendered += r
        grand_skipped_existing += se
        grand_skipped_failed += sf
        logger.info(
            "  -> rendered=%d skipped_existing=%d skipped_failed=%d",
            r, se, sf,
        )

    elapsed = time.monotonic() - grand_t0
    logger.info("=" * 60)
    logger.info("done in %.1fs", elapsed)
    logger.info("total rendered:        %d", grand_rendered)
    logger.info("total skipped (exist): %d", grand_skipped_existing)
    logger.info("total skipped (fail):  %d", grand_skipped_failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
