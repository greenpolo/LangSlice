"""Render the on-disk atlas slate JPGs that the SigLIP embedding cache expects.

The atlas embedding cache (``out/atlas_embeddings/<atlas>_<plane>.pt``) is
path-keyed by JPG basenames like ``0.20mm.jpg``. Lane B's prompt
construction reads those JPGs from disk before SigLIP's forward hook
substitutes the cached embedding back in. If the JPGs are missing the
prompt construction fails, even though the embeddings themselves are
present.

This script restores the JPGs by:
1. Walking ``out/atlas_embeddings/*.pt``.
2. For each cache, extracting the atlas name + plane + per-key positions.
3. Rendering each position via ``rlvr.atlas_grid.get_reference_slice``
   (which goes through ``langslice_harness.atlas.core`` and the cached
   BrainGlobe atlas).
4. Saving to ``data/atlas/<atlas>/<plane>/<basename>`` with the same
   basename the cache uses, so SigLIP cache hits at training time.

Usage::

    python -m langslice_training.embeddings.render_slates_from_cache \\
        --embedding-cache out/atlas_embeddings \\
        --output-root data/atlas

By default existing JPGs are skipped; pass ``--overwrite`` to force
re-render.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import torch

# Self-bootstrap for direct invocation (`python <this-file>` without
# PYTHONPATH configured). The script lives under
# models/training-core/langslice_training/embeddings/, so the package
# root (models/training-core) is __file__.parent.parent.parent.
_TRAINING_CORE = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_CORE) not in sys.path:
    sys.path.insert(0, str(_TRAINING_CORE))

from langslice_training.rl.common.atlas_grid import get_reference_slice, load_atlas  # noqa: E402

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^(-?\d+\.\d+)mm\.jpg$")


def _parse_position_from_key(key: str) -> float | None:
    m = _KEY_RE.match(key)
    if not m:
        return None
    return float(m.group(1))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("out/atlas_embeddings"),
        help="Directory of <atlas>_<plane>.pt files (default: out/atlas_embeddings).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/atlas"),
        help="Where to write <atlas>/<plane>/<X.XX>mm.jpg (default: data/atlas).",
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


def _render_one_cache(
    cache_path: Path, output_root: Path, *, overwrite: bool, quality: int,
) -> tuple[int, int, int]:
    """Render every JPG named in one .pt cache.

    Returns ``(rendered, skipped_existing, skipped_bad_key)``.
    """
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    atlas_name = cache["atlas"]
    plane = cache["plane"]
    embeddings = cache["embeddings"]

    out_dir = output_root / atlas_name / plane
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = load_atlas(atlas_name)

    rendered = 0
    skipped_existing = 0
    skipped_bad_key = 0

    keys = sorted(embeddings.keys())
    n = len(keys)
    t0 = time.monotonic()
    for i, key in enumerate(keys):
        position_mm = _parse_position_from_key(key)
        if position_mm is None:
            skipped_bad_key += 1
            continue

        out_path = out_dir / key
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            img = get_reference_slice(atlas, position_mm, plane=plane)
        except Exception as exc:
            logger.warning(
                "render failed for %s/%s @ %.2fmm: %s",
                atlas_name, plane, position_mm, exc,
            )
            skipped_bad_key += 1
            continue

        img.convert("RGB").save(out_path, format="JPEG", quality=quality, optimize=True)
        rendered += 1

        if rendered and rendered % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = rendered / elapsed if elapsed > 0 else 0
            logger.info(
                "  %s/%s: %d/%d rendered (%.1f img/s, ETA %.1fs)",
                atlas_name, plane, rendered, n - skipped_existing - skipped_bad_key,
                rate, (n - i) / rate if rate > 0 else 0,
            )

    return rendered, skipped_existing, skipped_bad_key


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    embedding_cache_dir: Path = args.embedding_cache
    output_root: Path = args.output_root

    if not embedding_cache_dir.is_dir():
        logger.error(
            "--embedding-cache %s does not exist or is not a directory", embedding_cache_dir,
        )
        return 2

    cache_files = sorted(embedding_cache_dir.glob("*.pt"))
    if not cache_files:
        logger.error("no *.pt files in %s", embedding_cache_dir)
        return 2

    logger.info("found %d embedding caches under %s", len(cache_files), embedding_cache_dir)
    logger.info("rendering JPGs to %s/<atlas>/<plane>/<X.XX>mm.jpg", output_root)
    logger.info("overwrite existing: %s", args.overwrite)

    total_rendered = 0
    total_skipped_existing = 0
    total_skipped_bad = 0
    t0 = time.monotonic()

    for i, cache_path in enumerate(cache_files):
        logger.info("[%d/%d] %s", i + 1, len(cache_files), cache_path.name)
        try:
            r, se, sb = _render_one_cache(
                cache_path, output_root,
                overwrite=args.overwrite, quality=args.quality,
            )
        except Exception as exc:
            logger.error("cache %s failed: %s", cache_path.name, exc)
            continue
        total_rendered += r
        total_skipped_existing += se
        total_skipped_bad += sb
        logger.info("  -> rendered=%d skipped_existing=%d skipped_bad=%d", r, se, sb)

    elapsed = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info("done in %.1fs", elapsed)
    logger.info("total rendered:        %d", total_rendered)
    logger.info("total skipped (exist): %d", total_skipped_existing)
    logger.info("total skipped (bad):   %d", total_skipped_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
