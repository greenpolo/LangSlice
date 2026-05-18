"""Measure atlas-embedding cache hit rate against a real SFT corpus.

Phase-1 gating tool for the atlas-embedding splice. Loads a JSONL corpus,
walks every tool_result image path in every example, and reports what
fraction of those paths snap to a cached embedding. The splice (Phase 2) is
worth shipping iff this rate clears 50% on the corpus the trainer will
actually consume.

This driver does NOT load the model — the cache discovery is path-based and
runs in seconds against the full corpus. Run it before kicking off the
heavyweight precompute job to confirm the cache pairs you intend to build
will cover most batch images.

Invocation::

    python models/langslice-gemma-4/training/tools/embeddings_hit_rate.py \\
        --corpus models/langslice-gemma-4/data/sft_examples.jsonl \\
        --cache-dir out/atlas_embeddings

If ``--cache-dir`` does not yet exist, the tool prints which (atlas, plane)
pairs would need precomputing, sorted by how many images each pair would
cover. Useful for sizing the precompute job before paying for the GPU
forward passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make the gemma training package importable when run directly from checkout.
_GEMMA4_TRAINING = Path(__file__).resolve().parents[1]
if str(_GEMMA4_TRAINING) not in sys.path:
    sys.path.insert(0, str(_GEMMA4_TRAINING))

from embeddings.cache import AtlasEmbeddingCache, parse_atlas_path  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-1 hit-rate measurement for the atlas embedding cache.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="JSONL of langslice-native trace examples to measure.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory where precomputed <atlas>_<plane>.pt files live "
             "(may be empty — the tool reports coverage shortfall).",
    )
    p.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Show the top-N (atlas, plane) pairs by image count.",
    )
    return p.parse_args(argv)


def _iter_image_paths(corpus_path: Path):
    """Yield every tool_result image path across the entire corpus.

    Skips query images (never atlas-grid hits) and the terminal submit step.
    """
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for step in row.get("trace", []):
                tr = step.get("tool_result")
                if not tr:
                    continue
                yield from tr.get("image_paths", [])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cache = AtlasEmbeddingCache(args.cache_dir)
    known_pairs = set(cache.pairs())

    total = 0
    atlas_grid = 0
    cache_hits = 0
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_misses: Counter[tuple[str, str]] = Counter()
    non_atlas = 0

    for img_path in _iter_image_paths(args.corpus):
        total += 1
        parts = parse_atlas_path(img_path)
        if parts is None:
            non_atlas += 1
            continue
        atlas_grid += 1
        pair_counts[(parts.atlas, parts.plane)] += 1
        emb = cache.lookup_by_path(img_path)
        if emb is None:
            pair_misses[(parts.atlas, parts.plane)] += 1
        else:
            cache_hits += 1

    if total == 0:
        print(f"corpus {args.corpus} contains no tool-result image paths")
        return 1

    print(f"corpus: {args.corpus}")
    print(f"cache_dir: {args.cache_dir} ({len(known_pairs)} pairs known)")
    print(f"total tool-result images: {total}")
    print(f"  atlas-grid pattern matches: {atlas_grid} ({atlas_grid/total:.1%})")
    print(f"  non-atlas paths (queries, augmentations): {non_atlas}")
    if atlas_grid > 0:
        hit_rate = cache_hits / atlas_grid
        overall = cache_hits / total
        print(f"  cache hits: {cache_hits} ({hit_rate:.1%} of atlas-grid, {overall:.1%} overall)")
    print()
    print(f"top {args.top_pairs} (atlas, plane) pairs by image count:")
    for (atlas, plane), n in pair_counts.most_common(args.top_pairs):
        miss = pair_misses[(atlas, plane)]
        status = "CACHED" if (atlas, plane) in known_pairs else "MISSING"
        print(f"  {atlas:50s} {plane:10s} {n:6d} images  ({miss} miss)  [{status}]")
    print()
    if atlas_grid > 0:
        rate = cache_hits / atlas_grid
        print(f"verdict: {'SHIP' if rate > 0.5 else 'HOLD' if rate > 0.2 else 'DROP'} "
              f"(atlas-grid hit rate {rate:.1%}; gate is 50%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
