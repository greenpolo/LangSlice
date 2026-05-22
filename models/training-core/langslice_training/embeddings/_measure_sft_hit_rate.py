"""Measure atlas/query cache hit rate against an SFT-format JSONL corpus.

Walks every ``query_image_paths`` entry plus every
``trace[*].tool_result.image_paths`` entry across the corpus, attempts
``lookup_by_path`` against ``AtlasEmbeddingCache`` first, then
``QueryEmbeddingCache``, and prints a per-source hit rate breakdown.

Invocation::

    python -m langslice_training.embeddings._measure_sft_hit_rate \\
        --corpus models/langslice-gemma-4/data/sft_examples.jsonl \\
        --atlas-cache out/atlas_embeddings \\
        --query-cache out/query_embeddings

The atlas-cache flag is optional — pass only one cache to measure that
side. Both unset is a no-op (every path is a miss).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure cache hit rate against an SFT JSONL corpus.",
    )
    p.add_argument("--corpus", type=Path, required=True,
                   help="SFT JSONL corpus (sft_examples.jsonl or iSFT round_*.jsonl).")
    p.add_argument("--atlas-cache", type=Path, default=None,
                   help="Directory holding <atlas>_<plane>.pt files.")
    p.add_argument("--query-cache", type=Path, default=None,
                   help="Directory holding <plane>__<dataset>.pt files.")
    return p.parse_args(argv)


def _iter_paths(corpus: Path):
    with corpus.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("line %d: skipping malformed JSON", lineno)
                continue
            for p in row.get("query_image_paths", []) or []:
                yield "query", p
            for step in row.get("trace", []) or []:
                tr = step.get("tool_result") or {}
                for p in tr.get("image_paths", []) or []:
                    yield "atlas_grid", p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    atlas_cache = None
    if args.atlas_cache is not None:
        from langslice_training.embeddings.cache import AtlasEmbeddingCache
        atlas_cache = AtlasEmbeddingCache(args.atlas_cache)
        logger.info(
            "loaded atlas cache from %s: %d pair(s)",
            args.atlas_cache, len(atlas_cache.pairs()),
        )

    query_cache = None
    if args.query_cache is not None:
        from langslice_training.embeddings.query_cache import QueryEmbeddingCache
        query_cache = QueryEmbeddingCache(args.query_cache)
        logger.info(
            "loaded query cache from %s: %d pair(s)",
            args.query_cache, len(query_cache.pairs()),
        )

    counts: Counter[str] = Counter()
    for source, path in _iter_paths(args.corpus):
        counts[f"{source}/total"] += 1
        emb = atlas_cache.lookup_by_path(path) if atlas_cache is not None else None
        if emb is not None:
            counts[f"{source}/atlas_hit"] += 1
            counts["overall/atlas_hit"] += 1
            counts["overall/total"] += 1
            continue
        emb = query_cache.lookup_by_path(path) if query_cache is not None else None
        if emb is not None:
            counts[f"{source}/query_hit"] += 1
            counts["overall/query_hit"] += 1
            counts["overall/total"] += 1
            continue
        counts[f"{source}/miss"] += 1
        counts["overall/miss"] += 1
        counts["overall/total"] += 1

    overall_total = counts["overall/total"]
    overall_hits = counts["overall/atlas_hit"] + counts["overall/query_hit"]
    print()
    print("======== hit-rate report ========")
    print(f"corpus: {args.corpus}")
    print(f"atlas-cache: {args.atlas_cache}")
    print(f"query-cache: {args.query_cache}")
    print()
    print(f"overall  total={overall_total} hits={overall_hits} "
          f"({overall_hits / max(1, overall_total):.1%})")
    print(f"  atlas-hits: {counts['overall/atlas_hit']}")
    print(f"  query-hits: {counts['overall/query_hit']}")
    print(f"  misses:     {counts['overall/miss']}")
    for source in ("query", "atlas_grid"):
        tot = counts[f"{source}/total"]
        ah = counts[f"{source}/atlas_hit"]
        qh = counts[f"{source}/query_hit"]
        m = counts[f"{source}/miss"]
        if tot == 0:
            continue
        print(f"  {source:10s} total={tot} atlas={ah} query={qh} "
              f"miss={m} hit-rate={(ah + qh) / tot:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
