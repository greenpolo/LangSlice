"""Build the overnight vision-on training corpus: distilled + multi-turn synth.

Reads:
  - models/langslice-gemma-4/data/sft_examples.jsonl  (1716 filtered Gemini distilled)
  - data/manifest/allocations/<plane>/rlvr.jsonl + shards/<plane>/*.jsonl  (synth pool)

Writes:
  - <out_dir>/combined_corpus.jsonl  (distilled + N synth, shuffled)
  - <out_dir>/build_report.txt       (counts + plane mix)

Usage (host-side):
  python models/langslice-gemma-4/training/tools/build_overnight_corpus.py \\
      --out out/synth/overnight_<TS>/ \\
      --n-synth 12000 \\
      --seed 42

Multi-turn floor is hard-pinned to 1.0 (every synth row gets the full
fetch_atlas → ... → submit_estimate prefix).

All paths are relative to the repo root; resolves cwd up if needed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import time
from pathlib import Path


# Locate repo root
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not locate LangSlice repo root from overnight-corpus tool path."
    raise RuntimeError(msg)


HERE = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root()
SYNTHDATA_ROOT = REPO_ROOT / "models" / "synthdata"

# Add synthdata + src roots for local package imports.
sys.path.insert(0, str(SYNTHDATA_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_overnight_corpus")


def _generate_chunk(args_tuple):
    """Worker entrypoint for ProcessPoolExecutor.

    Each worker process receives a chunk of specs and a derived seed, runs the
    synth iterator over its chunk, and returns the rows. sys.path injection
    must run again here because the worker's interpreter starts fresh under
    spawn semantics (Windows/macOS) — fork copies the parent's sys.path but
    spawn does not.
    """
    chunk_id, specs_chunk, seed, mt_floor, atlas_grid_cache_dir, max_total_images, mode = args_tuple

    sys.path.insert(0, str(SYNTHDATA_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from synthdata.sft_corpus import generate_synthetic_rows

    worker_rng = random.Random(seed)
    t0 = time.monotonic()
    rows = generate_synthetic_rows(
        specs_chunk,
        multi_turn_floor=mt_floor,
        rng=worker_rng,
        atlas_grid_cache_dir=(
            atlas_grid_cache_dir if atlas_grid_cache_dir and atlas_grid_cache_dir.is_dir() else None
        ),
        max_total_images=max_total_images,
        synthetic_reasoning_mode=mode,
    )
    elapsed = time.monotonic() - t0
    return chunk_id, rows, elapsed


_DISTILLED_IMG_ROOT = "models/langslice-gemma-4/data"


def _rewrite_distilled_path(p: str) -> str:
    """Rewrite a distilled row's image path to a repo-rooted form.

    Distilled corpus uses relative paths like 'queries/foo.jpg' and
    'atlas/<atlas>/<plane>/<pos>.jpg', which were valid when the corpus lived
    next to its 'queries/' and 'atlas/' sibling dirs at
    models/langslice-gemma-4/data/. Our combined corpus lives at a different
    parent dir, so these wouldn't resolve via dataset.py's primary root.
    Rewriting them to 'models/langslice-gemma-4/data/queries/foo.jpg' etc
    makes them resolve via the validator's repo-root fallback (the codepath
    that's already used for canonical 'data/datasets/...' paths).
    """
    if p.startswith("queries/") or p.startswith("atlas/"):
        return f"{_DISTILLED_IMG_ROOT}/{p}"
    return p


def _rewrite_distilled_row(row: dict) -> dict:
    row["query_image_paths"] = [
        _rewrite_distilled_path(p) for p in row.get("query_image_paths", [])
    ]
    for evt in row.get("trace", []):
        if isinstance(evt, dict) and "tool_result" in evt:
            tr = evt["tool_result"]
            paths = tr.get("image_paths", [])
            if isinstance(paths, list):
                tr["image_paths"] = [_rewrite_distilled_path(p) for p in paths]
    return row


def load_distilled(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(_rewrite_distilled_row(json.loads(line)))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--n-synth", type=int, default=12000, help="Target synth row count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--distilled",
        type=Path,
        default=REPO_ROOT / "models/langslice-gemma-4/data/sft_examples.jsonl",
    )
    p.add_argument(
        "--allocation-root",
        type=Path,
        default=REPO_ROOT / "data/manifest",
        help="Root containing allocations/<plane>/rlvr.jsonl + shards/<plane>/<ds>.jsonl",
    )
    p.add_argument(
        "--atlas-grid-cache-dir",
        type=Path,
        default=REPO_ROOT / "out/cache_fast/atlas",
        help=(
            "Directory of precomputed SigLIP atlas embeddings; basenames encode "
            "available atlas positions. Used to build the fetch_atlas position "
            "grid for synthetic traces."
        ),
    )
    p.add_argument(
        "--max-total-images",
        type=int,
        default=8,
        help="Per-row cap on atlas images (drops farthest-from-GT first).",
    )
    p.add_argument(
        "--multi-turn-floor",
        type=float,
        default=1.0,
        help="Fraction of synth rows forced multi-turn (rest are direct-submit).",
    )
    p.add_argument(
        "--synthetic-reasoning-mode",
        choices=("region_dump", "thinking_signature"),
        default="region_dump",
        help=(
            "Synthetic row target: legacy submit reasoning dumps, or Gemma "
            "thinking rows with compact atlas signatures and turn-split "
            "multi-turn examples."
        ),
    )
    p.add_argument(
        "--no-distilled",
        action="store_true",
        help="Skip loading the Gemini distilled traces; emit synth only.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help=(
            "Number of worker processes for parallel synth-row generation. "
            "Defaults to half the logical CPU count (skip hyperthreaded siblings). "
            "Set to 1 to disable the pool entirely. Each worker re-imports "
            "synthdata.sft_corpus and rebuilds its own atlas-grid cache on first "
            "use per (atlas, plane), so worker startup adds ~5-10 sec/worker; "
            "amortizes well above ~500 specs per worker."
        ),
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 1. Load distilled corpus -----------------------------------------
    from collections import Counter
    if args.no_distilled:
        distilled = []
        distilled_planes = Counter()
        logger.info("--no-distilled: skipping Gemini distilled traces")
    else:
        if not args.distilled.is_file():
            logger.error("Distilled corpus missing: %s", args.distilled)
            return 2
        distilled = load_distilled(args.distilled)
        logger.info("Loaded %d distilled rows from %s", len(distilled), args.distilled)
        distilled_planes = Counter(r.get("plane", "?") for r in distilled)
        logger.info("Distilled plane mix: %s", dict(distilled_planes))

    # ---- 2. Build synth specs from allocation manifest --------------------
    from synthdata.sft_corpus import (
        generate_synthetic_rows,
        load_specs_from_allocation,
    )

    # Pull from all three planes; final count is args.n_synth across union.
    specs_by_plane: dict[str, list] = {}
    for plane in ("coronal", "sagittal", "horizontal"):
        plane_specs = load_specs_from_allocation(
            args.allocation_root,
            plane,
            n=None,           # full eligible pool — sample after union
            rng=rng,
        )
        specs_by_plane[plane] = plane_specs
        logger.info("Loaded %d %s specs from allocation pool", len(plane_specs), plane)

    union_specs: list = [s for plane in specs_by_plane.values() for s in plane]
    logger.info("Union pool size: %d", len(union_specs))

    if args.n_synth > len(union_specs):
        logger.warning(
            "n_synth=%d > pool=%d; capping to pool",
            args.n_synth, len(union_specs),
        )
        target_n = len(union_specs)
    else:
        target_n = args.n_synth

    # Uniform sample without replacement
    sampled_specs = rng.sample(union_specs, target_n)
    spec_planes = Counter(s.plane for s in sampled_specs)
    logger.info("Sampled synth plane mix: %s", dict(spec_planes))

    # ---- 3. Generate synth rows -------------------------------------------
    mt_floor = float(args.multi_turn_floor)
    n_workers = max(1, int(args.n_workers))
    # Cap workers at the spec count so we don't spawn idle processes for tiny
    # corpora (e.g. smoke n=200 with default n_workers=12).
    if n_workers > target_n:
        n_workers = max(1, target_n)
    logger.info(
        "Generating %d synth rows across %d worker(s) (multi_turn_floor=%.2f, mode=%s)...",
        target_n,
        n_workers,
        mt_floor,
        args.synthetic_reasoning_mode,
    )

    if n_workers == 1:
        # Single-process path: keep the legacy behavior so tiny builds skip the
        # pool overhead and the seed-derivation path. Same rng instance the
        # rest of main() uses (and shuffles with).
        synth_rows = generate_synthetic_rows(
            sampled_specs,
            multi_turn_floor=mt_floor,
            rng=rng,
            atlas_grid_cache_dir=(
                args.atlas_grid_cache_dir if args.atlas_grid_cache_dir.is_dir() else None
            ),
            max_total_images=args.max_total_images,
            synthetic_reasoning_mode=args.synthetic_reasoning_mode,
        )
    else:
        # Round-robin chunking gives every worker a balanced plane mix (specs
        # arrived in plane order from load_specs_from_allocation, so a
        # contiguous-slice approach would dump all 24k coronal on workers 0-7
        # and leave the trailing sagittal/horizontal on a single worker).
        chunks: list[list] = [sampled_specs[i::n_workers] for i in range(n_workers)]
        # Derive per-worker seeds from the main rng so the build remains
        # deterministic given --seed. Workers must not share the same rng
        # state across processes.
        chunk_seeds = [rng.randint(0, 2**31 - 1) for _ in range(n_workers)]
        atlas_cache_dir = (
            args.atlas_grid_cache_dir if args.atlas_grid_cache_dir.is_dir() else None
        )
        arg_tuples = [
            (
                i,
                chunks[i],
                chunk_seeds[i],
                mt_floor,
                atlas_cache_dir,
                args.max_total_images,
                args.synthetic_reasoning_mode,
            )
            for i in range(n_workers)
        ]
        # Process pool — spawn semantics on Windows; each worker re-imports
        # everything in _generate_chunk. as_completed lets us log progress as
        # workers finish, but we collect into a chunk_id-keyed dict so the
        # final row order matches the chunk ordering (preserves determinism
        # for the subsequent shuffle: rng state at shuffle time depends on
        # input order).
        per_chunk_rows: dict[int, list[dict]] = {}
        t0 = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_generate_chunk, args_tup) for args_tup in arg_tuples]
            for fut in concurrent.futures.as_completed(futures):
                chunk_id, rows, elapsed = fut.result()
                per_chunk_rows[chunk_id] = rows
                done = len(per_chunk_rows)
                logger.info(
                    "  worker %d done: %d rows in %.1f sec (%d/%d chunks)",
                    chunk_id, len(rows), elapsed, done, n_workers,
                )
        wall = time.monotonic() - t0
        synth_rows = [row for cid in range(n_workers) for row in per_chunk_rows[cid]]
        logger.info(
            "Parallel synth phase wall time: %.1f sec (~%.2f sec/spec on %d workers)",
            wall, wall / max(1, target_n), n_workers,
        )
    logger.info("Generated %d synth rows", len(synth_rows))

    # ---- 4. Concatenate + shuffle ----------------------------------------
    combined = list(distilled) + list(synth_rows)
    rng.shuffle(combined)
    logger.info("Combined corpus: %d rows", len(combined))

    out_path = args.out / "combined_corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in combined:
            fh.write(json.dumps(row) + "\n")
    logger.info("Wrote %s", out_path)

    # ---- 5. Report --------------------------------------------------------
    combined_planes = Counter(r.get("plane", "?") for r in combined)
    report = [
        f"distilled rows:        {len(distilled)}",
        f"synth rows (target):   {target_n}",
        f"synth rows (actual):   {len(synth_rows)}",
        f"combined rows:         {len(combined)}",
        f"distilled plane mix:   {dict(distilled_planes)}",
        f"synth plane mix:       {dict(spec_planes)}",
        f"combined plane mix:    {dict(combined_planes)}",
        f"seed:                  {args.seed}",
        f"multi_turn_floor:      {mt_floor:.2f}",
        f"synthetic_reasoning_mode: {args.synthetic_reasoning_mode}",
        f"max_total_images:      {args.max_total_images}",
    ]
    (args.out / "build_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
