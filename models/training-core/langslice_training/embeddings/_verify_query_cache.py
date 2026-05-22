"""Bit-exact verification: cached query embeddings on disk match live model output.

Sibling of :mod:`langslice_training.embeddings._verify_cache` (which verifies the atlas
cache); this script samples a few entries from each ``<plane>__<dataset>.pt``
file, locates the matching on-disk slice image, runs the model live, and
asserts the cached tensor equals the live tensor to ``atol=1e-5``.

Why a separate script (not a shared helper)? The query cache uses the
trainer's actual preprocessing path (``rlvr.dataset.preprocess_query_image``
with atlas-aware downsample, CLAHE off), whereas the atlas cache encodes
already-rendered atlas reference images directly. Drift in the preprocess
defaults would silently break one or the other — keeping the verifiers
parallel surfaces that.

Run via the Docker training service::

    python -m langslice_training.embeddings._verify_query_cache \\
        --cache-dir /workspace/LangSlice/out/query_embeddings \\
        --manifest-root /workspace/LangSlice/data/manifest \\
        --repo-root /workspace/LangSlice \\
        --model unsloth/gemma-4-E4B-it
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify cached query embeddings match live model output.",
    )
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True,
                   help="Path to data/manifest (used to resolve atlas + plane for each section).")
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Repo root for resolving manifest-relative image paths.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument(
        "--atol", type=float, default=1e-5,
        help="Bit-exact tolerance — splice cannot be enabled if any pair fails.",
    )
    p.add_argument(
        "--per-pair-samples", type=int, default=2,
        help="How many image paths per (plane, dataset) pair to verify.",
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help="Optional cap on (plane, dataset) pairs to verify.",
    )
    return p.parse_args(argv)


def _parse_dtype(raw: str) -> torch.dtype:
    raw = raw.lower()
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    if raw in ("fp16", "float16"):
        return torch.float16
    if raw in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unknown dtype {raw!r}")


def _enumerate_cache_pairs(cache_dir: Path) -> list[tuple[str, str, Path]]:
    """List ``(plane, dataset, path)`` for each ``<plane>__<dataset>.pt`` file."""
    out: list[tuple[str, str, Path]] = []
    for child in sorted(cache_dir.iterdir()):
        if not child.is_file() or child.suffix != ".pt":
            continue
        stem = child.stem  # "<plane>__<dataset>"
        if "__" not in stem:
            continue
        plane, dataset = stem.split("__", 1)
        if plane not in ("coronal", "sagittal", "horizontal"):
            continue
        out.append((plane, dataset, child))
    return out


def _verify_pair(
    *,
    plane: str,
    dataset: str,
    cache_path: Path,
    manifest_root: Path,
    repo_root: Path,
    model: Any,
    processor: Any,
    device: str,
    dtype: torch.dtype,
    atol: float,
    per_pair_samples: int,
) -> bool:
    from langslice_training.embeddings.precompute import _encode_pil
    from langslice_training.rl.common.dataset import (
        _atlas_in_plane_long_edge,
        preprocess_query_image,
    )
    from langslice_training.rl.single_turn.manifest_index import ManifestIndex

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached: dict[str, torch.Tensor] = payload["embeddings"]
    paths = sorted(cached.keys())
    if not paths:
        logger.warning("%s/%s: empty cache file, skipping", plane, dataset)
        return True

    # Choose a few sample paths spread across the file.
    samples: list[str] = []
    samples.append(paths[len(paths) // 2])
    if per_pair_samples >= 2 and len(paths) >= 2:
        samples.append(paths[0])
    if per_pair_samples >= 3 and len(paths) >= 3:
        samples.append(paths[-1])
    samples = samples[:per_pair_samples]

    # Resolve atlas per section via the manifest index so preprocessing matches
    # the precompute path exactly (atlas-aware long-edge downsample).
    idx = ManifestIndex.from_manifest_root(manifest_root, repo_root=repo_root)
    sections = idx.query(plane=plane, atlas=None, split=None)
    path_to_atlas = {s.image_path: s.atlas for s in sections}

    all_pass = True
    for rel_path in samples:
        atlas = path_to_atlas.get(rel_path)
        if atlas is None:
            logger.warning(
                "%s/%s: cache has %s but no manifest row matches — skip",
                plane, dataset, rel_path,
            )
            continue
        on_disk = repo_root / rel_path
        if not on_disk.is_file():
            logger.warning(
                "%s/%s: cache has %s but on-disk file missing at %s — skip",
                plane, dataset, rel_path, on_disk,
            )
            continue
        try:
            atlas_long_edge = int(_atlas_in_plane_long_edge(atlas, plane))  # type: ignore[arg-type]
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning(
                "%s/%s @ %s: atlas_long_edge lookup failed (%s) — skip",
                plane, dataset, rel_path, exc,
            )
            continue
        pil = preprocess_query_image(
            on_disk, atlas_long_edge=atlas_long_edge, apply_clahe=False,
        )
        live = _encode_pil(model, processor, pil, device=device, dtype=dtype)
        if live.dim() > 2 and live.shape[0] == 1:
            live = live.squeeze(0)
        cached_emb = cached[rel_path]
        if cached_emb.dim() > 2 and cached_emb.shape[0] == 1:
            cached_emb = cached_emb.squeeze(0)
        if live.shape != cached_emb.shape:
            logger.error(
                "%s/%s @ %s: SHAPE MISMATCH live=%s vs cached=%s",
                plane, dataset, rel_path, tuple(live.shape), tuple(cached_emb.shape),
            )
            all_pass = False
            continue
        diff = (live - cached_emb).abs().max().item()
        ok = diff <= atol
        verdict = "OK" if ok else "FAIL"
        logger.info(
            "%s/%s @ %s: max|live-cached|=%.3e atol=%.0e %s",
            plane, dataset, rel_path, diff, atol, verdict,
        )
        all_pass = all_pass and ok

    return all_pass


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    dtype = _parse_dtype(args.dtype)
    pairs = _enumerate_cache_pairs(args.cache_dir)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        logger.error("no <plane>__<dataset>.pt files in %s", args.cache_dir)
        return 1
    logger.info("verifying %d pair file(s)", len(pairs))

    from langslice_training.embeddings.precompute import _load_model_and_processor
    model, processor = _load_model_and_processor(
        args.model, device=args.device, dtype=dtype, prefer_vanilla=False,
    )

    all_pass = True
    for plane, dataset, path in pairs:
        try:
            ok = _verify_pair(
                plane=plane,
                dataset=dataset,
                cache_path=path,
                manifest_root=args.manifest_root,
                repo_root=args.repo_root,
                model=model,
                processor=processor,
                device=args.device,
                dtype=dtype,
                atol=args.atol,
                per_pair_samples=args.per_pair_samples,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("%s/%s: verification raised: %r", plane, dataset, e)
            ok = False
        all_pass = all_pass and ok

    if all_pass:
        logger.info("ALL PAIRS PASS — query-cache splice safe to enable")
        return 0
    logger.error("AT LEAST ONE PAIR FAILED — splice unsafe; do NOT enable")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
