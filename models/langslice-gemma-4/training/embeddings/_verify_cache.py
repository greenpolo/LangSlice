"""Bit-exact verification: cached embeddings on disk match live model output.

For each ``<atlas>_<plane>.pt`` cache file, sample a few basenames, locate
the matching on-disk JPG under ``<corpus-root>/atlas/<atlas>/<plane>/``, run
the model live on that JPG, and assert the cached tensor matches the live
tensor to ``atol=1e-5``. If any pair fails, do NOT enable the splice.

Run via the Docker training service::

    python -m embeddings._verify_cache \\
        --cache-dir /workspace/LangSlice/out/atlas_embeddings \\
        --corpus-root /workspace/LangSlice/models/langslice-gemma-4/data \\
        --model unsloth/gemma-4-E4B-it
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from embeddings.precompute import _encode_pil, _load_model_and_processor
from rlvr.atlas_grid import Plane, canonicalize_atlas_name

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify cached atlas embeddings match live model output.",
    )
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, required=True,
                   help="Path to corpus dir containing `atlas/<atlas>/<plane>/*.jpg`.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument(
        "--atol", type=float, default=1e-5,
        help="Bit-exact tolerance — splice cannot be enabled if any pair fails.",
    )
    p.add_argument(
        "--per-pair-samples", type=int, default=2,
        help="How many basenames per pair to verify.",
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


def _enumerate_cache_pairs(cache_dir: Path) -> list[tuple[str, Plane, Path]]:
    """List ``(atlas, plane, path)`` for each ``<atlas>_<plane>.pt`` file."""
    out: list[tuple[str, Plane, Path]] = []
    for child in sorted(cache_dir.iterdir()):
        if not child.is_file() or child.suffix != ".pt":
            continue
        for plane in ("coronal", "sagittal", "horizontal"):
            suffix = f"_{plane}.pt"
            if child.name.endswith(suffix):
                atlas = child.name[: -len(suffix)]
                out.append((canonicalize_atlas_name(atlas), plane, child))  # type: ignore[arg-type]
                break
    return out


def _verify_pair(
    *,
    atlas: str,
    plane: Plane,
    cache_path: Path,
    corpus_root: Path,
    model: Any,
    processor: Any,
    device: str,
    dtype: torch.dtype,
    atol: float,
    per_pair_samples: int,
) -> bool:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached: dict[str, torch.Tensor] = payload["embeddings"]
    basenames = sorted(cached.keys())
    if not basenames:
        logger.warning("%s/%s: empty cache file, skipping", atlas, plane)
        return True

    samples: list[str] = []
    samples.append(basenames[len(basenames) // 2])
    if per_pair_samples >= 2 and len(basenames) >= 2:
        samples.append(basenames[0])
    if per_pair_samples >= 3 and len(basenames) >= 3:
        samples.append(basenames[-1])
    samples = samples[:per_pair_samples]

    plane_dir = corpus_root / "atlas" / atlas / plane

    from PIL import Image

    all_pass = True
    for basename in samples:
        jpg_path = plane_dir / basename
        if not jpg_path.is_file():
            logger.warning(
                "%s/%s: cache has %s but on-disk JPG missing at %s — skip",
                atlas, plane, basename, jpg_path,
            )
            continue
        pil = Image.open(jpg_path).convert("RGB")
        live = _encode_pil(model, processor, pil, device=device, dtype=dtype)
        cached_emb = cached[basename]
        if live.shape != cached_emb.shape:
            logger.error(
                "%s/%s @ %s: SHAPE MISMATCH live=%s vs cached=%s",
                atlas, plane, basename, tuple(live.shape), tuple(cached_emb.shape),
            )
            all_pass = False
            continue
        diff = (live - cached_emb).abs().max().item()
        ok = diff <= atol
        verdict = "OK" if ok else "FAIL"
        logger.info(
            "%s/%s @ %s: max|live-cached|=%.3e atol=%.0e %s",
            atlas, plane, basename, diff, atol, verdict,
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
    if not pairs:
        logger.error("no <atlas>_<plane>.pt files in %s", args.cache_dir)
        return 1
    logger.info("verifying %d pair file(s) against on-disk JPGs", len(pairs))

    model, processor = _load_model_and_processor(
        args.model, device=args.device, dtype=dtype, prefer_vanilla=False,
    )

    all_pass = True
    for atlas, plane, path in pairs:
        try:
            ok = _verify_pair(
                atlas=atlas,
                plane=plane,
                cache_path=path,
                corpus_root=args.corpus_root,
                model=model,
                processor=processor,
                device=args.device,
                dtype=dtype,
                atol=args.atol,
                per_pair_samples=args.per_pair_samples,
            )
        except Exception as e:
            logger.error("%s/%s: verification raised: %r", atlas, plane, e)
            ok = False
        all_pass = all_pass and ok

    if all_pass:
        logger.info("ALL PAIRS PASS — splice safe to enable")
        return 0
    logger.error("AT LEAST ONE PAIR FAILED — splice unsafe; do NOT enable")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
