"""Precompute SigLIP embeddings for every query (slice) image in the manifest.

Sibling of ``embeddings.precompute`` (which caches atlas reference images);
this script caches the per-row query/slice images that the trainer's
single-turn-RL dataset consumes. The trainer's
``--query-embedding-cache`` flag picks up the resulting per-(plane, dataset)
``.pt`` files at startup.

Output schema
=============

One ``.pt`` file per ``(plane, dataset)`` pair under ``--output-root``::

    <output_root>/<plane>__<dataset>.pt

Each file contains a single torch-saved dict::

    {
        "plane": "coronal",
        "dataset": "ad_bxd",
        "embeddings": {full_image_path_str: torch.Tensor},
    }

``full_image_path_str`` is the repo-relative path exactly as it appears in
``data/manifest/shards/<plane>/<dataset>.jsonl`` (i.e.
``data/datasets/<plane>/<dataset>/<subject>/<section_id>.png``). Tensors are
the per-image SigLIP ``last_hidden_state`` (shape ``(num_patches, hidden)``,
matching the atlas precompute's per-image shape).

Why per-(plane, dataset) shards?
================================

* **Operability.** A 60 GB total cache as one file makes per-shard reruns
  prohibitive; per-pair files mean ``--max-rows`` smoke tests + per-shard
  reruns are cheap.
* **Symmetry with the atlas cache.** Both caches use the same ``one .pt
  per (key1, key2)`` layout, so ``QueryEmbeddingCache`` parallels
  ``AtlasEmbeddingCache``.
* **Idempotent reruns.** ``--skip-existing`` (default ON) honours per-pair
  files: re-running with the same args writes nothing for pairs whose
  ``.pt`` is already on disk.

Bit-exact match with trainer runtime
====================================

The encoder uses **the same** preprocessing the trainer does at sample
time:

1. ``rlvr.dataset.preprocess_query_image`` — atlas-aware downsample to
   ``_atlas_in_plane_long_edge(atlas, plane)``. CLAHE is OFF (matches the
   single-turn-RL dataset's ``apply_clahe=False`` call site).
2. ``processor.apply_chat_template`` (image-only message) — same processor
   path as ``embeddings.precompute._encode_pil``, so the SigLIP
   ``last_hidden_state`` shape and value match what the trainer's live
   forward would produce for the same image.

Invocation
==========

::

    python -m embeddings.precompute_query \\
        --manifest-root data/manifest \\
        --split rlvr \\
        --output-root out/query_embeddings \\
        --model unsloth/gemma-4-E4B-it \\
        --device cuda --dtype bf16

For a smoke test::

    python -m embeddings.precompute_query \\
        --manifest-root data/manifest --split rlvr \\
        --output-root out/query_embeddings_smoke \\
        --max-rows 50 --device cpu --dtype fp32

The driver groups manifest rows by ``(plane, dataset)``, so ``--max-rows``
caps the *total* number of encodes (some pairs may end up with zero rows).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Real brain slices can exceed PIL's default 178M-pixel decompression-bomb
# safety check (e.g. Allen connectivity sections at native resolution can be
# 16k×16k = 256M pixels). Disable the check — these are trusted manifest
# inputs, not adversarial uploads.
Image.MAX_IMAGE_PIXELS = None

# Path bootstrap: rlvr / single_turn_rl live under models/langslice-gemma-4/training/
# which isn't a normal Python package (the dir name has a hyphen). This script
# lives under models/langslice-gemma-4/training/embeddings/, so the training
# root is __file__.parent.parent.
_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dtype(raw: str) -> torch.dtype:
    raw = raw.lower()
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    if raw in ("fp16", "float16", "half"):
        return torch.float16
    if raw in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"--dtype must be bf16/fp16/fp32, got {raw!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute SigLIP embeddings for query images in the manifest.",
    )
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifest"),
        help="Root of the data manifest (containing shards/ + allocations/). "
        "Defaults to data/manifest.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="rlvr",
        help="Allocation split to encode (rlvr/sft/eval/all). Defaults to rlvr "
        "to mirror the trainer's default --data-source. Use 'all' to encode "
        "every section regardless of split.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write <plane>__<dataset>.pt cache files.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root used to resolve repo-relative image paths from the "
        "manifest. Defaults to the current working directory.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-4-E4B-it",
        help="HuggingFace model id or local path to load the vision tower from.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the SigLIP forward pass (cuda or cpu).",
    )
    p.add_argument(
        "--dtype",
        type=_parse_dtype,
        default=torch.bfloat16,
        help="Compute dtype for the vision tower (bf16 / fp16 / fp32).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Sanity limit: encode at most N total rows across all pairs "
        "(debug only).",
    )
    p.add_argument(
        "--only-pairs",
        type=str,
        default=None,
        help="Comma-separated 'plane:dataset' filter. Default: every "
        "(plane, dataset) the manifest reports for the chosen --split.",
    )
    p.add_argument(
        "--prefer-vanilla",
        action="store_true",
        help="Skip Unsloth's FastVisionModel and load via vanilla HF (debug).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="If a <plane>__<dataset>.pt file already exists, skip it (default ON).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Force re-encoding even if the per-pair .pt file already exists.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Row enumeration
# ---------------------------------------------------------------------------


def _parse_only_pairs(raw: str | None) -> set[tuple[str, str]] | None:
    if raw is None:
        return None
    out: set[tuple[str, str]] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--only-pairs entry must be 'plane:dataset', got {chunk!r}"
            )
        plane, dataset = chunk.split(":", 1)
        plane = plane.strip()
        dataset = dataset.strip()
        if plane not in ("coronal", "sagittal", "horizontal"):
            raise ValueError(
                f"--only-pairs plane must be coronal/sagittal/horizontal, got {plane!r}"
            )
        out.add((plane, dataset))
    return out or None


def _group_rows_by_pair(
    *,
    manifest_root: Path,
    repo_root: Path,
    split: str,
    only_pairs: set[tuple[str, str]] | None,
    max_rows: int | None,
) -> list[tuple[str, str, list[Any]]]:
    """Walk the manifest index and group sections by ``(plane, dataset)``.

    Sections with no on-disk image (typo, partial sync) are skipped silently
    with a count printed at the end. ``max_rows`` caps the *total* row
    count across all pairs.
    """
    from single_turn_rl.manifest_index import ManifestIndex

    idx = ManifestIndex.from_manifest_root(manifest_root, repo_root=repo_root)
    logger.info(
        "ManifestIndex loaded from %s with %d sections.", manifest_root, len(idx),
    )
    use_split: str | None = None if split.lower() == "all" else split

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    skipped_missing = 0
    total = 0
    for atlas, plane in idx.pairs():
        sections = idx.query(plane=plane, atlas=atlas, split=use_split)
        for section in sections:
            pair = (plane, section.dataset)
            if only_pairs is not None and pair not in only_pairs:
                continue
            on_disk = (repo_root / section.image_path).is_file()
            if not on_disk:
                skipped_missing += 1
                continue
            grouped[pair].append(section)
            total += 1
            if max_rows is not None and total >= int(max_rows):
                logger.info("--max-rows %d reached; stopping enumeration.", max_rows)
                if skipped_missing:
                    logger.warning(
                        "skipped %d section(s) whose image file was missing on disk.",
                        skipped_missing,
                    )
                return _to_sorted_list(grouped)

    if skipped_missing:
        logger.warning(
            "skipped %d section(s) whose image file was missing on disk.",
            skipped_missing,
        )
    return _to_sorted_list(grouped)


def _to_sorted_list(
    grouped: dict[tuple[str, str], list[Any]],
) -> list[tuple[str, str, list[Any]]]:
    return [
        (plane, dataset, sections)
        for (plane, dataset), sections in sorted(grouped.items())
    ]


# ---------------------------------------------------------------------------
# Encoder (mirrors embeddings.precompute._encode_pil)
# ---------------------------------------------------------------------------


def _encode_pil(model: Any, processor: Any, pil_image, *, device: str, dtype: torch.dtype):
    """Run the vision tower on one PIL image and return the SigLIP last_hidden_state.

    Identical to :func:`embeddings.precompute._encode_pil` so the cached
    embedding is bit-exact to what the trainer would produce live for the
    same preprocessed image. We don't reuse that helper directly to avoid a
    cyclic import dance in the test harness.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": ""},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_position_ids = inputs.get("image_position_ids")
    if image_position_ids is not None:
        image_position_ids = image_position_ids.to(device)
    with torch.inference_mode():
        result = model.get_image_features(pixel_values, image_position_ids)
    if hasattr(result, "last_hidden_state"):
        feats = result.last_hidden_state
    elif isinstance(result, tuple):
        feats = result[0]
    else:
        feats = result
    if feats.dim() > 2 and feats.shape[0] == 1:
        feats = feats.squeeze(0)
    return feats.detach().to("cpu")


# ---------------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------------


def _encode_pair(
    *,
    plane: str,
    dataset: str,
    sections: list[Any],
    repo_root: Path,
    model: Any,
    processor: Any,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Encode every section's query image; return ``{image_path: tensor}``.

    Uses :func:`rlvr.dataset.preprocess_query_image` (the trainer's actual
    preprocessing call site) so cached embeddings match runtime exactly:
    atlas-aware downsample to ``_atlas_in_plane_long_edge(atlas, plane)``,
    CLAHE OFF (the single-turn-RL dataset uses ``apply_clahe=False``).
    """
    from rlvr.dataset import _atlas_in_plane_long_edge, preprocess_query_image

    out: dict[str, torch.Tensor] = {}
    n = len(sections)
    for i, section in enumerate(sections):
        rel_path = section.image_path
        on_disk = repo_root / rel_path
        try:
            atlas_long_edge = int(_atlas_in_plane_long_edge(section.atlas, plane))  # type: ignore[arg-type]
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning(
                "  skipping %s/%s @ %s: atlas_long_edge lookup failed (%s)",
                plane, dataset, section.section_id, exc,
            )
            continue
        try:
            pil = preprocess_query_image(
                on_disk, atlas_long_edge=atlas_long_edge, apply_clahe=False,
            )
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            logger.warning(
                "  skipping %s/%s @ %s: preprocess failed (%s)",
                plane, dataset, section.section_id, exc,
            )
            continue
        try:
            emb = _encode_pil(model, processor, pil, device=device, dtype=dtype)
        except (RuntimeError, OSError) as exc:
            logger.warning(
                "  skipping %s/%s @ %s: encode failed (%s)",
                plane, dataset, section.section_id, exc,
            )
            continue
        # Key by the manifest's repo-relative path string — this is what
        # the trainer's ``image_paths`` column holds + what the cache
        # lookup will receive at sidecar-build time.
        out[rel_path] = emb
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(
                "  %s/%s: %d/%d (%s)",
                plane, dataset, i + 1, n, Path(rel_path).name,
            )
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def precompute_query_embeddings(
    pairs: Iterable[tuple[str, str, list[Any]]],
    *,
    repo_root: Path,
    output_root: Path,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    skip_existing: bool = True,
    prefer_vanilla: bool = False,
) -> dict[tuple[str, str], Path]:
    """Encode every (plane, dataset) shard and write per-pair cache files.

    ``skip_existing=True`` makes the driver idempotent — re-running with
    the same args after a partial completion only encodes the un-saved
    pairs. ``--no-skip-existing`` from the CLI flips this off.
    """
    from embeddings.precompute import _load_model_and_processor
    from embeddings.query_cache import save_query_pair

    pairs = list(pairs)
    if not pairs:
        logger.warning("no (plane, dataset) shards to encode; exiting.")
        return {}

    # Skip-existing pre-pass — avoid loading the model if every pair is done.
    pending: list[tuple[str, str, list[Any]]] = []
    skipped_existing = 0
    for plane, dataset, sections in pairs:
        out_path = output_root / f"{plane}__{dataset}.pt"
        if skip_existing and out_path.is_file():
            skipped_existing += 1
            logger.info(
                "skip %s/%s — already on disk at %s", plane, dataset, out_path,
            )
            continue
        pending.append((plane, dataset, sections))
    if skipped_existing and not pending:
        logger.info(
            "all %d pair file(s) already exist; nothing to do.", skipped_existing,
        )
        return {}

    model, processor = _load_model_and_processor(
        model_id, device=device, dtype=dtype, prefer_vanilla=prefer_vanilla,
    )
    saved: dict[tuple[str, str], Path] = {}
    for plane, dataset, sections in pending:
        t0 = time.perf_counter()
        embeddings = _encode_pair(
            plane=plane,
            dataset=dataset,
            sections=sections,
            repo_root=repo_root,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )
        elapsed = time.perf_counter() - t0
        if not embeddings:
            logger.warning(
                "%s/%s: produced 0 embeddings (every section failed); skip save.",
                plane, dataset,
            )
            continue
        out_path = save_query_pair(
            output_root, plane=plane, dataset=dataset, embeddings=embeddings,
        )
        saved[(plane, dataset)] = out_path
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "saved %s/%s: %d images, %.1f MB, %.1fs",
            plane, dataset, len(embeddings), size_mb, elapsed,
        )
    return saved


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    only_pairs = _parse_only_pairs(args.only_pairs)
    pairs = _group_rows_by_pair(
        manifest_root=args.manifest_root.resolve(),
        repo_root=repo_root,
        split=args.split,
        only_pairs=only_pairs,
        max_rows=args.max_rows,
    )
    total_rows = sum(len(p[2]) for p in pairs)
    logger.info(
        "discovered %d (plane, dataset) pair(s), %d total row(s) (split=%s).",
        len(pairs), total_rows, args.split,
    )
    if not pairs:
        logger.error("no rows to encode under the given filters.")
        return 1
    saved = precompute_query_embeddings(
        pairs,
        repo_root=repo_root,
        output_root=args.output_root.resolve(),
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        skip_existing=args.skip_existing,
        prefer_vanilla=args.prefer_vanilla,
    )
    logger.info(
        "done — %d new pair file(s) cached at %s.",
        len(saved), args.output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
