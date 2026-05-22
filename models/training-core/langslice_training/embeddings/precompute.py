"""Precompute SigLIP embeddings for every on-disk atlas reference JPG.

Walks the staged corpus's ``data/atlas/<atlas>/<plane>/*.jpg`` directory tree
and runs each JPG through the same Gemma 4 ``processor.apply_chat_template``
call that the trainer uses, then saves the SigLIP ``last_hidden_state`` keyed
by basename. Path-based keying means non-grid-aligned positions (0.62, 0.65,
0.68 mm) get individual cache entries instead of fighting for the same
``qkey`` bucket.

Invocation::

    python -m langslice_training.embeddings.precompute \\
        --corpus-root /workspace/LangSlice/models/langslice-gemma-4/data \\
        --model unsloth/gemma-4-E4B-it \\
        --output-dir out/atlas_embeddings

The model is loaded the same way :mod:`sft.train_sft._train` loads it
(Unsloth's ``FastVisionModel.from_pretrained`` if available, vanilla HF
``Gemma4ForConditionalGeneration`` otherwise) so the embeddings are bit-exact
matched to what the trainer produces live.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from langslice_training.rl.common.atlas_grid import (
    Plane,
    canonicalize_atlas_name,
)

logger = logging.getLogger(__name__)


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
        description="Precompute SigLIP atlas embeddings keyed by on-disk JPG basename.",
    )
    p.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="Path to the corpus directory containing ``atlas/<atlas>/<plane>/*.jpg``.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-4-E4B-it",
        help="HuggingFace model id or local path to load the vision tower from.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write <atlas>_<plane>.pt cache files.",
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
        "--only-pairs",
        type=str,
        default=None,
        help="Comma-separated 'atlas:plane' filter (default: walk every pair "
             "found under <corpus-root>/atlas).",
    )
    p.add_argument(
        "--max-files-per-pair",
        type=int,
        default=None,
        help="Sanity limit: encode at most N files per pair (debug only).",
    )
    p.add_argument(
        "--prefer-vanilla",
        action="store_true",
        help="Skip Unsloth's FastVisionModel and load via vanilla HF (debug).",
    )
    return p.parse_args(argv)


def _load_model_and_processor(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    prefer_vanilla: bool,
) -> tuple[Any, Any]:
    """Mirror sft.train_sft's loading path so cached embeddings match training."""
    if not prefer_vanilla:
        try:
            from unsloth import FastVisionModel  # type: ignore[import-not-found]
        except ImportError:
            logger.info("unsloth unavailable; falling back to vanilla HF loader")
        else:
            model, processor = FastVisionModel.from_pretrained(
                model_id,
                load_in_4bit=False,  # need full-precision SigLIP for cached emb
                max_seq_length=8192,
            )
            FastVisionModel.for_inference(model)
            model.to(device=device, dtype=dtype)
            return model, processor

    from transformers import AutoProcessor

    try:
        from transformers import Gemma4ForConditionalGeneration  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "transformers does not expose Gemma4ForConditionalGeneration; "
            "upgrade transformers or install unsloth"
        ) from e
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _encode_pil(model: Any, processor: Any, pil_image, *, device: str, dtype: torch.dtype):
    """Run the vision tower on one PIL image and return the SigLIP last_hidden_state.

    Uses the same ``processor.apply_chat_template`` path the trainer hits so
    the cached output matches what ``Gemma4Model.get_image_features`` would
    emit at training time (same patch count, same pan-and-scan behavior).
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
    # The vision tower emits ``last_hidden_state`` as
    # ``(N_images * num_patches, hidden)``. With one image in the message,
    # that's ``(num_patches, hidden)`` already — perfect per-image cache shape.
    # If the model returns a 3D ``(1, num_patches, hidden)`` (older transformers
    # / different vision tower variant), squeeze the leading 1 so the cache
    # shape stays consistent.
    if feats.dim() > 2 and feats.shape[0] == 1:
        feats = feats.squeeze(0)
    return feats.detach().to("cpu")


def _enumerate_corpus_pairs(
    atlas_dir: Path,
    *,
    only_pairs: list[tuple[str, Plane]] | None = None,
) -> list[tuple[str, Plane, Path, list[Path]]]:
    """Walk ``atlas_dir`` and return ``(atlas, plane, plane_dir, jpg_paths)`` tuples.

    Plane dirs that contain no JPGs are skipped silently.
    """
    out: list[tuple[str, Plane, Path, list[Path]]] = []
    only_set = set(only_pairs) if only_pairs is not None else None
    for atlas_subdir in sorted(atlas_dir.iterdir()):
        if not atlas_subdir.is_dir():
            continue
        atlas = canonicalize_atlas_name(atlas_subdir.name)
        for plane_subdir in sorted(atlas_subdir.iterdir()):
            if not plane_subdir.is_dir():
                continue
            plane = plane_subdir.name
            if plane not in ("coronal", "sagittal", "horizontal"):
                continue
            if only_set is not None and (atlas, plane) not in only_set:
                continue
            jpgs = sorted(
                p for p in plane_subdir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".png")
            )
            if not jpgs:
                continue
            out.append((atlas, plane, plane_subdir, jpgs))  # type: ignore[arg-type]
    return out


def _parse_only_pairs(raw: str | None) -> list[tuple[str, Plane]] | None:
    if raw is None:
        return None
    pairs: list[tuple[str, Plane]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--only-pairs entry must be 'atlas:plane', got {chunk!r}")
        atlas, plane = chunk.split(":", 1)
        atlas = canonicalize_atlas_name(atlas.strip())
        plane = plane.strip()
        if plane not in ("coronal", "sagittal", "horizontal"):
            raise ValueError(
                f"--only-pairs plane must be coronal/sagittal/horizontal, got {plane!r}"
            )
        pairs.append((atlas, plane))  # type: ignore[arg-type]
    return pairs or None


def _encode_pair(
    *,
    atlas: str,
    plane: Plane,
    jpg_paths: list[Path],
    model: Any,
    processor: Any,
    device: str,
    dtype: torch.dtype,
    max_files: int | None,
) -> dict[str, torch.Tensor]:
    if max_files is not None:
        jpg_paths = jpg_paths[:max_files]
    out: dict[str, torch.Tensor] = {}
    from PIL import Image
    for i, p in enumerate(jpg_paths):
        pil = Image.open(p).convert("RGB")
        emb = _encode_pil(model, processor, pil, device=device, dtype=dtype)
        out[p.name] = emb
        if (i + 1) % 50 == 0 or (i + 1) == len(jpg_paths):
            logger.info(
                "  %s/%s: %d/%d (%s)",
                atlas, plane, i + 1, len(jpg_paths), p.name,
            )
    return out


def _save_pair(
    output_dir: Path,
    *,
    atlas: str,
    plane: Plane,
    embeddings: dict[str, torch.Tensor],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{atlas}_{plane}.pt"
    payload = {
        "atlas": atlas,
        "plane": plane,
        "embeddings": embeddings,
    }
    # Atomic write: save to .partial then rename. Avoids leaving a half-written
    # file behind if the precompute is interrupted mid-run.
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(out_path)
    return out_path


def precompute_embeddings(
    pairs: Iterable[tuple[str, Plane, Path, list[Path]]],
    *,
    model_id: str,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    max_files: int | None = None,
    prefer_vanilla: bool = False,
) -> dict[tuple[str, Plane], Path]:
    """Driver: encode every on-disk JPG, save per-pair cache files."""
    pairs = list(pairs)
    model, processor = _load_model_and_processor(
        model_id, device=device, dtype=dtype, prefer_vanilla=prefer_vanilla,
    )
    saved: dict[tuple[str, Plane], Path] = {}
    for atlas, plane, _plane_dir, jpgs in pairs:
        t0 = time.perf_counter()
        embeddings = _encode_pair(
            atlas=atlas,
            plane=plane,
            jpg_paths=jpgs,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            max_files=max_files,
        )
        elapsed = time.perf_counter() - t0
        out_path = _save_pair(
            output_dir,
            atlas=atlas,
            plane=plane,
            embeddings=embeddings,
        )
        saved[(atlas, plane)] = out_path
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "saved %s/%s: %d files, %.1f MB, %.1fs",
            atlas, plane, len(embeddings), size_mb, elapsed,
        )
    return saved


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    atlas_dir = args.corpus_root / "atlas"
    if not atlas_dir.is_dir():
        raise FileNotFoundError(
            f"--corpus-root must contain an `atlas/` subdir; got {args.corpus_root}"
        )
    only_pairs = _parse_only_pairs(args.only_pairs)
    pairs = _enumerate_corpus_pairs(atlas_dir, only_pairs=only_pairs)
    if not pairs:
        raise RuntimeError(f"no atlas JPGs found under {atlas_dir}")
    total_files = sum(len(p[3]) for p in pairs)
    logger.info(
        "discovered %d (atlas, plane) pairs, %d JPGs total under %s",
        len(pairs), total_files, atlas_dir,
    )
    saved = precompute_embeddings(
        pairs,
        model_id=args.model,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        max_files=args.max_files_per_pair,
        prefer_vanilla=args.prefer_vanilla,
    )
    logger.info("done — %d pairs cached at %s", len(saved), args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
