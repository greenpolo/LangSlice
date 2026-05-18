"""Driver script: CLI entry point for SFT training of Gemma 4 E4B via Unsloth + TRL.

Usage:
    cd models/langslice-gemma-4/training && python -m sft.train_sft \
        --config configs/sft_default.toml \
        --dataset ../../../models/langslice-gemma-4/data/sft_examples.jsonl \
        --output-dir ../../../out/sft/run0

Heavy deps (unsloth, trl) are imported lazily inside main() so unit tests can
import sibling modules without a runtime install.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import tomllib

from .collate import LangSliceCollator
from .dataset import load_examples, split_subject_aware
from .render import AtlasMetaCache, render_example

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT for Gemma 4 E4B on langslice-native trace JSONL."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="TOML config with [sft], [lora], [data] tables.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="JSONL of langslice-native trace examples.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save the LoRA adapter + tokenizer/processor.")
    p.add_argument("--test-images-root", type=Path,
                   default=Path("references/TestImages"),
                   help="Root containing M01-M09 ground-truth-labeled test images.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config's seed.")
    p.add_argument("--report-to", type=str, default=None,
                   help="Override SFTConfig report_to for smoke/debug runs. Use 'none' to disable.")
    p.add_argument("--skip-agent-eval", action="store_true",
                   help="Skip BaselineEvalCallback + AgentLoopEvalCallback (runtime smoke).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load everything but do not train (smoke for wiring).")
    p.add_argument(
        "--curriculum-weights", type=Path, default=None,
        help=(
            "Optional path to JSON mapping subject_id|section_id -> sampling weight. "
            "When set and the file exists, the train dataset exposes a _weights list "
            "and CurriculumSFTTrainer is used (WeightedRandomSampler). When unset or "
            "missing, training is byte-identical to the non-curriculum path."
        ),
    )
    p.add_argument(
        "--atlas-embedding-cache", type=Path, default=None,
        help=(
            "Optional path to a precomputed atlas-embedding cache directory "
            "(produced by `python -m embeddings.precompute`). When set, the "
            "collator emits per-image cache hits as a sidecar and a "
            "forward-pre-hook substitutes the cached SigLIP outputs in place, "
            "skipping the vision tower for cached atlas images. "
            "When unset, behavior is byte-identical to the no-splice path."
        ),
    )
    p.add_argument(
        "--query-embedding-cache", type=Path, default=None,
        help=(
            "Optional path to a precomputed query (slice) embedding cache "
            "directory (produced by `python -m embeddings.precompute_query`). "
            "Sibling of --atlas-embedding-cache: when set, slice images whose "
            "manifest-relative path is in the cache are spliced from disk in "
            "the same forward-pre-hook, skipping SigLIP for the per-row slice "
            "image. Combine with --atlas-embedding-cache for maximum "
            "SigLIP-skip coverage; atlas takes precedence on path collision. "
            "When unset, behavior is byte-identical to the no-splice path."
        ),
    )
    p.add_argument(
        "--initial-adapter", type=Path, default=None,
        help=(
            "Path to an existing PEFT/LoRA adapter directory to RESUME training from "
            "instead of attaching a fresh randomly-initialized LoRA on the base. Used "
            "by expert iteration so each round nudges the existing SFT adapter rather "
            "than re-learning the task from scratch on a small per-round corpus. When "
            "set, the [lora] table in --config is ignored — the adapter's own r/alpha/"
            "target_modules from its adapter_config.json are used."
        ),
    )
    p.add_argument(
        "--resume-from-checkpoint", type=Path, default=None,
        help=(
            "Path to a HF Trainer checkpoint directory (containing trainer_state.json, "
            "optimizer.pt, scheduler.pt, rng_state.pth) to RESUME a prior interrupted "
            "training run from. Restores optimizer/LR-scheduler/RNG state and continues "
            "at the saved global_step. Distinct from --initial-adapter: that starts a "
            "fresh trainer with a pre-initialized adapter; this continues an existing "
            "trainer mid-run. Useful when an SFT run was killed and you want to finish "
            "the remaining steps without re-doing the completed ones."
        ),
    )
    p.add_argument(
        "--bucketed-shape-sampler", action="store_true",
        help=(
            "Sample shape-pure batches: every micro-batch contains either all "
            "answer-only rows (trace length == 1) or all multi-turn rows (trace "
            "length > 1), never a mix. Required when training on the synthetic "
            "answer-only/multi-turn mix from iSFT's synth_corpus, where the per-row "
            "image count differs by ~5x between buckets and mixed batches produce "
            "high-variance loss. When set, bypasses --curriculum-weights "
            "(curriculum operates at section-selection time upstream of training)."
        ),
    )
    p.add_argument(
        "--clahe-augment-fraction", type=float, default=0.0,
        help=(
            "Fraction in [0.0, 1.0] of training rows that get CLAHE applied to "
            "their slice image at __getitem__ time (independent coin per call). "
            "Mirrors production preprocessing (langslice_harness.image_prep."
            "adaptive_preprocess) to close the train/inference preprocessing "
            "gap. CLAHE rows bypass the query embedding cache for their slice "
            "image so the live SigLIP forward sees the CLAHE'd pixels; atlas "
            "cache lookups are unaffected. Default 0.0 (byte-identical to "
            "current behavior)."
        ),
    )
    p.add_argument(
        "--skip-image-validation",
        action="store_true",
        help=(
            "Skip the per-row fs.stat() check on every query_image_path and "
            "trace[i].tool_result.image_paths entry. The corpus is assumed to "
            "have been pre-validated upstream (e.g. by "
            "models/langslice-gemma-4/training/tools/snap_corpus_atlas_paths.py, "
            "which snaps off-grid atlas positions and drops rows whose "
            "query/atlas files don't resolve). "
            "Saves ~30 min of NTFS-via-Docker p9 RPC overhead at 65k+ rows on "
            "Windows hosts; cheap on Linux native fs. Schema validation always "
            "runs regardless."
        ),
    )
    args = p.parse_args(argv)
    if not 0.0 <= args.clahe_augment_fraction <= 1.0:
        p.error(
            f"--clahe-augment-fraction must be in [0.0, 1.0]; got "
            f"{args.clahe_augment_fraction}"
        )
    return args


def _load_config(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve_test_images_root(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_relative = _REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path


_PER_IMAGE_SOFT_TOKENS = 280  # Gemma 4 image_seq_length from processor_config.json
_TEXT_BUDGET_TOKENS = 2000    # rough headroom for system prompt + tool args + response text


def _estimate_token_count(ex: Any) -> int:
    n_imgs = len(ex.query_image_paths)
    for step in ex.trace:
        tr = step.get("tool_result") or {}
        n_imgs += len(tr.get("image_paths", []))
    return n_imgs * _PER_IMAGE_SOFT_TOKENS + _TEXT_BUDGET_TOKENS


def _filter_overlong(examples: list, max_seq_length: int) -> list:
    kept, dropped = [], 0
    for ex in examples:
        if _estimate_token_count(ex) <= max_seq_length:
            kept.append(ex)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "filtered %d/%d examples whose estimated token count exceeds "
            "max_seq_length=%d (per-image soft tokens=%d, text budget=%d)",
            dropped, len(examples), max_seq_length,
            _PER_IMAGE_SOFT_TOKENS, _TEXT_BUDGET_TOKENS,
        )
    return kept


def _build_datasets(
    args: argparse.Namespace, data_cfg: dict[str, Any], max_seq_length: int,
) -> tuple[list, list]:
    examples = load_examples(
        args.dataset,
        validate_image_paths=not getattr(args, "skip_image_validation", False),
    )
    examples = _filter_overlong(examples, max_seq_length)
    if args.dry_run and len({e.subject_id for e in examples}) < 2:
        logger.warning("--dry-run with <2 subjects: skipping subject-aware split")
        return examples, []
    train, eval_ = split_subject_aware(
        examples,
        holdout_fraction=float(data_cfg["holdout_fraction"]),
        seed=args.seed if args.seed is not None else 0,
    )
    logger.info(
        "Loaded %d examples (%d train, %d eval) from %s",
        len(examples), len(train), len(eval_), args.dataset,
    )
    return train, eval_


class _RenderedDataset:
    """torch.utils.data.Dataset shim: lazily renders examples on __getitem__.

    Curriculum: when ``set_weights`` has been called, the dataset exposes
    a ``_weights`` list (one float per example). The
    :class:`curriculum.sampler.CurriculumSFTTrainer` picks this up and
    swaps in ``WeightedRandomSampler``. Without ``set_weights``, the
    attribute is absent and the trainer falls back to the default
    RandomSampler (byte-identical to the pre-curriculum path).

    CLAHE augmentation: when ``clahe_augment_fraction > 0``, each
    ``__getitem__`` call independently rolls a coin (seeded RNG) to decide
    whether that row's slice image is rendered with CLAHE applied. The roll
    is per-call (dropout-style), not locked to a section_id or epoch.
    """

    def __init__(
        self,
        examples: list,
        atlas_meta_cache: AtlasMetaCache,
        *,
        clahe_augment_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.examples = examples
        self.cache = atlas_meta_cache
        self.clahe_augment_fraction = float(clahe_augment_fraction)
        # Distinct RNG instance per dataset — train and eval splits get
        # different streams even with identical seeds, but each split's
        # stream is fully reproducible from the trainer's --seed.
        self._clahe_rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        # Short-circuit when augmentation is disabled — the RNG is never
        # consumed, preserving byte-identical behavior vs the pre-flag path.
        if self.clahe_augment_fraction <= 0.0:
            apply_clahe = False
        else:
            apply_clahe = self._clahe_rng.random() < self.clahe_augment_fraction
        return {
            "rendered": render_example(
                self.examples[idx],
                atlas_meta_cache=self.cache,
                apply_clahe=apply_clahe,
            )
        }

    def set_weights(self, weights: list[float]) -> None:
        """Install per-example sampling weights (curriculum opt-in)."""
        if len(weights) != len(self.examples):
            raise ValueError(
                f"set_weights: expected {len(self.examples)} values, got {len(weights)}"
            )
        coerced = [float(w) for w in weights]
        if any(w < 0.0 for w in coerced):
            raise ValueError("set_weights: weights must be non-negative")
        if not any(w > 0.0 for w in coerced):
            raise ValueError("set_weights: at least one weight must be > 0")
        self._weights = coerced


class ShapeBucketBatchSampler:
    """Batch sampler that yields shape-pure index batches.

    Splits a dataset's examples into two buckets keyed by trace length —
    answer-only (``len(trace) <= 1``) versus multi-turn
    (``len(trace) > 1``) — shuffles indices within each bucket per epoch,
    cuts each bucket into fixed-size batches, then shuffles the order
    of batches across buckets. Every yielded batch is shape-pure, so
    the per-batch image count is uniform within a micro-batch.

    Incomplete tail batches (last ``len(bucket) % batch_size`` indices)
    are dropped to keep batch size uniform. With a balanced mix this is
    a few rows per epoch; with very small buckets it can drop the whole
    smaller bucket — but in the synth_corpus workflow both buckets are
    sized in the hundreds-to-thousands range.
    """

    def __init__(self, examples: list, batch_size: int, seed: int = 0) -> None:
        self._batch_size = max(int(batch_size), 1)
        self._seed = int(seed)
        single: list[int] = []
        multi: list[int] = []
        for i, ex in enumerate(examples):
            (single if len(ex.trace) <= 1 else multi).append(i)
        self._buckets = [b for b in (single, multi) if b]
        logger.info(
            "ShapeBucketBatchSampler: %d answer-only + %d multi-turn rows; "
            "batch_size=%d → %d batches/epoch",
            len(single), len(multi), self._batch_size, len(self),
        )

    def __iter__(self):
        import random as _random  # noqa: PLC0415
        rng = _random.Random(self._seed)
        batches: list[list[int]] = []
        for bucket in self._buckets:
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            n_full = len(shuffled) // self._batch_size
            for k in range(n_full):
                batches.append(shuffled[k * self._batch_size : (k + 1) * self._batch_size])
        rng.shuffle(batches)
        self._seed += 1
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return sum(len(b) // self._batch_size for b in self._buckets)


def _bucketed_sft_trainer_cls() -> type:
    """Build a ``SFTTrainer`` subclass that yields shape-pure batches.

    Lazy class construction mirrors ``curriculum.sampler._curriculum_sft_trainer_cls``
    so importing ``train_sft`` from a unit-test container (no TRL installed)
    does not fail at import time.
    """
    from trl import SFTTrainer  # noqa: PLC0415

    class BucketedSFTTrainer(SFTTrainer):  # type: ignore[misc, valid-type]
        """``SFTTrainer`` that wraps the train DataLoader in a shape-pure
        batch sampler. Ignores per-row weights — the iSFT pipeline applies
        curriculum at section-selection time upstream of training."""

        def _get_train_dataloader(self):  # type: ignore[override]
            import torch  # noqa: PLC0415

            dataset = self.train_dataset
            args = self.args
            batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
            seed = int(getattr(args, "seed", 0) or 0)
            batch_sampler = ShapeBucketBatchSampler(
                examples=dataset.examples, batch_size=batch_size, seed=seed,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=getattr(self, "data_collator", None),
                num_workers=int(getattr(args, "dataloader_num_workers", 0) or 0),
            )

    return BucketedSFTTrainer


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    args.test_images_root = _resolve_test_images_root(args.test_images_root)
    config = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["sft"].get("seed", 0))

    max_seq_length = int(config["sft"].get("max_seq_length", 16384))
    train_examples, eval_examples = _build_datasets(args, config["data"], max_seq_length)

    cache = AtlasMetaCache()
    clahe_fraction = float(getattr(args, "clahe_augment_fraction", 0.0))
    # Eval split uses fraction=0 so per-step eval loss is computed on the same
    # un-augmented inputs across runs — the augmentation is a train-only knob.
    train_ds = _RenderedDataset(
        train_examples, cache,
        clahe_augment_fraction=clahe_fraction,
        seed=seed,
    )
    eval_ds = _RenderedDataset(eval_examples, cache)
    if clahe_fraction > 0.0:
        logger.info(
            "CLAHE augmentation enabled: fraction=%.3f (train split only); "
            "CLAHE'd rows bypass the query embedding cache so the live SigLIP "
            "forward sees the augmented pixels.",
            clahe_fraction,
        )

    # Optional curriculum weights — keyed by section_id (preferred, from RLVR
    # allocation rows) or subject_id (fallback for legacy distilled rows that
    # pre-date Phase 6 and have no section_id field).  Examples whose key
    # isn't in the map keep weight 1.0.
    if args.curriculum_weights is not None:
        if args.curriculum_weights.is_file():
            from adaptive.curriculum.weights import read_weights_json  # noqa: PLC0415

            weights_map = read_weights_json(args.curriculum_weights)

            def _lookup_weight(ex) -> float:
                # Prefer section_id when available (Phase 6+); fall back to
                # subject_id for pre-Phase-6 distilled rows.
                key = ex.section_id if ex.section_id is not None else ex.subject_id
                return float(weights_map.get(key, 1.0))

            new_weights = [_lookup_weight(ex) for ex in train_examples]
            train_ds.set_weights(new_weights)
            n_matched = sum(
                1 for ex in train_examples
                if (ex.section_id if ex.section_id is not None else ex.subject_id) in weights_map
            )
            logger.info(
                "Loaded %d curriculum weights from %s (matched %d/%d train examples)",
                len(weights_map), args.curriculum_weights, n_matched, len(train_examples),
            )
        else:
            logger.warning(
                "--curriculum-weights=%s does not exist; training with uniform weights.",
                args.curriculum_weights,
            )

    if args.dry_run:
        logger.info("--dry-run: skipping model load + training")
        return

    _train(args, config, train_ds, eval_ds, cache, seed)


def _train(args, config, train_ds, eval_ds, cache, seed: int) -> None:
    """Heavy-import path. Loads model, builds trainer, runs trainer.train()."""
    sft_cfg = dict(config["sft"])
    if args.report_to is not None:
        sft_cfg["report_to"] = "none" if args.report_to.lower() == "none" else args.report_to
    # HF TrainingArguments rejects persistent_workers=True when num_workers=0.
    if int(sft_cfg.get("dataloader_num_workers", 0)) <= 0:
        sft_cfg["dataloader_persistent_workers"] = False
    lora_cfg = dict(config["lora"])
    max_seq_length = int(sft_cfg.get("max_seq_length", 16384))

    # Lazy imports — keep dataset/render/collate unit tests cheap
    # VRAM micro-optimizations — must be set BEFORE Unsloth/torch initialize
    # the CUDA backend.
    #   - UNSLOTH_CE_LOSS_TARGET_GB caps per-chunk CE-loss memory; combined
    #     with UNSLOTH_CE_LOSS_N_CHUNKS already set in docker-compose this
    #     forces tighter chunking on Gemma 4's 262K-token vocab.
    #   - SDPA backend pinning: Gemma 4's head_dim=512 makes FA2 unavailable,
    #     so we must keep PyTorch on the mem-efficient/cuDNN backends and
    #     disable the math fallback (quadratic memory).
    import os as _os  # noqa: PLC0415

    _os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "1")

    from unsloth import FastVisionModel  # noqa: I001 - Unsloth must patch before TRL import
    import torch as _torch_sdpa  # noqa: PLC0415

    _torch_sdpa.backends.cuda.enable_mem_efficient_sdp(True)
    _torch_sdpa.backends.cuda.enable_cudnn_sdp(True)
    # 2026-05-14: math_sdp enabled as fallback. Live SigLIP on slice/atlas
    # images produces variable output_patch_count (pan-and-scan), so
    # nosplice training hits attention shapes cuDNN+mem_efficient both
    # reject ("Invalid backend"). Math is the floor — quadratic memory on
    # bad batches only, cheap on normal batches.
    _torch_sdpa.backends.cuda.enable_math_sdp(True)
    _torch_sdpa.backends.cuda.enable_flash_sdp(True)  # harmless; SDPA dispatcher just skips it for head_dim=512
    # Multi-turn rows with 3-8 images vary seq_len 1500-6144 — _dynamo's default
    # cache-size limit of 8 falls back to slow eager once exceeded. 64 leaves
    # plenty of room for every shape we hit; cached graphs are RAM-cheap.
    import torch._dynamo as _torch_dynamo  # noqa: PLC0415
    _torch_dynamo.config.recompile_limit = 64
    from trl import SFTConfig, SFTTrainer  # noqa: I001
    from rlvr.atlas_grid import build_atlas_grid

    from .eval import AgentLoopEvalCallback, BaselineEvalCallback

    # Pre-render the atlas grid once for the eval callbacks
    pairs = {(ex.atlas_name, ex.plane) for ex in train_ds.examples + eval_ds.examples}
    atlas_grid = build_atlas_grid(pairs)

    # Load base model + processor
    model, processor = FastVisionModel.from_pretrained(
        sft_cfg["base_model"],
        load_in_4bit=bool(sft_cfg.get("load_in_4bit", True)),
        max_seq_length=max_seq_length,
    )
    initial_adapter = getattr(args, "initial_adapter", None)
    if initial_adapter is not None:
        # Resume training of an existing adapter (expert-iteration mode).
        # Skips the random-init LoRA path so the model starts from the
        # converged SFT weights rather than essentially-random LoRA deltas.
        from peft import PeftModel  # noqa: PLC0415

        adapter_path = initial_adapter
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"--initial-adapter path does not exist: {adapter_path}"
            )
        logger.info("Resuming trainable LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=True,
        )
    else:
        # Fresh LoRA on the base — original SFT-from-scratch path.
        model = FastVisionModel.get_peft_model(
            model,
            max_seq_length=max_seq_length,
            finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
            finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
            finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
            finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            use_rslora=bool(lora_cfg.get("use_rslora", False)),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
            random_state=seed,
        )
    FastVisionModel.for_training(model)

    atlas_cache_path = getattr(args, "atlas_embedding_cache", None)
    atlas_cache = None
    if atlas_cache_path is not None:
        from embeddings.cache import AtlasEmbeddingCache

        atlas_cache = AtlasEmbeddingCache(atlas_cache_path)
        if not atlas_cache.pairs():
            raise FileNotFoundError(
                f"--atlas-embedding-cache points at {atlas_cache_path}, but no "
                "<atlas>_<plane>.pt cache files were found there. Run "
                "`python -m embeddings.precompute` first."
            )
        logger.info(
            "atlas-embedding splice enabled: %d pair files loaded from %s",
            len(atlas_cache.pairs()), atlas_cache_path,
        )

    query_cache_path = getattr(args, "query_embedding_cache", None)
    query_cache = None
    if query_cache_path is not None:
        from embeddings.query_cache import QueryEmbeddingCache

        query_cache = QueryEmbeddingCache(query_cache_path)
        if not query_cache.pairs():
            raise FileNotFoundError(
                f"--query-embedding-cache points at {query_cache_path}, but no "
                "<plane>__<dataset>.pt cache files were found there. Run "
                "`python -m embeddings.precompute_query` first."
            )
        logger.info(
            "query-embedding splice enabled: %d pair files loaded from %s",
            len(query_cache.pairs()), query_cache_path,
        )

    # Install the splice hook when EITHER cache is provided — the hook is
    # cache-agnostic; it just consumes the sidecars the collator emits. The
    # function name still mentions "atlas" for historical reasons but the
    # mechanism applies equally to atlas+query embeddings.
    enable_splice = atlas_cache is not None or query_cache is not None
    if enable_splice:
        from embeddings.splice import install_atlas_splice

        # When vision layers are frozen, wrap the splice's vision-tower and
        # projector forward calls in torch.no_grad() — autograd tape over a
        # frozen pipeline is pure activation-memory waste (~0.5-1.5 GB at
        # PDBS=4). Flips off automatically the moment the LoRA config opts
        # vision back into training.
        frozen_vision = not bool(lora_cfg.get("finetune_vision_layers", False))
        install_atlas_splice(model, frozen_vision=frozen_vision)

    collator = LangSliceCollator(
        processor=processor,
        max_seq_length=max_seq_length,
        atlas_cache=atlas_cache,
        query_cache=query_cache,
        enable_splice=enable_splice,
    )

    sft_config_kwargs = {
        k: v for k, v in sft_cfg.items()
        if k not in (
            "base_model",
            "load_in_4bit",
            "max_seq_length",
            "agent_eval_steps",
            "seed",
            "chat_template_kwargs",
            "dataset_kwargs",
            "remove_unused_columns",
        )
    }
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        seed=seed,
        # Critical: do not let TRL try to apply text-only assistant-only loss
        assistant_only_loss=False,
        # Critical for VLMs: no TRL truncation. The custom collator rejects
        # examples beyond max_seq_length before they reach the trainer.
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        **sft_config_kwargs,
    )

    if getattr(args, "skip_agent_eval", False):
        callbacks: list = []
    else:
        callbacks = [
            BaselineEvalCallback(
                processor=processor,
                atlas_grid=atlas_grid,
                test_images_root=args.test_images_root,
            ),
            AgentLoopEvalCallback(
                processor=processor,
                atlas_grid=atlas_grid,
                test_images_root=args.test_images_root,
                agent_eval_steps=int(sft_cfg.get("agent_eval_steps", 200)),
            ),
        ]

    # Trainer-class selection:
    # 1. --bucketed-shape-sampler wins: synthetic answer-only + multi-turn rows
    #    have ~5x different per-row image counts, so mixed batches would blow
    #    up loss variance. The iSFT pipeline applies curriculum upstream at
    #    section-selection time, so we drop the per-row WeightedRandomSampler
    #    in this mode even if --curriculum-weights was also passed.
    # 2. Else, curriculum-aware path when the dataset has _weights wired.
    # 3. Else, vanilla SFTTrainer (byte-identical to the pre-curriculum path).
    use_bucketed = bool(getattr(args, "bucketed_shape_sampler", False))
    use_curriculum = hasattr(train_ds, "_weights")
    if use_bucketed:
        if use_curriculum:
            logger.warning(
                "--bucketed-shape-sampler set: ignoring --curriculum-weights "
                "(curriculum is applied at section-selection time upstream)."
            )
        trainer_cls = _bucketed_sft_trainer_cls()
    elif use_curriculum:
        from curriculum.sampler import CurriculumSFTTrainer  # noqa: PLC0415

        trainer_cls = CurriculumSFTTrainer
    else:
        trainer_cls = SFTTrainer
    trainer = trainer_cls(
        model=model,                  # already a PeftModel — do NOT pass peft_config
        processing_class=processor,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    # Unsloth can replace the data collator while wrapping SFTTrainer. The
    # LangSlice collator owns VLM chat-template rendering and assistant masking,
    # so force it back before the first dataloader fetch.
    trainer.data_collator = collator
    resume_path = getattr(args, "resume_from_checkpoint", None)
    if resume_path is not None:
        logger.info("Resuming training from checkpoint: %s", resume_path)
        trainer.train(resume_from_checkpoint=str(resume_path))
    else:
        trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    logger.info("Saved adapter + processor to %s", args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
