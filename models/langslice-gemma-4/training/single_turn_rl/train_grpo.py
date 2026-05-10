"""GRPO driver for single-turn final-answer RL (Lane A).

Sibling of :mod:`rlvr.train_grpo`. The differences are deliberate:

* **No** ``environment_factory`` — single-turn rollouts have no tool calls,
  so the GRPOTrainer is constructed with just ``reward_funcs``.
* **No group code** — single-turn-singles only.
* Reward gets ground truth via TRL's dataset-column kwargs passthrough, not
  via env state.

Optional features (mirroring the rlvr / SFT lanes):

* **Curriculum sampling** via ``--curriculum-weights PATH``. When set + file
  exists, the train dataset becomes a :class:`WeightedRowDataset` and the
  trainer becomes :class:`curriculum.sampler.CurriculumGRPOTrainer` (which
  swaps in :class:`torch.utils.data.WeightedRandomSampler`). A
  :class:`curriculum.log.CurriculumLogger` writes one JSONL row per rollout
  to ``<output_dir>/curriculum_log.jsonl`` for the next round's weight update.
* **Atlas embedding splice** via ``--atlas-embedding-cache PATH``. When set,
  loads the precomputed SigLIP cache, calls
  :func:`embeddings.splice.install_atlas_splice` on the wrapped model, and
  uses :class:`SingleTurnGRPOTrainer` (a GRPOTrainer subclass) which wraps
  the processor to emit per-batch sidecars
  (``precomputed_image_mask`` / ``precomputed_cached_flat`` /
  ``precomputed_cached_patch_counts``) so the splice hook can skip SigLIP
  for cached atlas reference images.

Heavy deps (``unsloth``, ``trl``) are imported lazily inside :func:`main` so
unit tests can import sibling modules without a runtime install.

Usage
-----
::

    # From repo root via the langslice_single_turn_rl shim:
    python -m langslice_single_turn_rl \\
        --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml \\
        --sft-model out/sft/docker-sft-1011-merged-bf16 \\
        --terminal-states out/single_turn_rl/terminal_states.jsonl \\
        --output-dir out/rlvr_single_turn/terminal_smoke \\
        --atlas-embedding-cache out/atlas_embeddings \\
        --curriculum-weights out/single_turn_rl/round_0_weights.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import tomllib

from rlvr.train_grpo import (
    _adapter_base_model_name,
    _filter_grpo_config_for_installed_trl,
    _install_optional_dep_stubs,
)

from .dataset import (
    RowDataset,
    WeightedRowDataset,
    build_datasets,
    specs_to_single_slice_examples,
)
from .rewards import make_terminal_reward

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _wrap_reward_with_curriculum_log(
    reward_fn: Any,
    *,
    log: Any,
    section_bins: dict[str, int],
    round_idx: int,
):
    """Return a reward callable that emits a JSONL row per (rollout, row).

    Mirrors :func:`rlvr.train_grpo._wrap_reward_with_curriculum_log` but for
    the single-turn lane: the reward signature is
    ``func(completions, prompts, **dataset_columns)`` (no environments).
    Computes per-rollout abs-error and accuracy_pct against the row's
    valid_range_mm + ground_truth_mm so the curriculum loop can aggregate
    per-bin MAE between rounds.
    """

    def _logging_reward(
        completions: list[Any] | None = None,
        prompts: list[Any] | None = None,
        ground_truth_mm: list[float] | None = None,
        valid_range_mm: list[tuple[float, float]] | None = None,
        section_id: list[str] | None = None,
        plane: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        scores = reward_fn(
            completions=completions,
            prompts=prompts,
            ground_truth_mm=ground_truth_mm,
            valid_range_mm=valid_range_mm,
            section_id=section_id,
            plane=plane,
            **kwargs,
        )
        # Defensive defaults so logging never breaks training when an upstream
        # column is missing.
        sids = section_id or []
        planes = plane or []
        gts = ground_truth_mm or []
        ranges = valid_range_mm or []
        from .rewards import _ParseError, parse_position_mm  # noqa: PLC0415
        for i, comp in enumerate(completions or []):
            try:
                # Re-extract assistant text the same way rewards.score_completion does.
                from .rewards import _extract_completion_text  # noqa: PLC0415
                text = _extract_completion_text(comp)
                try:
                    pred = parse_position_mm(text)
                except _ParseError:
                    pred = None
                gt = float(gts[i]) if i < len(gts) else float("nan")
                vr = ranges[i] if i < len(ranges) else (0.0, 1.0)
                axis_span = max(float(vr[1]) - float(vr[0]), 1e-9)
                if pred is None:
                    abs_err = axis_span
                elif pred < float(vr[0]) or pred > float(vr[1]):
                    abs_err = max(abs(pred - gt), axis_span)
                else:
                    abs_err = abs(pred - gt)
                accuracy_pct = max(0.0, 100.0 * (1.0 - abs_err / axis_span))
                sid = sids[i] if i < len(sids) else ""
                pl = planes[i] if i < len(planes) else ""
                if not sid:
                    continue
                log.append({
                    "round": int(round_idx),
                    "section_id": str(sid),
                    "bin_idx": int(section_bins.get(str(sid), -1)),
                    "plane": str(pl),
                    "abs_err_mm": round(float(abs_err), 4),
                    "accuracy_pct": round(float(accuracy_pct), 3),
                })
            except (AttributeError, ValueError, TypeError, OSError):  # pragma: no cover
                # Logging must never break training; swallow per-row errors.
                continue
        return scores

    _logging_reward.__name__ = getattr(reward_fn, "__name__", "terminal_reward")
    _logging_reward.__qualname__ = getattr(reward_fn, "__qualname__", "terminal_reward")
    return _logging_reward


def _atlas_pairs_in_dataset(dataset: RowDataset) -> set[tuple[str, str]]:
    """Distinct ``(atlas, plane)`` pairs in the dataset's specs."""
    return {(str(s["atlas_name"]), str(s["plane"])) for s in dataset._specs}  # noqa: SLF001


def _validate_atlas_cache_coverage(cache: Any, dataset: RowDataset) -> None:
    """Warn about (atlas, plane) pairs in the dataset that have no cache file."""
    needed = _atlas_pairs_in_dataset(dataset)
    cached = set(cache.pairs())
    missing = needed - cached
    if missing:
        logger.warning(
            "atlas-embedding cache is missing %d (atlas, plane) pair(s) the "
            "training set uses: %s. Splice will fall back to live SigLIP for "
            "those images. Re-run `python -m embeddings.precompute` to fill them.",
            len(missing), sorted(missing),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run single-turn GRPO on LangSlice training data. Defaults wire the "
            "unified RL pipeline (adaptive curriculum + adaptive reward + "
            "index-driven dataset). Pass --curriculum-mode/--reward-mode/"
            "--data-source to fall back to the legacy static paths."
        ),
    )
    p.add_argument("--config", type=Path, required=True,
                   help="TOML config with [grpo], [lora], [data], [reward], "
                   "and (for adaptive defaults) [adaptive] sections.")
    p.add_argument("--sft-model", type=Path, required=True,
                   help="Path or HF id of the SFT base to start from. May be a "
                   "merged checkpoint OR a PEFT adapter directory; the latter "
                   "is loaded on top of the adapter's recorded base model.")
    p.add_argument("--resume-from-adapter", type=Path, default=None,
                   help="Optional path to a previously-saved RL LoRA adapter "
                   "to resume training.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save adapter + logs.")
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Repo root used to resolve repo-relative image paths "
                   "in the JSONL. Defaults to current working directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-to", type=str, default=None,
                   help="Override GRPOConfig report_to (use 'none' to disable).")

    # ---- Mode selection (unified pipeline) ----
    p.add_argument(
        "--curriculum-mode",
        choices=["adaptive", "static_weights", "none"],
        default="adaptive",
        help="adaptive: AdaRFT band sampler + per-rollout difficulty write-back "
        "(default). static_weights: legacy WeightedRandomSampler driven by a "
        "weights JSON. none: vanilla random sampling.",
    )
    p.add_argument(
        "--reward-mode",
        choices=["adaptive", "static"],
        default="adaptive",
        help="adaptive: bell sigma/cutoff scale to recent achievement quantiles "
        "(default). static: fixed cutoff_frac/sigma_frac from [reward] in TOML.",
    )
    p.add_argument(
        "--data-source",
        choices=["index", "terminal_states"],
        default="index",
        help="index: pull rows from data/manifest via ManifestIndex (default; "
        "needed for the adaptive curriculum's full-pool query). "
        "terminal_states: legacy per-trace JSONL from "
        "single_turn_rl.terminal_states build.",
    )

    # ---- Data-source paths (one is required depending on --data-source) ----
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifest"),
        help="Root of the data manifest (used when --data-source=index). "
        "Defaults to data/manifest.",
    )
    p.add_argument(
        "--terminal-states",
        type=Path,
        default=None,
        help="Required when --data-source=terminal_states. Path to the "
        "terminal-state JSONL produced by single_turn_rl.terminal_states build.",
    )

    # ---- Adaptive seeding + cadence ----
    p.add_argument(
        "--difficulty-seed",
        type=Path,
        default=None,
        help="Optional JSON from tools/seed_difficulty_from_slicebench.py — "
        "bulk-loads per-section seed difficulty into the index at startup.",
    )

    # ---- Curriculum-static-weights (legacy) ----
    p.add_argument(
        "--curriculum-weights",
        type=Path,
        default=None,
        help="Required when --curriculum-mode=static_weights. JSON file "
        "mapping section_id -> sampling weight; train dataset becomes a "
        "WeightedRowDataset sampled with WeightedRandomSampler.",
    )

    # ---- Splice (orthogonal to mode selection) ----
    p.add_argument(
        "--atlas-embedding-cache",
        type=Path,
        default=None,
        help="Optional directory containing precomputed SigLIP cache files "
        "(<atlas>_<plane>.pt). Enables the atlas-embedding splice (skips "
        "SigLIP for cached atlas reference images).",
    )

    # ---- Adaptive-mode knobs (overridable from [adaptive] in TOML) ----
    # These are intentionally optional with default=None so we can tell apart
    # "user passed an explicit value" from "use the TOML default". The merge
    # logic in _resolve_adaptive_cfg honours CLI > TOML > hardcoded default.
    adp = p.add_argument_group(
        "adaptive",
        "Adaptive curriculum + reward knobs (overridable in TOML [adaptive]).",
    )
    adp.add_argument("--adaptive-target-reward", type=float, default=None)
    adp.add_argument("--adaptive-alpha", type=float, default=None)
    adp.add_argument("--adaptive-eta", type=float, default=None)
    adp.add_argument("--adaptive-ema-decay", type=float, default=None)
    adp.add_argument("--adaptive-band-width", type=float, default=None)
    adp.add_argument("--max-per-subject-in-batch", type=int, default=None)
    adp.add_argument("--adaptive-sigma-quantile", type=float, default=None)
    adp.add_argument("--adaptive-cutoff-quantile", type=float, default=None)
    adp.add_argument("--adaptive-min-sigma-frac", type=float, default=None)
    adp.add_argument("--adaptive-max-sigma-frac", type=float, default=None)
    adp.add_argument("--adaptive-min-observations", type=int, default=None)
    adp.add_argument(
        "--persist-difficulty-every-n-ticks",
        type=int,
        default=None,
        help="0 disables the JSON sidecar persist; otherwise persist every N "
        "callback ticks.",
    )

    args = p.parse_args(argv)
    _enforce_arg_mutex(p, args)
    return args


# Hardcoded defaults for the [adaptive] knobs. Match the documented spec; TOML
# [adaptive] section + per-knob CLI flag both override these.
_ADAPTIVE_DEFAULTS: dict[str, Any] = {
    "target_reward": 0.5,
    "alpha": 2.0,
    "eta": 0.05,
    "ema_decay": 0.7,
    "band_width": 0.15,
    "max_per_subject_in_batch": 2,
    "sigma_quantile": 0.5,
    "cutoff_quantile": 0.95,
    "min_sigma_frac": 0.005,
    "max_sigma_frac": 0.05,
    "min_observations": 50,
    "persist_difficulty_every_n_ticks": 50,
}


# Mapping from CLI dest-name to TOML [adaptive] key. Used by both mutex
# enforcement (so the error messages refer to the user-facing flag) and the
# CLI > TOML > default merge.
_ADAPTIVE_CLI_TO_TOML: dict[str, str] = {
    "adaptive_target_reward": "target_reward",
    "adaptive_alpha": "alpha",
    "adaptive_eta": "eta",
    "adaptive_ema_decay": "ema_decay",
    "adaptive_band_width": "band_width",
    "max_per_subject_in_batch": "max_per_subject_in_batch",
    "adaptive_sigma_quantile": "sigma_quantile",
    "adaptive_cutoff_quantile": "cutoff_quantile",
    "adaptive_min_sigma_frac": "min_sigma_frac",
    "adaptive_max_sigma_frac": "max_sigma_frac",
    "adaptive_min_observations": "min_observations",
    "persist_difficulty_every_n_ticks": "persist_difficulty_every_n_ticks",
}


def _enforce_arg_mutex(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> None:
    """Enforce the cross-flag mutual-exclusion + required-companion rules.

    Argparse's ``add_mutually_exclusive_group`` only supports same-group
    exclusion; we want cross-group rules ("--curriculum-mode adaptive" is
    incompatible with "--curriculum-weights"). Doing the check here also
    keeps the ``--help`` output readable.
    """
    if args.curriculum_mode == "adaptive" and args.curriculum_weights is not None:
        parser.error(
            "--curriculum-mode adaptive is incompatible with --curriculum-weights "
            "(weights are only honoured under --curriculum-mode static_weights)."
        )
    if args.curriculum_mode == "static_weights" and args.curriculum_weights is None:
        parser.error(
            "--curriculum-mode static_weights requires --curriculum-weights PATH."
        )
    if args.data_source == "terminal_states" and args.terminal_states is None:
        parser.error(
            "--data-source terminal_states requires --terminal-states PATH."
        )
    if args.data_source == "index" and args.terminal_states is not None:
        # Not strictly an error, but flag it because it's almost certainly a
        # mistake — the index path doesn't read --terminal-states.
        logger.warning(
            "--terminal-states %s is ignored under --data-source=index.",
            args.terminal_states,
        )


def _resolve_adaptive_cfg(
    args: argparse.Namespace, toml_adaptive: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI > TOML > hardcoded defaults for the [adaptive] knobs.

    Returns a dict keyed by the TOML key (one entry per knob in
    :data:`_ADAPTIVE_DEFAULTS`). Coerces numerics so callers can rely on the
    shape regardless of how each value arrived (TOML int, CLI float, etc.).
    """
    out: dict[str, Any] = dict(_ADAPTIVE_DEFAULTS)
    # TOML overrides defaults.
    for k in out:
        if k in toml_adaptive:
            out[k] = toml_adaptive[k]
    # CLI overrides TOML.
    for cli_key, toml_key in _ADAPTIVE_CLI_TO_TOML.items():
        cli_val = getattr(args, cli_key, None)
        if cli_val is not None:
            out[toml_key] = cli_val
    # Coerce numerics defensively.
    int_keys = {
        "max_per_subject_in_batch",
        "min_observations",
        "persist_difficulty_every_n_ticks",
    }
    for k, v in list(out.items()):
        if k in int_keys:
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)

    grpo_cfg = dict(config.get("grpo", {}))
    if args.report_to is not None:
        grpo_cfg["report_to"] = "none" if args.report_to.lower() == "none" else args.report_to
    lora_cfg = dict(config.get("lora", {}))
    data_cfg = dict(config.get("data", {}))
    reward_cfg = dict(config.get("reward", {}))
    adaptive_cfg = _resolve_adaptive_cfg(args, dict(config.get("adaptive", {})))

    # ---- Mode flags (resolved up front so the rest of main reads cleanly) ----
    use_static_weights = args.curriculum_mode == "static_weights"
    use_adaptive_curriculum = args.curriculum_mode == "adaptive"
    use_adaptive_reward = args.reward_mode == "adaptive"
    use_index_data = args.data_source == "index"

    # ---- ManifestIndex (Lane B) ----
    manifest_index: Any = None
    if use_index_data or use_adaptive_curriculum:
        # The adaptive curriculum needs the manifest index for the live
        # difficulty write-back even when the dataset itself is loaded from
        # terminal states; build it eagerly so both code paths can reference
        # the same instance.
        from .manifest_index import ManifestIndex  # noqa: PLC0415

        manifest_index = ManifestIndex.from_manifest_root(
            args.manifest_root, repo_root=args.repo_root,
        )
        logger.info(
            "ManifestIndex loaded from %s with %d sections.",
            args.manifest_root, len(manifest_index),
        )
        if args.difficulty_seed is not None:
            n_seeded = manifest_index.bulk_load_difficulty(args.difficulty_seed)
            logger.info(
                "Bulk-seeded %d difficulty rows from %s.",
                n_seeded, args.difficulty_seed,
            )

    # ---- Dataset assembly ----
    if use_index_data:
        if manifest_index is None:  # pragma: no cover — defensive
            raise RuntimeError(
                "Internal: --data-source=index but ManifestIndex was not built."
            )
        from .dataset import build_datasets_from_index  # noqa: PLC0415

        slate_root = args.repo_root / "models" / "langslice-gemma-4" / "data"
        train_dataset, eval_dataset = build_datasets_from_index(
            manifest_index=manifest_index,
            split=str(data_cfg.get("split", "rlvr")) or None,
            repo_root=args.repo_root,
            slate_root=slate_root,
            n_positions=int(data_cfg.get("slate_n_positions", 9)),
            eval_holdout_every=int(data_cfg.get("eval_holdout_every", 5)),
            max_examples=data_cfg.get("max_examples"),
            seed=args.seed,
        )
        data_source_label = f"manifest_index({args.manifest_root})"
    else:
        train_dataset, eval_dataset = build_datasets(
            args.terminal_states,
            repo_root=args.repo_root,
            eval_holdout_every=int(data_cfg.get("eval_holdout_every", 5)),
            max_examples=data_cfg.get("max_examples"),
            seed=args.seed,
            weighted=use_static_weights,
        )
        data_source_label = str(args.terminal_states)

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Train dataset is empty — check {data_source_label} and "
            f"eval_holdout_every (currently {data_cfg.get('eval_holdout_every', 5)})."
        )
    logger.info(
        "Assembled %d train rows, %d eval rows from %s",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
        data_source_label,
    )

    # ---- Static-weights curriculum (legacy path) ----
    if use_static_weights:
        from curriculum.weights import (  # noqa: PLC0415
            read_weights_json,
            update_weighted_dataset,
        )

        weights_map = read_weights_json(args.curriculum_weights)
        assert isinstance(train_dataset, WeightedRowDataset)
        n_matched = update_weighted_dataset(cast(Any, train_dataset), weights_map)
        logger.info(
            "Loaded %d curriculum weights from %s (matched %d/%d train rows)",
            len(weights_map), args.curriculum_weights, n_matched, len(train_dataset),
        )

    # ---- Reward function ----
    schedule: Any = None
    if use_adaptive_reward:
        from .adaptive_reward import (  # noqa: PLC0415
            AdaptiveRewardSchedule,
            make_adaptive_terminal_reward,
        )

        schedule = AdaptiveRewardSchedule(
            min_sigma_frac=adaptive_cfg["min_sigma_frac"],
            max_sigma_frac=adaptive_cfg["max_sigma_frac"],
            sigma_quantile=adaptive_cfg["sigma_quantile"],
            cutoff_quantile=adaptive_cfg["cutoff_quantile"],
            min_observations=adaptive_cfg["min_observations"],
            static_fallback=(
                float(reward_cfg.get("sigma_frac", 0.05)),
                float(reward_cfg.get("cutoff_frac", 0.15)),
            ),
        )
        reward_fn = make_adaptive_terminal_reward(
            schedule=schedule,
            format_penalty=float(reward_cfg.get("format_penalty", -1.0)),
            out_of_range_reward=float(reward_cfg.get("out_of_range_reward", 0.0)),
        )
        logger.info(
            "Adaptive reward enabled: sigma_frac in [%.4f, %.4f], "
            "sigma q=%.2f, cutoff q=%.2f, warmup=%d.",
            adaptive_cfg["min_sigma_frac"], adaptive_cfg["max_sigma_frac"],
            adaptive_cfg["sigma_quantile"], adaptive_cfg["cutoff_quantile"],
            adaptive_cfg["min_observations"],
        )
    else:
        reward_fn = make_terminal_reward(
            cutoff_frac=float(reward_cfg.get("cutoff_frac", 0.15)),
            sigma_frac=float(reward_cfg.get("sigma_frac", 0.05)),
            format_penalty=float(reward_cfg.get("format_penalty", -1.0)),
            out_of_range_reward=float(reward_cfg.get("out_of_range_reward", 0.0)),
        )

    # ---- Static-weights curriculum logging (legacy) ----
    section_bin_lookup: dict[str, int] = {}
    curriculum_round = int(data_cfg.get("curriculum_round", 0))
    if use_static_weights:
        try:
            from curriculum.bins import compute_section_bins  # noqa: PLC0415

            singles_stub = specs_to_single_slice_examples(
                train_dataset._specs  # noqa: SLF001
            )
            section_bin_lookup = compute_section_bins(singles_stub)
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError, KeyError) as exc:
            logger.warning(
                "Curriculum logging: bin computation skipped (%s); "
                "log rows will record bin_idx=-1.", exc,
            )

        from curriculum.log import CurriculumLogger  # noqa: PLC0415

        log_path = args.output_dir / "curriculum_log.jsonl"
        curriculum_logger = CurriculumLogger(log_path)
        reward_fn = _wrap_reward_with_curriculum_log(
            reward_fn,
            log=curriculum_logger,
            section_bins=section_bin_lookup,
            round_idx=curriculum_round,
        )
        logger.info(
            "Curriculum logging enabled: round=%d -> %s (section_bins=%d)",
            curriculum_round, log_path, len(section_bin_lookup),
        )

    # ---- Optional atlas-embedding splice ----
    atlas_cache: Any = None
    if args.atlas_embedding_cache is not None:
        from embeddings.cache import AtlasEmbeddingCache  # noqa: PLC0415

        atlas_cache = AtlasEmbeddingCache(args.atlas_embedding_cache)
        if not atlas_cache.pairs():
            raise FileNotFoundError(
                f"--atlas-embedding-cache points at {args.atlas_embedding_cache}, "
                "but no <atlas>_<plane>.pt cache files were found there. Run "
                "`python -m embeddings.precompute` first."
            )
        _validate_atlas_cache_coverage(atlas_cache, train_dataset)

    # ---- Adaptive curriculum sampler (built before TRL imports so the
    # smoke-test path can short-circuit just before model load) ----
    sampler: Any = None
    if use_adaptive_curriculum:
        # The sampler is built early so AdaRFTCurriculumCallback can be
        # constructed and wired into the trainer's callbacks list. Heavy
        # imports stay deferred — the sampler module only needs torch,
        # which is already a hard dep of this whole training stack.
        sampler = _build_adaptive_sampler(
            train_dataset,
            grpo_cfg=grpo_cfg,
            adaptive_cfg=adaptive_cfg,
            seed=args.seed,
        )
        logger.info(
            "Adaptive curriculum sampler built: target_reward=%.2f, "
            "alpha=%.2f, eta=%.3f, ema_decay=%.2f, band_width=%.2f, "
            "subject_cap=%d.",
            adaptive_cfg["target_reward"], adaptive_cfg["alpha"],
            adaptive_cfg["eta"], adaptive_cfg["ema_decay"],
            adaptive_cfg["band_width"], adaptive_cfg["max_per_subject_in_batch"],
        )

    # ---- Heavy imports happen here so unit tests can skip them ----
    from unsloth import FastVisionModel  # noqa: PLC0415
    _install_optional_dep_stubs()
    from trl import (  # noqa: PLC0415
        GRPOConfig,  # pyright: ignore[reportPrivateImportUsage]
        GRPOTrainer,  # pyright: ignore[reportPrivateImportUsage]
    )

    max_seq_length = int(grpo_cfg.pop("max_seq_length", 4096))
    load_in_4bit = bool(grpo_cfg.pop("load_in_4bit", True))
    sft_adapter_base = _adapter_base_model_name(args.sft_model)
    model_name_or_path = sft_adapter_base or str(args.sft_model)
    model, processor = FastVisionModel.from_pretrained(
        model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=False,  # Gemma 4 GRPO requires Unsloth-native generation.
    )

    if args.resume_from_adapter is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Resuming trainable LoRA adapter from %s", args.resume_from_adapter)
        model = PeftModel.from_pretrained(
            model, str(args.resume_from_adapter), is_trainable=True,
        )
    elif sft_adapter_base is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Loading SFT LoRA adapter from %s", args.sft_model)
        model = PeftModel.from_pretrained(
            model, str(args.sft_model), is_trainable=True,
        )
    else:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
            finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
            finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
            finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
            random_state=args.seed,
        )

    # Install the splice AFTER PEFT wrap (so the inner Gemma 4 model is the
    # wrapped one). Idempotent — no-op if already installed.
    if atlas_cache is not None:
        from embeddings.splice import install_atlas_splice  # noqa: PLC0415

        install_atlas_splice(model)
        logger.info(
            "atlas-embedding splice installed: %d pair files loaded from %s",
            len(atlas_cache.pairs()), args.atlas_embedding_cache,
        )

    grpo_cfg = _filter_grpo_config_for_installed_trl(GRPOConfig, grpo_cfg)
    training_args = GRPOConfig(output_dir=str(args.output_dir), seed=args.seed, **grpo_cfg)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": processor,
        "train_dataset": train_dataset,
        "reward_funcs": [reward_fn],
        "args": training_args,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    # Pick the trainer class based on the active mixin combination.
    trainer_cls = _select_trainer_cls(
        adarft_enabled=use_adaptive_curriculum,
        static_curriculum_enabled=use_static_weights,
        atlas_cache=atlas_cache,
    )
    if atlas_cache is not None and use_adaptive_curriculum:
        trainer = trainer_cls(
            atlas_cache=atlas_cache,
            curriculum_sampler=sampler,
            **trainer_kwargs,
        )
    elif atlas_cache is not None:
        trainer = trainer_cls(atlas_cache=atlas_cache, **trainer_kwargs)
    elif use_adaptive_curriculum:
        trainer = trainer_cls(curriculum_sampler=sampler, **trainer_kwargs)
    else:
        trainer = trainer_cls(**trainer_kwargs)

    # Wire the AdaRFT callback after construction so we can pass the actual
    # sampler instance the trainer holds. Mirrors the pattern documented in
    # AdaRFTCurriculumCallback's docstring (callback hooks ``on_log``).
    if use_adaptive_curriculum and manifest_index is not None and schedule is not None:
        from .curriculum import AdaRFTCurriculumCallback  # noqa: PLC0415

        # Persist cadence: a positive value enables the JSON sidecar dump
        # every N callback ticks. Wrapped in a tick-counting callback that
        # delegates to AdaRFTCurriculumCallback for the actual update.
        callback = AdaRFTCurriculumCallback(
            sampler=sampler,
            manifest_index=manifest_index,
            schedule=schedule,
            write_back=True,
        )
        trainer.add_callback(callback)
        persist_n = int(adaptive_cfg.get("persist_difficulty_every_n_ticks", 0))
        if persist_n > 0:
            persist_cb = _DifficultyPersistCallback(
                manifest_index=manifest_index,
                every_n_ticks=persist_n,
            )
            trainer.add_callback(persist_cb)
            logger.info(
                "AdaRFT callback wired; difficulty persist every %d ticks.",
                persist_n,
            )
        else:
            logger.info("AdaRFT callback wired; difficulty persist disabled.")

    trainer.train()
    trainer.save_model(str(args.output_dir))
    logger.info("Saved adapter to %s", args.output_dir)


def _build_adaptive_sampler(
    dataset: Any,
    *,
    grpo_cfg: dict[str, Any],
    adaptive_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """Construct a :class:`CurriculumRepeatingSampler` for the adaptive path.

    ``num_generations`` and ``batch_size`` come from the GRPO config —
    GRPOTrainer's group-relative-advantage requires identical prompts in
    consecutive slots within a group, so the sampler must repeat each
    unique prompt ``num_generations`` times.

    The torch generator is seeded from ``args.seed`` so two runs with the
    same seed make the same band-selection sequence.
    """
    import torch  # noqa: PLC0415

    from .curriculum import CurriculumRepeatingSampler  # noqa: PLC0415

    num_generations = int(grpo_cfg.get("num_generations", 8))
    batch_size = int(grpo_cfg.get("generation_batch_size") or num_generations)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return CurriculumRepeatingSampler(
        dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        target_reward=adaptive_cfg["target_reward"],
        alpha=adaptive_cfg["alpha"],
        eta=adaptive_cfg["eta"],
        ema_decay=adaptive_cfg["ema_decay"],
        band_width=adaptive_cfg["band_width"],
        max_per_subject_in_batch=adaptive_cfg["max_per_subject_in_batch"],
        generator=gen,
    )


def _select_trainer_cls(
    *,
    adarft_enabled: bool,
    static_curriculum_enabled: bool,
    atlas_cache: Any | None,
) -> type:
    """Compose the right GRPOTrainer subclass for the active feature flags.

    The mixin order (outermost first) is::

        splice → adarft → curriculum_static_weights → CurriculumGRPOTrainer →
        GRPOTrainer

    * ``splice`` mixin overrides ``__init__`` (wraps the processor); always
      outermost when active so its ``__init__`` pops the ``atlas_cache``
      kwarg before delegating to the next mixin.
    * ``adarft`` mixin overrides ``_get_train_sampler``; mutually exclusive
      with ``curriculum_static_weights`` (CLI mutex enforces this).
    * ``curriculum_static_weights`` is the legacy
      :class:`curriculum.sampler.CurriculumGRPOTrainer` which overrides
      ``_get_train_dataloader``.

    Six effective combinations (3 curriculum modes × splice on/off) are
    materialised lazily via a small lookup table — each tuple key encodes
    ``(adarft_enabled, static_curriculum_enabled, splice_enabled)``.
    """
    if adarft_enabled and static_curriculum_enabled:
        # Caught at CLI parse time; defensive in case main() is called
        # programmatically with mismatched flags.
        raise ValueError(
            "adarft and static_weights curricula are mutually exclusive — "
            "the CLI mutex should have caught this before _select_trainer_cls."
        )
    splice_enabled = atlas_cache is not None

    from trl import GRPOTrainer  # noqa: PLC0415  # pyright: ignore[reportPrivateImportUsage]

    # Pick the curriculum base. The static-weights path uses TRL's existing
    # CurriculumGRPOTrainer (which overrides _get_train_dataloader); the
    # AdaRFT path uses an additional mixin layered on top of vanilla
    # GRPOTrainer (which overrides _get_train_sampler — the correct hook
    # for sampler swap, per the plan's "RepeatSampler doesn't wrap arbitrary
    # samplers" footnote).
    if static_curriculum_enabled:
        from curriculum.sampler import CurriculumGRPOTrainer  # noqa: PLC0415

        base_cls: type = CurriculumGRPOTrainer
    else:
        base_cls = GRPOTrainer

    if adarft_enabled:
        base_cls = _build_adarft_trainer_cls(base_cls)

    if splice_enabled:
        base_cls = _build_splice_trainer_cls(base_cls)

    return base_cls


def _build_adarft_trainer_cls(grpo_trainer_cls: type) -> type:
    """AdaRFT mixin: override ``_get_train_sampler`` to return the curriculum sampler.

    The plan footnote (TRL's RepeatSampler doesn't wrap arbitrary samplers)
    is what motivates overriding the sampler-construction hook instead of
    the dataloader-construction hook. The
    :class:`CurriculumRepeatingSampler` already handles GRPO's per-prompt
    repetition (``num_generations`` × per-prompt) so the parent dataloader
    can use it verbatim.
    """

    class AdaRFTGRPOTrainer(grpo_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *, curriculum_sampler: Any, **kwargs: Any) -> None:
            self._curriculum_sampler = curriculum_sampler
            super().__init__(**kwargs)

        def _get_train_sampler(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            # TRL's GRPOTrainer signature has shifted across versions; some
            # builds pass the dataset positionally, others pass it as a
            # kwarg. Ignoring all extras keeps this future-proof — the
            # sampler is owned by the curriculum module, not by
            # GRPOTrainer's default plumbing.
            return self._curriculum_sampler

    return AdaRFTGRPOTrainer


def _build_splice_trainer_cls(grpo_trainer_cls: type) -> type:
    """Subclass GRPOTrainer with sidecar emission for the atlas-embedding splice.

    The trainer wraps :attr:`processing_class` with a sidecar-emitting
    proxy. Per-row image paths are stashed on the trainer in
    :meth:`_generate_and_score_completions` so the proxy can look them up
    when the parent class invokes ``processor(images=..., text=...)``.

    Built lazily so importing this module doesn't drag TRL in.
    """

    class SingleTurnGRPOTrainer(grpo_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *, atlas_cache: Any, **kwargs: Any) -> None:
            self._atlas_cache = atlas_cache
            self._current_image_paths: list[list[str]] = []
            base_processor = kwargs["processing_class"]
            kwargs["processing_class"] = _SidecarEmittingProcessor(
                base=base_processor,
                cache=atlas_cache,
                paths_provider=lambda: self._current_image_paths,
            )
            super().__init__(**kwargs)

        def _generate_and_score_completions(self, generation_batch):  # type: ignore[override]
            # Stash per-row image_paths so the processor wrapper can look them
            # up at processor-call time. The batch is a list-of-row-dicts
            # because GRPOTrainer uses ``data_collator=identity`` (line 629
            # in the upstream trl/trainer/grpo_trainer.py).
            self._current_image_paths = [
                list(row.get("image_paths", [])) for row in generation_batch
            ]
            return super()._generate_and_score_completions(generation_batch)

    return SingleTurnGRPOTrainer


class _SidecarEmittingProcessor:
    """Proxy around the VLM processor that adds atlas-cache sidecars to its output.

    The proxy delegates every attribute / method to the underlying processor
    (so :meth:`apply_chat_template`, :attr:`tokenizer`, and friends work
    unchanged) but wraps :meth:`__call__` to:

    1. Run the underlying processor on the same args.
    2. Look up each image's cached SigLIP embedding via the atlas cache.
    3. If at least one hit, append three sidecar tensors to the output:
       ``precomputed_image_mask`` (bool), ``precomputed_cached_flat``
       (concatenated cached embeddings), ``precomputed_cached_patch_counts``
       (per-cached-image patch count).

    The forward pre-hook installed by
    :func:`embeddings.splice.install_atlas_splice` strips these sidecars
    from forward kwargs and stashes them on the inner Gemma 4 model so
    the patched ``get_image_features`` can splice cached embeddings in.
    """

    def __init__(
        self,
        *,
        base: Any,
        cache: Any,
        paths_provider: Any,  # callable returning list[list[str]]
    ) -> None:
        self._base = base
        self._cache = cache
        self._paths_provider = paths_provider

    def __getattr__(self, name: str) -> Any:
        # Delegate every unknown attribute to the base processor.
        return getattr(self._base, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._base(*args, **kwargs)
        images = kwargs.get("images")
        if images is None or self._cache is None:
            return out
        try:
            sidecars = _build_sidecars(images, self._paths_provider(), self._cache)
        except (ValueError, RuntimeError) as exc:
            logger.warning("atlas splice: sidecar build failed (%s); skipping", exc)
            return out
        if sidecars is None:
            return out
        mask, cached_flat, cached_pc = sidecars
        # ``out`` may be a BatchEncoding/dict — both support __setitem__.
        out["precomputed_image_mask"] = mask
        out["precomputed_cached_flat"] = cached_flat
        out["precomputed_cached_patch_counts"] = cached_pc
        return out


def _build_sidecars(
    images: list[Any],
    paths_per_row: list[list[str]],
    cache: Any,
):
    """Build the (mask, cached_flat, cached_patch_counts) sidecar tuple.

    Returns ``None`` if either the path/image structure doesn't align or no
    cache hits are present (the splice falls back to live SigLIP).

    Parameters
    ----------
    images:
        TRL passes a ``list[list[PIL]]`` (per-row, per-image) to processor.
    paths_per_row:
        Parallel ``list[list[str]]`` from the dataset's ``image_paths`` column.
    cache:
        :class:`embeddings.cache.AtlasEmbeddingCache`.
    """
    import torch  # noqa: PLC0415

    # Flatten in the same nested order the processor sees.
    flat_images: list[Any] = []
    for row in images:
        if row is None:
            continue
        flat_images.extend(row)
    flat_paths: list[str] = []
    for row in paths_per_row:
        flat_paths.extend(row)

    if len(flat_paths) != len(flat_images):
        # Misalignment — bail rather than splice the wrong cache entries.
        raise ValueError(
            f"sidecar alignment: flat_images={len(flat_images)} != "
            f"flat_paths={len(flat_paths)}"
        )

    mask_list: list[bool] = []
    cached_tensors: list[Any] = []
    for path in flat_paths:
        emb = cache.lookup_by_path(path)
        if emb is None:
            mask_list.append(False)
            continue
        # Cache files store per-image SigLIP output as
        # ``(1, num_patches, hidden)`` — peel the leading singleton batch dim
        # so per-image tensors stack into a flat ``(sum_P, hidden)`` shape.
        if emb.dim() > 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        mask_list.append(True)
        cached_tensors.append(emb)

    if not cached_tensors:
        return None

    mask = torch.tensor(mask_list, dtype=torch.bool)
    cached_flat = torch.cat(cached_tensors, dim=0)
    cached_pc = torch.tensor([t.shape[0] for t in cached_tensors], dtype=torch.long)
    return mask, cached_flat, cached_pc


# ---------------------------------------------------------------------------
# Live-difficulty persistence callback
# ---------------------------------------------------------------------------


def _make_persist_callback_cls() -> type:
    """Lazily build the persist-callback class against the installed transformers.

    The class is built inside a function so importing this module doesn't
    require ``transformers`` (which the test harness can stub out).
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    class _Cls(TrainerCallback):
        """Persist :meth:`ManifestIndex.persist_live_difficulty` every N ticks.

        Decoupled from :class:`AdaRFTCurriculumCallback` so the callback
        order (update first, then persist) is stable: register update
        before persist, and ``trainer.callbacks`` honours insertion order.
        """

        def __init__(self, *, manifest_index: Any, every_n_ticks: int) -> None:
            super().__init__()
            self._manifest_index = manifest_index
            self._every_n_ticks = max(1, int(every_n_ticks))
            self._tick_count = 0

        def on_log(  # noqa: ANN001
            self, args, state, control, logs=None, **kwargs,
        ):
            self._tick_count += 1
            if self._tick_count % self._every_n_ticks != 0:
                return control
            try:
                n = self._manifest_index.persist_live_difficulty()
                logger.info(
                    "[difficulty_persist] persisted %d live-difficulty rows "
                    "after %d ticks.", n, self._tick_count,
                )
            except OSError as exc:
                logger.warning(
                    "[difficulty_persist] failed (%s); continuing training.",
                    exc,
                )
            return control

    return _Cls


def _DifficultyPersistCallback(*, manifest_index: Any, every_n_ticks: int) -> Any:  # noqa: N802
    """Constructor wrapper that lazily builds + instantiates the callback class.

    Public name uses CapWords because it logically constructs an object;
    the underscore-name wrapping keeps ``transformers`` out of the import
    surface for module-level imports of this file.
    """
    cls = _make_persist_callback_cls()
    return cls(manifest_index=manifest_index, every_n_ticks=every_n_ticks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(sys.argv[1:])
