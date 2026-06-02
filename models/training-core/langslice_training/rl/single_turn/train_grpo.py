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
  :func:`langslice_training.embeddings.splice.install_atlas_splice` on the wrapped model, and
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

    # From repo root via the canonical training launcher. Default mode is
    # adaptive curriculum + adaptive reward + index data source (deterministic
    # canonical slate). For procedural Lane A prefixes drawn from the full
    # ~25k+ RLVR allocation:
    python -m langslice_training.rl.single_turn.train_grpo \\
        --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml \\
        --sft-model out/sft/docker-sft-1716-merged-bf16 \\
        --output-dir out/rlvr_single_turn/ckpt300_lane_a \\
        --data-source index_lane_a \\
        --atlas-embedding-cache out/atlas_embeddings
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import tomllib

from langslice_training.rl.common.train_grpo_helpers import (
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


# ---------------------------------------------------------------------------
# submit_estimate stop-criteria + post-generate eos rewrite
# ---------------------------------------------------------------------------
#
# Background. TRL GRPOTrainer v0.27.x (see trl/trainer/grpo_trainer.py lines
# 1874-1878 / 1921-1925) decides whether a rollout is "truncated" by checking
# *only* whether the last completion token is in {eos_token_id, pad_token_id}::
#
#     if self.mask_truncated_completions:
#         eos_and_pad = [self.eos_token_id, self.pad_token_id]
#         is_truncated = torch.tensor(
#             [ids[-1] not in eos_and_pad for ids in completion_ids_list],
#             device=device,
#         )
#         completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
#
# Stop-strings (transformers ``StopStringCriteria``) halt generation but do
# NOT append eos, so every stop-string-terminated rollout is marked truncated
# and zero-masked -> wasted gradient + misleading completions/clipped_ratio.
#
# Empirically the bare ``}`` stop-string also fires on inner braces in the
# model's reasoning (e.g. ``{atlas_name}`` constructs), producing 26-token
# garbage completions with reward=-1.
#
# Fix (both halves run inside the same ``_generate_with_tokenizer`` wrapper):
#
# 1. ``SubmitEstimateCloseCriteria`` -- string-aware balanced-brace parse over
#    the freshly-generated tail of each row. Returns True for a row only when
#    a ``call:submit_estimate{...}`` marker has appeared AND the matching
#    closing ``}`` has been emitted. ``}`` inside quoted reasoning text does
#    not decrement depth.
# 2. ``rewrite_completions_with_eos_after_submit_close`` -- post-generate
#    walk over the completion portion of ``prompt_completion_ids``. For each
#    row whose completion contains a balanced ``submit_estimate{...}``,
#    overwrite the token *after* the close-brace with eos and pad the rest.
#    That makes TRL's ``ids[-1] in eos_and_pad`` check pass -> no zero-mask.
#
# Both halves are independently testable; see
# ``tests/test_stop_after_submit_close.py``.


SUBMIT_MARKER = "call:submit_estimate{"


def _find_submit_close_index(text: str) -> int | None:
    """Return the character index *after* the balanced close of
    ``call:submit_estimate{...}`` in ``text``, or ``None`` if not yet closed.

    String-aware: ``{`` / ``}`` inside ``"..."`` (with ``\\`` escapes) are
    ignored for depth counting. The marker itself must appear before any
    counted brace; characters before the marker are skipped entirely so
    earlier braces (e.g. f-string placeholders in the reasoning) do not
    poison the count.

    Returns the *exclusive* end position so callers can slice ``text[:end]``
    to get the substring up to and including the close-brace.
    """
    marker_at = text.find(SUBMIT_MARKER)
    if marker_at < 0:
        return None
    # Start scanning at the opening ``{`` of the marker -- depth begins at 0
    # and the marker's own ``{`` ticks us to 1.
    i = marker_at + len(SUBMIT_MARKER) - 1
    n = len(text)
    depth = 0
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def _completion_close_token_index(
    completion_ids_row: Any,  # 1-D LongTensor or list[int]
    tokenizer: Any,
) -> int | None:
    """Return the smallest token index ``k`` in ``completion_ids_row`` such
    that decoding ``completion_ids_row[: k + 1]`` yields a string containing
    a balanced ``call:submit_estimate{...}``. ``None`` if no such close.

    Implementation: decode the whole row once (cheap on the post-generate
    path) and find the close character index, then walk forward through the
    tokens decoding cumulative prefixes until the decoded prefix length
    covers the close char.
    """
    import torch  # local import to keep module-import cheap

    if isinstance(completion_ids_row, torch.Tensor):
        ids_list = completion_ids_row.tolist()
    else:
        ids_list = list(completion_ids_row)

    full_text = tokenizer.decode(ids_list, skip_special_tokens=False)
    close_char = _find_submit_close_index(full_text)
    if close_char is None:
        return None

    # Walk token-by-token. For each prefix length ``k + 1`` decode and check
    # whether the decoded length has reached ``close_char``. This is O(L^2)
    # in tokens decoded, but only runs once per row at end-of-generation and
    # the relevant prefix is < 600 tokens in practice.
    for k in range(len(ids_list)):
        prefix_text = tokenizer.decode(ids_list[: k + 1], skip_special_tokens=False)
        if len(prefix_text) >= close_char:
            return k
    return None


def rewrite_completions_with_eos_after_submit_close(
    prompt_completion_ids: Any,  # (B, P+C) LongTensor
    prompt_length: int,
    tokenizer: Any,
) -> Any:
    """In-place rewrite of the completion portion of ``prompt_completion_ids``.

    For each row whose completion contains a balanced
    ``call:submit_estimate{...}``, the token immediately *after* the closing
    ``}`` is overwritten with ``eos_token_id`` and every subsequent token is
    overwritten with ``pad_token_id``. Rows whose final token is already
    eos/pad are left untouched. Rows with no balanced submit_estimate are
    left untouched (TRL will still mask them via ``mask_truncated_completions``
    -- which is correct, since the model never produced a final answer).

    Returns the (possibly cloned) tensor. TRL/Unsloth run ``model.generate``
    under ``torch.inference_mode()``, so the ids it returns are an *inference
    tensor* that cannot be mutated in place outside inference mode. When that's
    the case we clone to a normal tensor first and mutate the clone, so the
    caller MUST use the return value rather than rely on in-place mutation.
    (For an ordinary tensor the original is still mutated in place and the same
    object is returned, preserving the previous contract.)
    """
    import torch  # local import

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    if not isinstance(prompt_completion_ids, torch.Tensor):
        return prompt_completion_ids  # defensive no-op for non-tensor paths
    if prompt_completion_ids.dim() != 2:
        return prompt_completion_ids
    total_len = prompt_completion_ids.size(1)
    if prompt_length >= total_len:
        return prompt_completion_ids

    # generate() output is an inference tensor (created under
    # torch.inference_mode()); in-place writes below would raise
    # "Inplace update to inference tensor outside InferenceMode is not
    # allowed". Clone to a normal tensor first and mutate the clone -- the
    # caller uses the returned value.
    if torch.is_inference(prompt_completion_ids):
        prompt_completion_ids = prompt_completion_ids.clone()

    completion_slice = prompt_completion_ids[:, prompt_length:]
    completion_len = completion_slice.size(1)
    for row_idx in range(completion_slice.size(0)):
        row = completion_slice[row_idx]
        last_tok = int(row[-1].item())
        if last_tok == eos_id or last_tok == pad_id:
            continue  # already terminated naturally
        close_tok = _completion_close_token_index(row, tokenizer)
        if close_tok is None:
            continue  # no balanced submit_estimate -> leave for TRL to mask
        eos_pos = close_tok + 1
        if eos_pos >= completion_len:
            # Close brace is the very last generated token; overwrite the
            # close-brace itself with eos so TRL's check still passes. This
            # loses the visible ``}`` from logs but does not affect rewards
            # because the reward parser already saw the balanced args.
            row[-1] = eos_id
            continue
        row[eos_pos] = eos_id
        if eos_pos + 1 < completion_len:
            row[eos_pos + 1 :] = pad_id
    return prompt_completion_ids


def _build_submit_estimate_stopping_criteria(
    tokenizer: Any,
    prompt_length: int,
    *,
    decode_tail_tokens: int = 128,
) -> Any:
    """Construct a per-row ``StoppingCriteria`` that fires when the freshly
    generated tail contains a balanced ``call:submit_estimate{...}``.

    ``prompt_length`` is the length of the prompt portion of ``input_ids`` at
    generate-time (so the criterion can slice off the prompt before decoding).
    Only the trailing ``decode_tail_tokens`` of the generated portion are
    decoded each step to keep the per-token cost bounded; rows that have
    already fired stay fired (latch).

    Returns a transformers ``StoppingCriteria`` instance.
    """
    import torch  # local import
    from transformers import StoppingCriteria  # local import

    class _SubmitEstimateCloseCriteria(StoppingCriteria):
        """Per-row stop when a balanced ``submit_estimate{...}`` is emitted.

        Background: see TRL ``grpo_trainer.py`` lines 1874-1878 / 1921-1925
        comment block above -- TRL only counts a rollout as "naturally
        terminated" when the last token is eos/pad. Stop-strings do not
        append eos, so a literal ``}`` stop-string causes every successful
        rollout to be masked. This criterion stops generation at the right
        moment; the companion post-generate rewrite then overwrites the next
        token with eos so TRL sees a clean termination.
        """

        def __init__(self) -> None:
            super().__init__()
            self._fired: torch.BoolTensor | None = None

        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs: Any,
        ) -> torch.BoolTensor:
            device = input_ids.device
            batch_size = input_ids.size(0)
            if self._fired is None or self._fired.size(0) != batch_size:
                self._fired = torch.zeros(batch_size, dtype=torch.bool, device=device)

            full_len = input_ids.size(1)
            if full_len <= prompt_length:
                return self._fired.clone()

            # Decode only the recent tail of the GENERATED portion to amortize.
            gen_start = max(prompt_length, full_len - decode_tail_tokens)
            for row_idx in range(batch_size):
                if bool(self._fired[row_idx].item()):
                    continue
                gen_tail = input_ids[row_idx, gen_start:].tolist()
                tail_text = tokenizer.decode(gen_tail, skip_special_tokens=False)
                # The marker must be visible in the tail for the brace counter
                # to find it. Tail window is 128 tokens; submit_estimate args
                # are typically 30-60 tokens so this is safe.
                if SUBMIT_MARKER not in tail_text:
                    # Marker may have started just before the tail window.
                    # Fall back to decoding the full generated portion only
                    # when the tail contains a ``}`` candidate -- otherwise
                    # we can short-circuit.
                    if "}" not in tail_text:
                        continue
                    gen_all = input_ids[row_idx, prompt_length:].tolist()
                    full_text = tokenizer.decode(gen_all, skip_special_tokens=False)
                    if _find_submit_close_index(full_text) is not None:
                        self._fired[row_idx] = True
                    continue
                if _find_submit_close_index(tail_text) is not None:
                    self._fired[row_idx] = True

            return self._fired.clone()

    return _SubmitEstimateCloseCriteria()


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
            "those images. Re-run "
            "`python -m langslice_training.embeddings.precompute` to fill them.",
            len(missing), sorted(missing),
        )


def _validate_query_cache_loaded(cache: Any, cache_dir: Path) -> None:
    """Mirror :func:`_validate_atlas_cache_coverage`'s posture for the query cache.

    The atlas cache validator warns on missing coverage but tolerates any
    pair-count >= 0; we mirror that for query coverage by **raising** when
    the directory the operator pointed at is empty (likely typo or missing
    precompute). Empty dir + opt-in flag is almost certainly a mistake the
    trainer should refuse to start with.
    """
    if not cache.pairs():
        raise FileNotFoundError(
            f"--query-embedding-cache points at {cache_dir}, but no "
            "<plane>__<dataset>.pt cache files were found there. Run "
            "`python -m langslice_training.embeddings.precompute_query` first."
        )


class _ChainedEmbeddingCache:
    """Two-tier path-keyed cache: atlas takes precedence, query fills the rest.

    Mirrors the public surface of :class:`langslice_training.embeddings.cache.AtlasEmbeddingCache`
    that the sidecar build consumes (just ``lookup_by_path``). ``pairs()`` is
    omitted because the chain doesn't have a single notion of pair —
    callers needing the underlying lists should hold the originals.

    Order is deterministic: atlas hit shadows query hit on path collision.
    """

    def __init__(self, atlas: Any | None, query: Any | None) -> None:
        self._atlas = atlas
        self._query = query

    def lookup_by_path(self, path: str):  # type: ignore[no-untyped-def]
        if self._atlas is not None:
            hit = self._atlas.lookup_by_path(path)
            if hit is not None:
                return hit
        if self._query is not None:
            return self._query.lookup_by_path(path)
        return None


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
                   help="Warm-start: load a previously-saved RL LoRA adapter's "
                   "WEIGHTS ONLY onto a fresh trainer. Optimizer/LR-scheduler/"
                   "RNG/global_step all reset, and the AdaRFT curriculum "
                   "re-explores from its initial T. Use for the first RL launch "
                   "off an SFT/base adapter, or a deliberate warm restart.")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                   help="Full continuation: resume a prior GRPO run from a HF "
                   "Trainer checkpoint directory (containing adapter_model."
                   "safetensors + optimizer.pt + scheduler.pt + rng_state.pth + "
                   "trainer_state.json). Restores optimizer/LR-scheduler/RNG and "
                   "continues at the saved global_step. Mutually exclusive with "
                   "--resume-from-adapter. NOTE: the AdaRFT curriculum sampler "
                   "state (T/ladder/histogram) is NOT stored in the checkpoint "
                   "and re-initializes on resume.")
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
        choices=["index", "terminal_states", "index_lane_a"],
        default="index",
        help="index: pull rows from data/manifest via ManifestIndex (default; "
        "feeds the adaptive curriculum from the full ~25k+ RLVR allocation) "
        "with a deterministic canonical slate per (atlas, plane). "
        "index_lane_a: pull rows from ManifestIndex and synthesize a "
        "procedural Lane A multi-step prefix per row via "
        "langslice_traces.lane_a_prefix; requires --atlas-embedding-cache. "
        "terminal_states: per-trace JSONL produced by "
        "single_turn_rl.terminal_states build (used for runs against an "
        "external trace-factory artifact; requires --terminal-states PATH).",
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
        help=(
            "Optional JSON from "
            "models/langslice-gemma-4/training/tools/seed_difficulty_from_slicebench.py - "
            "bulk-loads per-section seed difficulty into the index at startup."
        ),
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
    p.add_argument(
        "--query-embedding-cache",
        type=Path,
        default=None,
        help="Optional directory containing precomputed SigLIP cache files "
        "for query (slice) images (<plane>__<dataset>.pt). When set, the "
        "splice consults this cache for the per-row query image after "
        "(failing) the atlas-cache lookup. Atlas-cache hits win on any "
        "path collision. Generate via "
        "`python -m langslice_training.embeddings.precompute_query`.",
    )

    # ---- Adaptive-mode knobs (overridable from [adaptive] in TOML) ----
    # These are intentionally optional with default=None so we can tell apart
    # "user passed an explicit value" from "use the TOML default". The merge
    # logic in _resolve_adaptive_cfg honours CLI > TOML > hardcoded default.
    adp = p.add_argument_group(
        "adaptive",
        "Adaptive curriculum + reward knobs (overridable in TOML [adaptive]).",
    )
    adp.add_argument(
        "--adaptive-target-reward", type=float, default=None,
        help="AdaRFT target average reward (beta); T moves up when above, "
        "down when below.",
    )
    adp.add_argument(
        "--adaptive-alpha", type=float, default=None,
        help="AdaRFT tanh slope on the (R - target) error; higher = sharper "
        "T response per unit reward error.",
    )
    adp.add_argument(
        "--adaptive-eta", type=float, default=None,
        help="AdaRFT step size cap on T per update (max change before EMA).",
    )
    adp.add_argument(
        "--adaptive-ema-decay", type=float, default=None,
        help="EMA smoothing factor on the AdaRFT T-step (0 = no smoothing, "
        "1 = freeze).",
    )
    adp.add_argument(
        "--adaptive-band-width", type=float, default=None,
        help="Half-width of the difficulty band [T - bw, T + bw] used to "
        "filter sampling candidates.",
    )
    adp.add_argument(
        "--adaptive-d-min", type=float, default=None,
        help="AdaRFT lower clamp on T. Prevents the curriculum from "
        "collapsing to the easiest-rows-only sampling zone when reward is "
        "persistently below target.",
    )
    adp.add_argument(
        "--adaptive-d-max", type=float, default=None,
        help="AdaRFT upper clamp on T. Prevents the curriculum from "
        "drifting past the hardest-row difficulty when reward is "
        "persistently above target.",
    )
    adp.add_argument(
        "--max-per-subject-in-batch", type=int, default=None,
        help="Subject-diversity cap inside one batch (per-subject upper "
        "bound on selected unique prompts).",
    )
    adp.add_argument(
        "--adaptive-min-visits-per-bin", type=int, default=None,
        help="Coverage quota: minimum number of times each AP bin must be "
        "sampled before AdaRFT is allowed to concentrate. 0 disables.",
    )
    adp.add_argument(
        "--adaptive-quota-slots-per-batch", type=int, default=None,
        help="When the coverage quota is active, reserve this many slots per "
        "batch for picks from the most-deficit bins. 0 disables the quota "
        "even if min_visits_per_bin > 0.",
    )
    adp.add_argument(
        "--adaptive-sigma-quantile", type=float, default=None,
        help="Quantile of the recent error_frac buffer that becomes the "
        "adaptive bell sigma_frac.",
    )
    adp.add_argument(
        "--adaptive-cutoff-quantile", type=float, default=None,
        help="Quantile of the recent error_frac buffer that becomes the "
        "adaptive bell cutoff_frac.",
    )
    adp.add_argument(
        "--adaptive-min-sigma-frac", type=float, default=None,
        help="Lower clamp on the adaptive sigma_frac (prevents collapse to "
        "near-zero on very small errors).",
    )
    adp.add_argument(
        "--adaptive-max-sigma-frac", type=float, default=None,
        help="Upper clamp on the adaptive sigma_frac (prevents blow-up on "
        "early-run all-OOR observations).",
    )
    adp.add_argument(
        "--adaptive-min-observations", type=int, default=None,
        help="Minimum recent error samples before the adaptive schedule "
        "activates; below this, the static fallback is used.",
    )
    adp.add_argument(
        "--adaptive-max-cutoff-frac", type=float, default=None,
        help="Upper clamp on the adaptive cutoff_frac. Format-fail rows "
        "record axis_span as the observed error; without this clamp a "
        "buffer dominated by such rows pushes cutoff toward 1.0 and shapes "
        "reward across the entire axis once in-range answers come back.",
    )
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
    "d_min": 0.0,
    "d_max": 1.0,
    "max_per_subject_in_batch": 2,
    "min_visits_per_bin": 0,
    "quota_slots_per_batch": 0,
    "sigma_quantile": 0.5,
    "cutoff_quantile": 0.95,
    "min_sigma_frac": 0.005,
    "max_sigma_frac": 0.05,
    "max_cutoff_frac": 0.25,
    "min_observations": 50,
    "persist_difficulty_every_n_ticks": 50,
    # Per-bin adaptive sigma: when True, each AP bin derives its bell width
    # from its own recent error history (a range the policy has mastered
    # tightens; a range it's weak at stays lenient) instead of one sigma per
    # plane. ``min_bin_observations`` is the per-bin warmup before a bin trusts
    # its own signal (else it falls back to the plane-global sigma).
    "per_bin": False,
    "min_bin_observations": 12,
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
    "adaptive_d_min": "d_min",
    "adaptive_d_max": "d_max",
    "max_per_subject_in_batch": "max_per_subject_in_batch",
    "adaptive_min_visits_per_bin": "min_visits_per_bin",
    "adaptive_quota_slots_per_batch": "quota_slots_per_batch",
    "adaptive_sigma_quantile": "sigma_quantile",
    "adaptive_cutoff_quantile": "cutoff_quantile",
    "adaptive_min_sigma_frac": "min_sigma_frac",
    "adaptive_max_sigma_frac": "max_sigma_frac",
    "adaptive_max_cutoff_frac": "max_cutoff_frac",
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
    if args.resume_from_adapter is not None and args.resume_from_checkpoint is not None:
        parser.error(
            "--resume-from-adapter and --resume-from-checkpoint are mutually "
            "exclusive: the former warm-starts a fresh trainer from adapter "
            "weights only, the latter continues a prior run with full "
            "optimizer/scheduler/step state. Pick one."
        )
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
    if args.data_source == "index_lane_a" and args.atlas_embedding_cache is None:
        parser.error(
            "--data-source index_lane_a requires --atlas-embedding-cache PATH "
            "(the procedural Lane A generator pre-encodes atlas tiles into the "
            "splice cache; without it every rollout would hit SigLIP for the "
            "full 9-tile slate per row)."
        )
    if (
        args.data_source in ("index", "index_lane_a")
        and args.curriculum_mode == "static_weights"
    ):
        # static_weights expects a WeightedRowDataset (built only by
        # build_datasets() from terminal-state JSONL); build_datasets_from_index
        # returns a plain RowDataset, which would fail an isinstance assert
        # later with a bare AssertionError. Surface a usable error here instead.
        parser.error(
            f"--data-source {args.data_source} is incompatible with "
            "--curriculum-mode static_weights — static_weights requires the "
            "terminal_states data source (it expects a WeightedRowDataset "
            "built from per-trace specs). For index-driven data, use "
            "--curriculum-mode adaptive or none."
        )
    if (
        args.data_source in ("index", "index_lane_a")
        and args.terminal_states is not None
    ):
        # Not strictly an error, but flag it because it's almost certainly a
        # mistake — the index path doesn't read --terminal-states.
        logger.warning(
            "--terminal-states %s is ignored under --data-source=%s.",
            args.terminal_states, args.data_source,
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
    # Warn about unrecognised keys in the TOML [adaptive] table so typos like
    # `target_rward = 0.7` are surfaced rather than silently ignored.
    unknown = set(toml_adaptive.keys()) - set(_ADAPTIVE_DEFAULTS.keys())
    if unknown:
        logger.warning(
            "[adaptive] TOML table contains unrecognised key(s) that will be "
            "ignored: %s. Known keys: %s.",
            sorted(unknown), sorted(_ADAPTIVE_DEFAULTS.keys()),
        )
    # Coerce numerics defensively.
    int_keys = {
        "max_per_subject_in_batch",
        "min_visits_per_bin",
        "quota_slots_per_batch",
        "min_observations",
        "min_bin_observations",
        "persist_difficulty_every_n_ticks",
    }
    bool_keys = {"per_bin"}
    for k, v in list(out.items()):
        if k in bool_keys:
            out[k] = bool(v)
        elif k in int_keys:
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def _install_trl_tool_role_prompt_compat() -> None:
    """Backport TRL PR #4300 (tool-role generation prompts) onto TRL < 1.0.

    Our single-turn rollouts hand GRPO a ``prompt`` whose final message has
    role ``"tool"`` (the pre-rendered ``fetch_atlas`` result); the policy is
    expected to generate the next assistant turn (``submit_estimate``). TRL
    0.23.1's ``apply_chat_template`` only accepts a prompt ending in ``"user"``
    or ``"assistant"`` and raises ``ValueError: Invalid role in the last
    message: tool`` for anything else. TRL >= 1.0 (PR #4300) treats a trailing
    ``"tool"`` message exactly like ``"user"`` (i.e. ``add_generation_prompt``).
    This shim restores that behaviour by intercepting only the tool-terminated,
    prompt-only example our dataset emits and delegating every other shape to
    the original function. ``maybe_apply_chat_template`` resolves
    ``apply_chat_template`` via the module global, so patching the attribute is
    picked up by TRL's own call sites. Idempotent; a no-op on TRL >= 1.0.
    """
    import trl  # noqa: PLC0415
    import trl.data_utils as _du  # noqa: PLC0415

    # Tool-role generation prompts are native in TRL >= 1.0 (PR #4300 — the line
    # that produced v1.0). Only backport onto older TRL (0.23.x) that lacks it;
    # standing down on >= 1.0 keeps v1.0's exact native rendering.
    try:
        if int(str(trl.__version__).split(".", 1)[0]) >= 1:
            return
    except (ValueError, AttributeError):
        pass

    if getattr(_du.apply_chat_template, "_langslice_tool_compat", False):
        return

    _orig_apply_chat_template = _du.apply_chat_template
    _other_keys = {"chosen", "rejected", "completion", "messages", "label"}

    def _apply_chat_template_tool_compat(
        example: dict[str, Any],
        tokenizer: Any,
        tools: Any = None,
        **template_kwargs: Any,
    ) -> dict[str, Any]:
        prompt = example.get("prompt") if isinstance(example, dict) else None
        if (
            isinstance(prompt, list)
            and prompt
            and isinstance(prompt[-1], dict)
            and prompt[-1].get("role") == "tool"
            and not (_other_keys & set(example))
        ):
            # Same call the "user" branch of TRL's apply_chat_template makes,
            # but reached for a trailing "tool" message. The original messages
            # (role still "tool") are passed through unchanged so the Gemma
            # chat template renders the tool response correctly; only TRL's
            # user/assistant gate is bypassed.
            rendered = tokenizer.apply_chat_template(
                prompt,
                tools=tools,
                continue_final_message=False,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            return {**example, "prompt": rendered}
        return _orig_apply_chat_template(example, tokenizer, tools=tools, **template_kwargs)

    _apply_chat_template_tool_compat._langslice_tool_compat = True  # type: ignore[attr-defined]
    _du.apply_chat_template = _apply_chat_template_tool_compat
    logger.info(
        "Installed TRL tool-role prompt-compat shim (trailing role='tool' -> "
        "add_generation_prompt=True; backport of TRL PR #4300 for TRL < 1.0)."
    )


def main(argv: list[str] | None = None) -> None:
    # Ensure stdout sees INFO-level logs even when invoked via the
    # wrapper launchers (which do NOT reach the
    # ``if __name__ == "__main__":`` block below, so the shim path used to
    # silently drop every ``logger.info`` call). ``basicConfig`` is a no-op
    # if the root logger already has handlers, so this is safe when the
    # module IS invoked directly.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    # Mirror INFO for our own package even if some upstream installed a
    # WARNING-only root handler (Unsloth historically has). cheap idempotent.
    logging.getLogger("single_turn_rl").setLevel(logging.INFO)
    print(
        "[langslice-grpo] train_grpo.main() reached; logging configured "
        "(level=INFO, single_turn_rl.* enabled)",
        flush=True,
    )

    args = _parse_args(argv)

    # Flag (or, with LANGSLICE_STRICT_FAST_IO=1, refuse) hot training I/O paths
    # that resolve onto the slow Windows/9p bind instead of the fast WSL2 ext4
    # volume (out/cache_fast). No-op off-Linux / when the mount table is unknown.
    from langslice_training.utils.fast_io import warn_if_slow_io

    warn_if_slow_io(
        {
            "--output-dir": getattr(args, "output_dir", None),
            "--atlas-embedding-cache": getattr(args, "atlas_embedding_cache", None),
            "--query-embedding-cache": getattr(args, "query_embedding_cache", None),
        }
    )

    config = _load_config(args.config)

    grpo_cfg = dict(config.get("grpo", {}))
    if args.report_to is not None:
        grpo_cfg["report_to"] = "none" if args.report_to.lower() == "none" else args.report_to
    lora_cfg = dict(config.get("lora", {}))
    data_cfg = dict(config.get("data", {}))
    reward_cfg = dict(config.get("reward", {}))
    adaptive_cfg = _resolve_adaptive_cfg(args, dict(config.get("adaptive", {})))

    # Self-documenting run card: snapshot the fully-resolved config + lineage into
    # the output dir at launch. Best-effort — wrapped so a logging hiccup (or a
    # missing module) can never abort the run.
    try:
        from langslice_training.run_logging import write_run_card

        write_run_card(
            output_dir=args.output_dir,
            kind="grpo",
            config_path=args.config,
            resolved={
                "grpo": grpo_cfg,
                "lora": lora_cfg,
                "data": data_cfg,
                "reward": reward_cfg,
                "adaptive": adaptive_cfg,
            },
            cli_args=vars(args),
            base_model=getattr(args, "sft_model", None) or grpo_cfg.get("base_model"),
            dataset=getattr(args, "data_source", None),
        )
    except Exception as _rc_exc:  # noqa: BLE001 — logging must never break training
        print(f"[langslice-grpo] run-card write skipped: {_rc_exc!r}")

    # ---- Mode flags (resolved up front so the rest of main reads cleanly) ----
    use_static_weights = args.curriculum_mode == "static_weights"
    use_adaptive_curriculum = args.curriculum_mode == "adaptive"
    use_adaptive_reward = args.reward_mode == "adaptive"
    use_index_data = args.data_source in ("index", "index_lane_a")
    use_lane_a_prefix = args.data_source == "index_lane_a"

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
                f"Internal: --data-source={args.data_source} but ManifestIndex "
                "was not built."
            )
        from .dataset import build_datasets_from_index  # noqa: PLC0415

        slate_root = args.repo_root / "models" / "langslice-gemma-4" / "data"
        if use_lane_a_prefix:
            import random as _random  # noqa: PLC0415

            train_dataset, eval_dataset = build_datasets_from_index(
                manifest_index=manifest_index,
                split=str(data_cfg.get("split", "rlvr")) or None,
                repo_root=args.repo_root,
                slate_root=slate_root,
                n_positions=int(data_cfg.get("slate_n_positions", 9)),
                eval_holdout_every=int(data_cfg.get("eval_holdout_every", 5)),
                max_examples=data_cfg.get("max_examples"),
                seed=args.seed,
                randomized=True,
                strategy="lane_a_prefix",
                atlas_embedding_cache_dir=args.atlas_embedding_cache,
                rng=_random.Random(args.seed),
            )
            data_source_label = (
                f"manifest_index({args.manifest_root})+lane_a_prefix"
            )
        else:
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
        from langslice_training.curriculum.weights import (  # noqa: PLC0415
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
            max_cutoff_frac=adaptive_cfg["max_cutoff_frac"],
            sigma_quantile=adaptive_cfg["sigma_quantile"],
            cutoff_quantile=adaptive_cfg["cutoff_quantile"],
            min_observations=adaptive_cfg["min_observations"],
            per_bin=adaptive_cfg["per_bin"],
            min_bin_observations=adaptive_cfg["min_bin_observations"],
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
            "sigma q=%.2f, cutoff q=%.2f, warmup=%d, per_bin=%s (bin_warmup=%d).",
            adaptive_cfg["min_sigma_frac"], adaptive_cfg["max_sigma_frac"],
            adaptive_cfg["sigma_quantile"], adaptive_cfg["cutoff_quantile"],
            adaptive_cfg["min_observations"],
            adaptive_cfg["per_bin"], adaptive_cfg["min_bin_observations"],
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
            from langslice_training.curriculum.bins import compute_section_bins  # noqa: PLC0415

            singles_stub = specs_to_single_slice_examples(
                train_dataset._specs  # noqa: SLF001
            )
            section_bin_lookup = compute_section_bins(singles_stub)
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError, KeyError) as exc:
            logger.warning(
                "Curriculum logging: bin computation skipped (%s); "
                "log rows will record bin_idx=-1.", exc,
            )

        from langslice_training.curriculum.log import CurriculumLogger  # noqa: PLC0415

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

    # ---- Atlas / query embedding caches (both opt-in) ----
    atlas_cache: Any = None
    if args.atlas_embedding_cache is not None:
        from langslice_training.embeddings.cache import AtlasEmbeddingCache  # noqa: PLC0415

        atlas_cache = AtlasEmbeddingCache(args.atlas_embedding_cache)
        if not atlas_cache.pairs():
            raise FileNotFoundError(
                f"--atlas-embedding-cache points at {args.atlas_embedding_cache}, "
                "but no <atlas>_<plane>.pt cache files were found there. Run "
                "`python -m langslice_training.embeddings.precompute` first."
            )
        _validate_atlas_cache_coverage(atlas_cache, train_dataset)

    query_cache: Any = None
    if args.query_embedding_cache is not None:
        from langslice_training.embeddings.query_cache import QueryEmbeddingCache  # noqa: PLC0415

        query_cache = QueryEmbeddingCache(args.query_embedding_cache)
        _validate_query_cache_loaded(query_cache, args.query_embedding_cache)

    # Chain: atlas-precedes-query so atlas hits shadow query hits on collision.
    splice_cache: Any = None
    if atlas_cache is not None and query_cache is not None:
        splice_cache = _ChainedEmbeddingCache(atlas=atlas_cache, query=query_cache)
    elif atlas_cache is not None:
        splice_cache = atlas_cache
    elif query_cache is not None:
        splice_cache = _ChainedEmbeddingCache(atlas=None, query=query_cache)

    # ---- Adaptive curriculum sampler (built before TRL imports so the
    # smoke-test path can short-circuit just before model load) ----
    sampler: Any = None
    bin_difficulty: Any = None
    if use_adaptive_curriculum:
        # The sampler is built early so AdaRFTCurriculumCallback can be
        # constructed and wired into the trainer's callbacks list. Heavy
        # imports stay deferred — the sampler module only needs torch,
        # which is already a hard dep of this whole training stack.
        from .adaptive_reward import (
            _AP_BIN_COUNT as _AR_N_BINS,
        )
        from .adaptive_reward import (  # noqa: PLC0415
            DEFAULT_BUFFER_MAXLEN as _AR_BUFFER,
        )
        from .curriculum import BinDifficultyMap  # noqa: PLC0415

        bin_difficulty = BinDifficultyMap()
        sampler = _build_adaptive_sampler(
            train_dataset,
            grpo_cfg=grpo_cfg,
            adaptive_cfg=adaptive_cfg,
            seed=args.seed,
            bin_difficulty=bin_difficulty,
        )
        # Logger-agnostic stdout breadcrumb so the user can verify wire-up
        # from launch.log regardless of any upstream logging-handler quirks.
        print(
            f"[langslice-grpo] adaptive curriculum: AP-bin keying, "
            f"{_AR_N_BINS} bins, buffer {_AR_BUFFER} obs",
            flush=True,
        )
        logger.info(
            "Adaptive curriculum sampler built: target_reward=%.2f, "
            "alpha=%.2f, eta=%.3f, ema_decay=%.2f, band_width=%.2f, "
            "d_min=%.2f, d_max=%.2f, subject_cap=%d, "
            "min_visits_per_bin=%d, quota_slots_per_batch=%d.",
            adaptive_cfg["target_reward"], adaptive_cfg["alpha"],
            adaptive_cfg["eta"], adaptive_cfg["ema_decay"],
            adaptive_cfg["band_width"],
            adaptive_cfg["d_min"], adaptive_cfg["d_max"],
            adaptive_cfg["max_per_subject_in_batch"],
            adaptive_cfg["min_visits_per_bin"],
            adaptive_cfg["quota_slots_per_batch"],
        )
        logger.info(
            "Adaptive curriculum: AP-bin keying, %d bins, buffer %d obs",
            _AR_N_BINS, _AR_BUFFER,
        )

    # ---- Heavy imports happen here so unit tests can skip them ----
    # Mirror the SFT pipeline's env + SDPA pinning per
    # reference_unsloth_ce_and_sdpa_env_vars and
    # reference_sdpa_math_dispatcher_gotcha. Must run BEFORE TRL import.
    import os as _os  # noqa: PLC0415
    _os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "1")

    from unsloth import FastVisionModel  # noqa: PLC0415
    _install_optional_dep_stubs()

    # Gemma 4 head_dim=512 rules out FA2; pin mem_efficient + cuDNN + math
    # (variable patch counts in live SigLIP can produce shapes both reject
    # without math as fallback). Matches sft/train_sft.py:468-476.
    import torch as _torch_sdpa  # noqa: PLC0415
    _torch_sdpa.backends.cuda.enable_mem_efficient_sdp(True)
    _torch_sdpa.backends.cuda.enable_cudnn_sdp(True)
    _torch_sdpa.backends.cuda.enable_math_sdp(True)
    _torch_sdpa.backends.cuda.enable_flash_sdp(True)
    import torch._dynamo as _torch_dynamo  # noqa: PLC0415
    _torch_dynamo.config.recompile_limit = 64

    from trl import (  # noqa: PLC0415
        GRPOConfig,  # pyright: ignore[reportPrivateImportUsage]
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

    # Both resume paths point at a LoRA adapter directory; --resume-from-checkpoint
    # additionally carries optimizer/scheduler/RNG/step state that HF's
    # Trainer.train(resume_from_checkpoint=...) restores below. The adapter
    # structure + weights must exist on the model *before* that restore, so we
    # load it here for either flag. (The two are mutually exclusive — enforced
    # in _enforce_arg_mutex — so at most one is set.)
    _resume_adapter_dir = args.resume_from_checkpoint or args.resume_from_adapter
    if _resume_adapter_dir is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Loading trainable LoRA adapter from %s", _resume_adapter_dir)
        model = PeftModel.from_pretrained(
            model, str(_resume_adapter_dir), is_trainable=True,
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
    # wrapped one). Idempotent — no-op if already installed. Splice is
    # required by both the atlas-cache and query-cache paths AND by the
    # within-step query dedup (which routes "fresh first-occurrence
    # encodes" through the same sidecar mechanism).
    splice_active = splice_cache is not None
    if splice_active:
        from langslice_training.embeddings.splice import install_atlas_splice  # noqa: PLC0415

        # Wrap the splice's vision-tower / projector forward in no_grad when
        # the LoRA config has vision frozen — saves ~0.5-1.5 GB at typical
        # batch sizes. Flips off automatically when finetune_vision_layers
        # is enabled.
        frozen_vision = not bool(lora_cfg.get("finetune_vision_layers", False))
        install_atlas_splice(model, frozen_vision=frozen_vision)
        atlas_n = len(atlas_cache.pairs()) if atlas_cache is not None else 0
        query_n = len(query_cache.pairs()) if query_cache is not None else 0
        logger.info(
            "embedding splice installed: %d atlas pair file(s), %d query "
            "pair file(s)",
            atlas_n, query_n,
        )

    # Pop LangSlice-only knobs that don't belong in GRPOConfig before the
    # MRO-aware filter sees them. ``submit_estimate_stop_criterion`` controls
    # the custom stopping criterion + post-generate eos rewrite installed
    # below; default True.
    _use_submit_stop = bool(grpo_cfg.pop("submit_estimate_stop_criterion", True))

    # Our rollouts end on a role="tool" message (the pre-rendered fetch_atlas
    # result); TRL 0.23.1 raises on that, TRL >= 1.0 accepts it. Install the
    # compat shim before any prompt is templated (no-op on TRL >= 1.0).
    _install_trl_tool_role_prompt_compat()

    # Unsloth's compiled GRPO trainer derives the per-token-logps row-chunk count
    # as ``total_rows // autotune_rows_per_chunk`` with NO max(1, ...) guard
    # (UnslothGRPOTrainer._get_per_token_logps_and_entropies). For a small
    # generation batch (generation_batch_size=3) and Gemma 4's large vocab the
    # autotuned rows-per-chunk exceeds total_rows, so that floor-divide is 0 and
    # ``torch.chunk(..., chunks=0)`` raises "chunks ... got: 0". Setting the field
    # explicitly routes both the logps and loss paths to their safe branches
    # (``B = unsloth_grpo_mini_batch``, capped at total_rows by torch.chunk).
    # B=1 (one row-chunk) matches what the *guarded* loss-path autotune already
    # picks for our batch (max(1, total_rows//big)); logit-memory is bounded by
    # ``unsloth_logit_chunk_multiplier`` independently, so this does not raise VRAM.
    # Set on the instance *after* construction to bypass both the MRO filter
    # (vanilla trl.GRPOConfig lacks the field) and Unsloth's config-init gate; the
    # runtime only ever reads ``self.args.unsloth_grpo_mini_batch``.
    _unsloth_mini_batch = grpo_cfg.pop("unsloth_grpo_mini_batch", 1)

    grpo_cfg = _filter_grpo_config_for_installed_trl(GRPOConfig, grpo_cfg)
    training_args = GRPOConfig(output_dir=str(args.output_dir), seed=args.seed, **grpo_cfg)
    if _unsloth_mini_batch is not None:
        _gen_batch = int(getattr(training_args, "generation_batch_size", None)
                         or training_args.num_generations or 1)
        training_args.unsloth_grpo_mini_batch = max(1, min(int(_unsloth_mini_batch), _gen_batch))
        logger.info("Set unsloth_grpo_mini_batch=%s (gen_batch=%s) to bypass "
                    "Unsloth's B=0 row-chunk autotune crash",
                    training_args.unsloth_grpo_mini_batch, _gen_batch)

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
        atlas_cache=splice_cache,
    )
    if splice_active and use_adaptive_curriculum:
        trainer = trainer_cls(
            atlas_cache=splice_cache,
            curriculum_sampler=sampler,
            **trainer_kwargs,
        )
    elif splice_active:
        trainer = trainer_cls(atlas_cache=splice_cache, **trainer_kwargs)
    elif use_adaptive_curriculum:
        trainer = trainer_cls(curriculum_sampler=sampler, **trainer_kwargs)
    else:
        trainer = trainer_cls(**trainer_kwargs)

    # Wire the AdaRFT callback after construction so we can pass the actual
    # sampler instance the trainer holds. Mirrors the pattern documented in
    # AdaRFTCurriculumCallback's docstring (callback hooks ``on_log``).
    if use_adaptive_curriculum and bin_difficulty is not None and schedule is not None:
        from .curriculum import AdaRFTCurriculumCallback  # noqa: PLC0415

        # Persist cadence: a positive value enables the JSON sidecar dump
        # every N callback ticks. Wrapped in a tick-counting callback that
        # delegates to AdaRFTCurriculumCallback for the actual update.
        callback = AdaRFTCurriculumCallback(
            sampler=sampler,
            bin_difficulty=bin_difficulty,
            schedule=schedule,
            write_back=True,
        )
        trainer.add_callback(callback)
        persist_n = int(adaptive_cfg.get("persist_difficulty_every_n_ticks", 0))
        if persist_n > 0 and manifest_index is not None:
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

    # Inject the tokenizer into trainer.model.generate kwargs so transformers'
    # StopStringCriteria can resolve. TRL's GRPOTrainer otherwise does not
    # pass `tokenizer` to model.generate, which makes any stop_strings in
    # generation_kwargs raise:
    #   ValueError: There are one or more stop strings ... but we could not
    #   locate a tokenizer.
    # We grab the existing bound method (which already has Unsloth's RL clone
    # wrapper installed during GRPOTrainer.__init__) and wrap it so the
    # tokenizer is in kwargs before generate runs. The clone-wrapper chain is
    # preserved because we call the saved bound method.
    #
    # The same wrapper also installs ``SubmitEstimateCloseCriteria`` and runs
    # the post-generate eos-rewrite -- see the module-level comment block
    # above ``SUBMIT_MARKER`` for the bug context (TRL grpo_trainer.py lines
    # 1874-1878 / 1921-1925). The wrapper is enabled whenever stop_strings is
    # configured OR whenever the submit-close criterion is requested via the
    # ``[grpo]`` config flag ``submit_estimate_stop_criterion`` (default
    # True). Setting it to False disables both the criterion and the rewrite
    # for ablations.
    _stop_strings = (grpo_cfg.get("generation_kwargs") or {}).get("stop_strings")
    # ``_use_submit_stop`` was popped from grpo_cfg above (before the GRPOConfig
    # filter ran) so the value lives in the local of the same name.
    if _stop_strings or _use_submit_stop:
        _stop_tokenizer = getattr(processor, "tokenizer", processor)
        _orig_generate = trainer.model.generate

        def _generate_with_tokenizer(
            *args: Any,
            _orig: Any = _orig_generate,
            _tok: Any = _stop_tokenizer,
            _install_criterion: bool = _use_submit_stop,
            **kwargs: Any,
        ) -> Any:
            kwargs.setdefault("tokenizer", _tok)
            # Capture prompt length BEFORE generate so the post-generate
            # rewrite knows where the completion portion begins. TRL passes
            # input_ids via **generate_inputs (see grpo_trainer.py line 1414)
            # so it always lands in kwargs here, never positional.
            _input_ids = kwargs.get("input_ids")
            _has_input_ids = _input_ids is not None and hasattr(_input_ids, "dim")
            _prompt_length = None
            if _has_input_ids and _input_ids.dim() >= 2:
                _prompt_length = int(_input_ids.size(1))
            if _install_criterion and _prompt_length is not None:
                criterion = _build_submit_estimate_stopping_criteria(
                    _tok, prompt_length=_prompt_length,
                )
                existing = kwargs.get("stopping_criteria")
                if existing is None:
                    kwargs["stopping_criteria"] = [criterion]
                else:
                    # transformers accepts either a StoppingCriteriaList or a
                    # plain list; either way ``.append`` works.
                    try:
                        existing.append(criterion)
                    except AttributeError:
                        kwargs["stopping_criteria"] = list(existing) + [criterion]

            out = _orig(*args, **kwargs)

            if _install_criterion and _prompt_length is not None:
                try:
                    out = rewrite_completions_with_eos_after_submit_close(
                        out, prompt_length=_prompt_length, tokenizer=_tok,
                    )
                except (RuntimeError, AttributeError, ValueError) as exc:
                    # Defensive: never let the post-rewrite block training.
                    # Underlying issues will surface in tests/logs.
                    logger.warning(
                        "Post-generate eos rewrite skipped (%s); TRL will "
                        "mask this batch as truncated.", exc,
                    )
            return out

        trainer.model.generate = _generate_with_tokenizer  # type: ignore[method-assign]
        logger.info(
            "Patched trainer.model.generate: stop_strings=%r, "
            "submit_estimate_stop_criterion=%s",
            _stop_strings, _use_submit_stop,
        )

    if args.resume_from_checkpoint is not None:
        logger.info(
            "Full-state resume from checkpoint %s: restoring optimizer / "
            "LR-scheduler / RNG / global_step. (AdaRFT curriculum state is not "
            "persisted in the checkpoint and re-initializes.)",
            args.resume_from_checkpoint,
        )
        trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint))
    else:
        trainer.train()
    trainer.save_model(str(args.output_dir))
    logger.info("Saved adapter to %s", args.output_dir)


def _build_adaptive_sampler(
    dataset: Any,
    *,
    grpo_cfg: dict[str, Any],
    adaptive_cfg: dict[str, Any],
    seed: int,
    bin_difficulty: Any | None = None,
) -> Any:
    """Construct a :class:`CurriculumRepeatingSampler` for the adaptive path.

    ``num_generations`` and ``batch_size`` come from the GRPO config —
    GRPOTrainer's group-relative-advantage requires identical prompts in
    consecutive slots within a group, so the sampler must repeat each
    unique prompt ``num_generations`` times.

    ``repeat_count`` is computed from the GRPO config the same way
    GRPOTrainer's default :class:`trl.trainer.utils.RepeatSampler` computes
    it (``num_iterations * steps_per_generation`` — see
    ``trl/trainer/grpo_trainer.py:921``). When the TOML omits
    ``steps_per_generation`` GRPOConfig defaults it to
    ``gradient_accumulation_steps``, so the sampler must consult that key
    too. Passing the wrong ``repeat_count`` would emit fewer indices than
    the dataloader expects and break the inner generation-reuse loop.

    The torch generator is seeded from ``args.seed`` so two runs with the
    same seed make the same band-selection sequence.

    ``bin_difficulty`` (when provided) is wired into the sampler so each
    ``__iter__`` re-pulls live per-AP-bin difficulty before the band
    query — closing the curriculum loop instead of leaving difficulties
    frozen at construction time.
    """
    import torch  # noqa: PLC0415

    from .curriculum import CurriculumRepeatingSampler  # noqa: PLC0415

    num_generations = int(grpo_cfg.get("num_generations", 8))
    batch_size = int(grpo_cfg.get("generation_batch_size") or num_generations)
    repeat_count = _grpo_repeat_count(grpo_cfg)
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
        d_min=adaptive_cfg["d_min"],
        d_max=adaptive_cfg["d_max"],
        max_per_subject_in_batch=adaptive_cfg["max_per_subject_in_batch"],
        min_visits_per_bin=adaptive_cfg["min_visits_per_bin"],
        quota_slots_per_batch=adaptive_cfg["quota_slots_per_batch"],
        generator=gen,
        bin_difficulty=bin_difficulty,
        repeat_count=repeat_count,
    )


def _grpo_repeat_count(grpo_cfg: dict[str, Any]) -> int:
    """Mirror GRPOConfig's effective ``steps_per_generation * num_iterations``.

    GRPOConfig (``trl/trainer/grpo_config.py``) defaults
    ``steps_per_generation`` to ``gradient_accumulation_steps`` when neither
    ``generation_batch_size`` nor ``steps_per_generation`` is set, and
    derives it as ``generation_batch_size // (per_device_train_batch_size *
    num_processes)`` when only ``generation_batch_size`` is set.
    Single-process training is the only configuration this trainer ships
    with today (DDP is parked); the multi-process world-size factor would
    have to be threaded in here too if that changes.
    """
    num_iterations = int(grpo_cfg.get("num_iterations", 1))
    explicit_steps = grpo_cfg.get("steps_per_generation")
    if explicit_steps is not None:
        steps_per_generation = int(explicit_steps)
    else:
        gen_batch = grpo_cfg.get("generation_batch_size")
        per_device = int(grpo_cfg.get("per_device_train_batch_size", 1))
        if gen_batch is not None:
            if per_device <= 0:
                per_device = 1
            steps_per_generation = max(1, int(gen_batch) // per_device)
        else:
            steps_per_generation = int(grpo_cfg.get("gradient_accumulation_steps", 1))
    return max(1, num_iterations * steps_per_generation)


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
        from langslice_training.adaptive.curriculum.sampler import (
            CurriculumGRPOTrainer,  # noqa: PLC0415
        )

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
            # Re-class the processor in place to a dynamic subclass instead of
            # wrapping in a separate proxy. TRL's GRPOTrainer rejects anything
            # that isn't a PreTrainedTokenizerBase / ProcessorMixin instance,
            # so a plain proxy fails isinstance. The dynamic subclass keeps
            # isinstance happy while still overriding __call__ to emit splice
            # sidecars.
            kwargs["processing_class"] = _install_splice_in_place(
                base_processor,
                cache=atlas_cache,
                paths_provider=lambda: self._current_image_paths,
                encoder_provider=lambda: self._dedup_encoder,
            )
            # Lazy-built encoder closure: the model isn't accessible until
            # super().__init__ completes (TRL stashes it on ``self.model``).
            # Build it on first sidecar-time access via ``encoder_provider``.
            # The encoder is None until ``self.model`` is available; the
            # provider closure picks it up on first call.
            self._dedup_encoder: Any | None = None
            super().__init__(**kwargs)
            # Best-effort: install the dedup encoder now if super().__init__
            # set ``self.model`` (real TRL path). Some test stubs skip the
            # GRPOTrainer parent and never assign self.model — in that case
            # leave the encoder None and let dedup degrade gracefully (the
            # sidecar path still works for disk-cached entries).
            if hasattr(self, "model") and self.model is not None:
                self._dedup_encoder = _make_dedup_encoder(self.model)

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


def _install_splice_in_place(
    base_processor: Any,
    *,
    cache: Any,
    paths_provider: Any,
    encoder_provider: Any | None = None,
) -> Any:
    """Re-class ``base_processor`` so its ``__call__`` emits splice sidecars.

    Returns the same instance with its ``__class__`` swapped to a dynamic
    subclass of ``type(base_processor)``. This preserves isinstance checks
    against ``ProcessorMixin`` / ``PreTrainedTokenizerBase`` (which TRL's
    GRPOTrainer enforces strictly via line 2177 of UnslothGRPOTrainer.py)
    while overriding ``__call__`` to append the three splice sidecars.

    Used by the splice mixin instead of the older ``_SidecarEmittingProcessor``
    proxy because the proxy isn't an isinstance match.

    ``encoder_provider`` (optional) is a zero-arg callable returning the
    dedup encoder (a ``(pv, pid) -> output`` callable, see
    :func:`_make_dedup_encoder`). When set, within-step query dedup is
    enabled — duplicate paths within one batch are encoded once and the
    result reused for all duplicates. Provider-style indirection lets the
    trainer's ``__init__`` install the processor BEFORE ``self.model``
    exists (the encoder closes over the model, which TRL stashes during
    super().__init__).
    """
    if type(base_processor).__module__ == "unittest.mock":
        return _SidecarEmittingProcessor(
            base=base_processor,
            cache=cache,
            paths_provider=paths_provider,
        )

    base_cls = type(base_processor)
    base_call = base_cls.__call__

    class SpliceProcessor(base_cls):  # type: ignore[valid-type, misc]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            out = base_call(self, *args, **kwargs)
            images = kwargs.get("images")
            if images is None or self._splice_cache is None:
                return out
            pv = out.get("pixel_values") if hasattr(out, "get") else None
            pid = out.get("image_position_ids") if hasattr(out, "get") else None
            # Dedup-encode path disabled (2026-05-16): when the vision_tower
            # has any CPU-offloaded weights under Unsloth, the manual
            # ``inner.get_image_features(pv, pid)`` call in the dedup encoder
            # raises a CPU/CUDA mm device mismatch. We rely solely on the
            # disk-side atlas cache, which gives 100% hits on the staged
            # Lane A corpus. The query image gets SigLIP'd live every
            # rollout — small extra cost vs the previous all-rollouts-encode
            # behaviour.
            encoder = None
            try:
                sidecars = _build_sidecars(
                    images,
                    self._splice_paths_provider(),
                    self._splice_cache,
                    pixel_values=pv,
                    image_position_ids=pid,
                    encode_fn=encoder,
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning("atlas splice: sidecar build failed (%s); skipping", exc)
                return out
            if sidecars is None:
                return out
            mask, cached_flat, cached_pc = sidecars
            out["precomputed_image_mask"] = mask
            out["precomputed_cached_flat"] = cached_flat
            out["precomputed_cached_patch_counts"] = cached_pc
            return out

    base_processor.__class__ = SpliceProcessor
    base_processor._splice_cache = cache
    base_processor._splice_paths_provider = paths_provider
    base_processor._splice_encoder_provider = encoder_provider
    return base_processor


class _SidecarEmittingProcessor:
    """Compatibility proxy that appends splice sidecars to processor output.

    Kept for unit-test and helper compatibility; production GRPO code uses
    ``_install_splice_in_place`` to satisfy TRL's strict isinstance checks.
    """

    def __init__(self, *, base: Any, cache: Any, paths_provider: Any) -> None:
        self._base = base
        self._cache = cache
        self._paths_provider = paths_provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._base(*args, **kwargs)
        images = kwargs.get("images")
        if images is None:
            return out
        try:
            sidecars = _build_sidecars(
                images=images,
                paths_per_row=self._paths_provider(),
                cache=self._cache,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("atlas splice: sidecar build failed (%s); skipping", exc)
            return out
        if sidecars is None:
            return out
        mask, cached_flat, cached_pc = sidecars
        out["precomputed_image_mask"] = mask
        out["precomputed_cached_flat"] = cached_flat
        out["precomputed_cached_patch_counts"] = cached_pc
        return out


def _make_dedup_encoder(top_model: Any) -> Any:
    """Build a ``(pixel_values, image_position_ids) -> output`` callable for dedup.

    The encoder calls ``inner.get_image_features(pv, pid)`` on the inner
    Gemma 4 model. Because :func:`langslice_training.embeddings.splice.install_atlas_splice`
    monkey-patches that method, calling it naively would re-enter the
    splice's mask logic. The encoder defends against that by:

      1. Snapshotting + clearing the inner model's sidecar attrs before
         the call so the patched function falls through to the original
         SigLIP path (the "no-sidecar" branch at the top of
         ``spliced_get_image_features``).
      2. Restoring the snapshot on the way out so the actual forward (which
         is what STASHES the sidecars in the first place) sees them
         unchanged.

    Returning ``None`` (no encoder) is acceptable when the splice isn't
    installed — sidecar-build then can't dedup, which is the legacy behaviour.
    """
    import torch  # noqa: PLC0415

    # Re-find the inner Gemma 4 model. We can't import _find_gemma_inner from
    # splice.py because that module is imported lazily in main(); duplicate
    # the lookup here.
    inner: Any | None = None
    for submodule in top_model.modules():
        if (
            hasattr(submodule, "vision_tower")
            and hasattr(submodule, "embed_vision")
            and callable(getattr(submodule, "get_image_features", None))
        ):
            inner = submodule
            break
    if inner is None:
        logger.warning(
            "dedup encoder: could not find inner Gemma 4 model on %r; "
            "within-step dedup disabled.", type(top_model).__name__,
        )
        return None

    # Sidecar attribute names (mirror splice.py constants — duplicated to
    # avoid the lazy-import dance).
    mask_attr = "_atlas_splice_mask"
    flat_attr = "_atlas_splice_cached_flat"
    pc_attr = "_atlas_splice_cached_patch_counts"

    def _encode(pv: torch.Tensor, pid: torch.Tensor) -> Any:
        # Snapshot + clear so the patched get_image_features falls through to
        # original (no-sidecar branch). The actual forward will re-stash the
        # current step's sidecars via the pre-forward-hook regardless.
        saved = (
            getattr(inner, mask_attr, None),
            getattr(inner, flat_attr, None),
            getattr(inner, pc_attr, None),
        )
        setattr(inner, mask_attr, None)
        setattr(inner, flat_attr, None)
        setattr(inner, pc_attr, None)
        try:
            with torch.inference_mode():
                return inner.get_image_features(pv, image_position_ids=pid)
        finally:
            setattr(inner, mask_attr, saved[0])
            setattr(inner, flat_attr, saved[1])
            setattr(inner, pc_attr, saved[2])

    return _encode


def _build_sidecars(
    images: list[Any],
    paths_per_row: list[list[str]],
    cache: Any,
    *,
    pixel_values: Any = None,
    image_position_ids: Any = None,
    encode_fn: Any = None,
):
    """Build the (mask, cached_flat, cached_patch_counts) sidecar tuple.

    Returns ``None`` if either the path/image structure doesn't align or no
    cache hits and no within-step dedup opportunity is present (the splice
    falls back to live SigLIP).

    Parameters
    ----------
    images:
        TRL passes a ``list[list[PIL]]`` (per-row, per-image) to processor.
    paths_per_row:
        Parallel ``list[list[str]]`` from the dataset's ``image_paths`` column.
    cache:
        :class:`langslice_training.embeddings.cache.AtlasEmbeddingCache` (or a chain — anything
        with ``lookup_by_path(path) -> Tensor | None``).
    pixel_values, image_position_ids:
        Processor output tensors. Required only when ``encode_fn`` is set
        (within-step dedup path).
    encode_fn:
        Optional callable ``(pixel_values, image_position_ids) -> output``
        whose ``.last_hidden_state`` is the SigLIP per-image output. When
        provided, duplicate paths within a batch are encoded once and the
        result is reused for every duplicate slot via the cached-flat
        sidecar — the splice forward then skips SigLIP entirely for those
        slots. Saves ``num_generations - 1`` SigLIP calls per repeated
        prompt at GRPO time. Without ``encode_fn`` (or without duplicates)
        the function falls back to disk-cache-only behaviour.

    Within-step dedup (WIN #1)
    --------------------------
    GRPO's group-relative-advantage requires ``num_generations`` rollouts
    of the same prompt. The single-turn-RL row carries one query image
    per row; with N=4 rollouts the same query gets SigLIP-encoded N times
    per generation step. Pre-encoding it once (here) and marking the N-1
    duplicates as cached-from-this-batch saves the SigLIP work in the
    splice forward. The splice forward doesn't need to know whether the
    cached entry came from disk or a fresh in-batch encode — both look
    like a tensor in ``cached_flat`` with a row entry in ``cached_pc``.
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

    # Step 1: per-position disk lookup (None when miss).
    disk_embs: list[Any] = []
    for path in flat_paths:
        emb = cache.lookup_by_path(path) if cache is not None else None
        if emb is not None and emb.dim() > 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        disk_embs.append(emb)

    # Step 2: identify the first occurrence of each path (in flat order). Any
    # later position with the same path is a duplicate.
    first_pos_for_path: dict[str, int] = {}
    for i, path in enumerate(flat_paths):
        first_pos_for_path.setdefault(path, i)

    can_dedup = (
        encode_fn is not None
        and pixel_values is not None
        and image_position_ids is not None
    )

    # Step 3: collect the unique (cache-missed) first-occurrence positions
    # we'd encode if there's any duplicate of them downstream. Encoding a
    # unique that has zero duplicates is wasted work — same number of SigLIP
    # calls either way, plus extra splice machinery — so gate the encode on
    # "this novel first-occurrence has at least one duplicate".
    novel_first_positions: list[int] = []
    if can_dedup:
        # Per-path: total occurrences = how many positions reference it.
        path_count: dict[str, int] = {}
        for path in flat_paths:
            path_count[path] = path_count.get(path, 0) + 1
        for i, path in enumerate(flat_paths):
            if disk_embs[i] is not None:
                continue
            if first_pos_for_path[path] != i:
                continue
            if path_count[path] >= 2:
                novel_first_positions.append(i)

    # Step 4: encode novel firsts in one batched encoder call.
    fresh_embs: dict[int, Any] = {}  # first_position → per-image embedding tensor
    if can_dedup and novel_first_positions:
        idx = torch.tensor(
            novel_first_positions, dtype=torch.long, device=pixel_values.device,
        )
        pv_sub = pixel_values.index_select(0, idx)
        pid_sub = image_position_ids.index_select(0, idx)
        result = encode_fn(pv_sub, pid_sub)
        # Result is the same shape contract as the splice's
        # ``original_get_image_features``: ``last_hidden_state`` of shape
        # ``(sum_real_patches, hidden)``.
        if hasattr(result, "last_hidden_state"):
            lhs = result.last_hidden_state
        elif isinstance(result, tuple):
            lhs = result[0]
        else:
            lhs = result
        # Per-image patch counts come from the same image_position_ids the
        # splice forward uses (padding patches use ``(-1, -1)``).
        real_mask = (pid_sub >= 0).all(dim=-1)
        per_image_pc = real_mask.sum(dim=-1).tolist()
        offset = 0
        for j, first_pos in enumerate(novel_first_positions):
            cnt = int(per_image_pc[j])
            fresh_embs[first_pos] = lhs[offset:offset + cnt].detach()
            offset += cnt
        if offset != int(lhs.shape[0]):
            # Encoder produced more (or fewer) patches than image_position_ids
            # accounts for — strong signal that dedup encoding is misaligned.
            raise RuntimeError(
                f"dedup encode: scattered {offset} patches but encoder "
                f"output had {int(lhs.shape[0])} — image_position_ids and "
                "encoder per-image pooling disagree."
            )

    # Step 5: assemble mask + cached_flat. A position is marked cached when
    # either disk has it OR the path's first occurrence was pre-encoded.
    mask_list: list[bool] = []
    cached_tensors: list[Any] = []
    for i, path in enumerate(flat_paths):
        if disk_embs[i] is not None:
            mask_list.append(True)
            cached_tensors.append(disk_embs[i])
            continue
        first_pos = first_pos_for_path[path]
        emb = fresh_embs.get(first_pos)
        if emb is not None:
            mask_list.append(True)
            cached_tensors.append(emb)
            continue
        mask_list.append(False)

    if not cached_tensors:
        return None

    # Match pixel_values' device when present so the downstream forward-hook
    # mm against ``cached_flat`` doesn't hit a CPU↔CUDA mismatch under
    # Unsloth's grad-offload (the cached disk tensors load to CPU by
    # default; SFT's collator path transfers them through pin-memory but
    # GRPO writes sidecars directly into the processor output).
    target_device = pixel_values.device if pixel_values is not None else None
    if target_device is not None:
        cached_tensors = [t.to(device=target_device) for t in cached_tensors]
    mask = torch.tensor(mask_list, dtype=torch.bool)
    cached_flat = torch.cat(cached_tensors, dim=0)
    cached_pc = torch.tensor([t.shape[0] for t in cached_tensors], dtype=torch.long)
    if target_device is not None:
        mask = mask.to(device=target_device)
        cached_pc = cached_pc.to(device=target_device)
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
            if int(every_n_ticks) <= 0:
                raise ValueError(
                    "every_n_ticks must be positive; pass "
                    "--persist-difficulty-every-n-ticks 0 to disable persist "
                    "via the call site (main() gates on persist_n > 0 before "
                    "constructing this callback)."
                )
            self._manifest_index = manifest_index
            self._every_n_ticks = int(every_n_ticks)
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
