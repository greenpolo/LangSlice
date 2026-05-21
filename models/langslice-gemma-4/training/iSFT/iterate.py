"""Multi-round Expert Iteration SFT driver (wave 2).

End-to-end pipeline (per round k):

    1. Determine the current best adapter:
         - round 0  → ``--base-checkpoint`` (already merged-bf16)
         - round k>0 → output of round k-1 SFT retrain, merged on the fly
    2. Bring up vLLM via docker compose (with a per-round override that
       points ``--model`` at the round's merged-bf16 path).
    3. Sample N prompts × M rollouts, run them against the served model,
       score, filter (best-of-N or threshold-accept), convert ADK events
       to langslice-native trace rows, and append to the iterative corpus.
    4. Build a unioned corpus (base + every iterative round so far) into
       ``<output_dir>/round_<k>_corpus.jsonl``, with all referenced images
       symlinked/copied into a unified tree under ``<output_dir>/``.
       Validate every row through ``sft.dataset._validate_row``.
    5. Tear down vLLM, then invoke ``train_sft.py`` via
       ``docker compose run --rm training`` to retrain a LoRA adapter from
       the immutable distilled-SFT base + the unioned corpus.
    6. Run a slicebench tiny eval against the new adapter; persist
       ``summary.json`` next to the round artifacts.
    7. Persist ``state.json`` after every phase boundary; on restart the
       driver resumes from the next-pending phase.

Run with ``--rounds 1`` to reproduce the wave-1 single-round MVP behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Repo-relative imports.
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from iSFT import filter as filter_mod  # noqa: E402
from iSFT import path_rewriter, trace_format, vllm_lifecycle  # noqa: E402
from iSFT import state as state_mod  # noqa: E402
from iSFT.rollout import RolloutSpec, run_rollouts  # noqa: E402
from iSFT.seen_ledger import SeenLedger, StaleLedgerError  # noqa: E402

from langslice_harness.atlas.core import get_position_range_mm, load_atlas  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m iSFT.iterate",
        description="Expert Iteration SFT - multi-round driver.",
    )
    # Core paths
    p.add_argument("--base-checkpoint", type=Path, required=True,
                   help="Path to merged-bf16 base SFT checkpoint.")
    p.add_argument("--base-corpus", type=Path, required=True,
                   help="Distilled SFT JSONL corpus (immutable across rounds).")
    p.add_argument("--iterative-corpus-dir", type=Path, required=True,
                   help="Where to append round_<k>.jsonl (and stage images).")
    p.add_argument("--allocation-root", type=Path, required=True,
                   help="data/manifest root containing allocations/ + shards/.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Per-run output directory (state.json, round_k_*).")

    # Multi-round chaining
    p.add_argument("--rounds", type=int, default=1,
                   help=(
                       "Number of expert-iteration rounds to chain "
                       "(default 1 = wave-1 single-round behavior)."
                   ))
    p.add_argument("--start-round", type=int, default=0,
                   help="Lowest round index to (re)run; resume picks up from here.")

    # Rollout knobs
    p.add_argument("--rollouts-per-prompt", type=int, default=4)
    p.add_argument("--prompts-per-round", type=int, default=800)
    p.add_argument("--filter-mode", choices=("best-of-n", "threshold-accept", "both"),
                   default="best-of-n")
    p.add_argument("--threshold-pct", type=float, default=0.015)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--media-resolution", default="medium")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--apply-clahe", action="store_true")
    p.add_argument("--max-fetch-calls", type=int, default=2,
                   help="Cap on fetch_atlas turns kept by trim_trace on real-rollout rows.")
    p.add_argument("--max-total-images", type=int, default=8,
                   help=(
                       "Cap on total atlas images per multi-turn row.  Applies to both "
                       "real-rollout rows (via trim_trace, drops whole tool steps) AND "
                       "synthetic multi-turn rows (via _cap_canonical_tool_step_positions, "
                       "drops farthest-from-GT positions while keeping multi-step shape). "
                       "Answer-only synth rows have no atlas images and are unaffected.  "
                       "Default 8 trims the worst-tail ~30%% of synth multi-turn rows whose "
                       "empirical position counts exceed 8 (mean ~8.3 from _empirical.py) "
                       "without disturbing the median row; chosen for PDBS=4 VRAM headroom."
                   ))
    p.add_argument("--planes", nargs="+",
                   default=["coronal", "sagittal", "horizontal"])

    # Persistent training container — train_sft, slicebench eval, and the
    # LoRA->bf16 merge all `docker exec` into this container instead of
    # spinning up `docker compose run --rm` fresh containers each call.
    # Saves the Unsloth + Gemma cold-start (~5-7 min per fresh container)
    # AND makes the editable-install guard in those subprocesses actually
    # skip the reinstall on every call. The container must be running
    # before iterate.py launches.
    p.add_argument("--training-container-name", default="langslice-training-dev",
                   help="Name of the persistent training container. iterate.py runs all "
                        "in-container subprocesses (train_sft, slicebench eval, "
                        "LoRA->bf16 merge) via `docker exec` against this name. "
                        "A pre-flight check at startup fails fast if the container "
                        "isn't running. Pass --skip-container-check to bypass "
                        "the pre-flight (useful for tests).")
    p.add_argument("--skip-container-check", action="store_true",
                   help="Bypass the docker-ps pre-flight that confirms "
                        "--training-container-name is up. Tests + dry-runs use this.")

    # vLLM lifecycle
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--vllm-base-compose", type=Path,
                   default=Path("docker-compose.training.yml"),
                   help="Base docker-compose file containing the vllm service. "
                        "Used for vllm up/down only - train_sft + merge go "
                        "through --training-container-name instead.")
    p.add_argument("--vllm-served-name", default="langslice-ft")
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.85)
    p.add_argument("--vllm-startup-timeout", type=float, default=300.0)
    p.add_argument("--manage-vllm", action="store_true",
                   help="Spawn / tear down vLLM via docker compose between rounds. "
                        "Without this flag, the driver assumes vLLM is already serving "
                        "the right adapter (wave-1 compatibility).")
    p.add_argument("--skip-vllm-check", action="store_true",
                   help="Skip the /v1/models GET probe before each round.")
    p.add_argument("--model-alias", default="langslice-ft")
    p.add_argument("--vllm-lora-mode", action="store_true",
                   help="Skip the per-round LoRA->bf16 merge by serving the SFT base "
                        "with --enable-lora and hot-swapping new adapters via the "
                        "/v1/load_lora_adapter REST endpoint. Each round k registers "
                        "its adapter under model name 'round_<k>' and routes rollouts/"
                        "eval to that name. Defaults to off (existing merge+restart "
                        "flow stays the default for back-compat).")
    p.add_argument("--vllm-lora-max-rank", type=int, default=16,
                   help="--max-lora-rank passed to vLLM when --vllm-lora-mode is set.")
    p.add_argument("--vllm-lora-max-loras", type=int, default=4,
                   help="--max-loras passed to vLLM when --vllm-lora-mode is set; "
                        "old adapters are unloaded as new rounds finish to stay under cap.")
    p.add_argument("--vllm-base-model-path", type=str, default=None,
                   help="Override the in-container path vLLM uses for its --model arg. "
                        "Defaults to the in-container view of --base-checkpoint, which "
                        "lives on the slow Windows-NTFS-via-WSL2 bind mount (~10 min cold "
                        "start). When the langslice-models-fast volume already holds an "
                        "identical copy of the SFT base, set this to '/models/sft-base' "
                        "for ~2-3 min cold start instead. Only affects --vllm-lora-mode.")

    # SFT retrain
    p.add_argument("--sft-config", type=Path,
                   default=Path("models/langslice-gemma-4/training/configs/sft_default.toml"))
    p.add_argument("--sft-initial-adapter", type=Path, default=None,
                   help=(
                       "Path to a PEFT/LoRA adapter to RESUME each round's SFT from "
                       "(passed through to train_sft.py --initial-adapter). When set, "
                       "every round nudges this same anchor adapter forward on its "
                       "round corpus instead of attaching a fresh randomly-initialized "
                       "LoRA. This is the recommended Option D path for expert "
                       "iteration on small per-round corpora - fresh-init on <500 "
                       "examples for ~63 steps catastrophically degrades the model. "
                       "Defaults to None (fresh-init each round; OK only with the "
                       "full distilled corpus + many steps)."
                   ))
    p.add_argument("--skip-retrain", action="store_true",
                   help="Build the unioned corpus and skip SFT retrain + eval (debug).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip slicebench tiny eval (still retrains).")
    p.add_argument("--eval-bench", default="small",
                   choices=["tiny", "small", "large"])
    p.add_argument("--eval-num-generations", type=int, default=1,
                   help="Slicebench --num-generations for the post-round eval. "
                        "Default 1 (single-shot per section, more sections = better "
                        "diversity vs variance tradeoff).")
    p.add_argument("--eval-concurrency", type=int, default=16,
                   help="Slicebench --concurrency for the post-round eval. "
                        "Default 16: vLLM's continuous batcher overlaps requests, "
                        "5-7x speedup vs serial. Drop to 1 if vLLM OOMs under "
                        "concurrent KV-cache pressure.")
    p.add_argument("--distilled-sample-n", type=int, default=None,
                   help="None (default)=use full base corpus; N>0=sample N rows "
                        "from --base-corpus to mix in as anchor data; "
                        "0=exclude the base corpus entirely (fully-synth iSFT "
                        "round; the base SFT corpus has already shaped the anchor "
                        "checkpoint, so re-feeding it is pure re-SFT). "
                        "Use ~100-300 for hackathon-pace iteration with anchor; "
                        "0 once you trust synth to drive improvement alone.")
    p.add_argument("--distilled-sample-seed", type=int, default=0,
                   help="Seed for --distilled-sample-n random sampling. "
                        "Set deterministically for reproducible runs.")
    p.add_argument("--atlas-embedding-cache", type=Path, default=None,
                   help=(
                       "Host-side directory holding precomputed SigLIP atlas embeddings "
                       "(produced by `python -m embeddings.precompute`). When set, this "
                       "path is translated to its in-container equivalent and passed to "
                       "every round's train_sft.py subprocess as --atlas-embedding-cache, "
                       "which installs the forward-hook splice that skips SigLIP for "
                       "cached atlas reference images. Saves ~30-40 min per round on the "
                       "SFT retrain phase at production batch sizes. Cache must be "
                       "bit-exact-verified before use (see embeddings/_verify_cache.py)."
                   ))
    p.add_argument("--query-embedding-cache", type=Path, default=None,
                   help=(
                       "Host-side directory holding precomputed SigLIP query (slice) "
                       "embeddings (produced by `python -m embeddings.precompute_query`). "
                       "Sibling of --atlas-embedding-cache: when set, this path is "
                       "translated to its in-container equivalent and passed to every "
                       "round's train_sft.py subprocess as --query-embedding-cache, which "
                       "splices cached slice embeddings in the same forward-hook so SigLIP "
                       "is skipped for cached per-row slice images. Combine with "
                       "--atlas-embedding-cache for maximum SigLIP-skip coverage."
                   ))
    p.add_argument("--repo-root", type=Path, default=_REPO,
                   help="Host-side repo root (used to build docker compose -f paths).")

    # Misc
    p.add_argument("--no-validate", action="store_true",
                   help="Skip _validate_row check on the unioned corpus (debug).")
    p.add_argument("--run-id", default=None,
                   help="Friendly identifier persisted in state.json.")

    # Synthetic terminal traces (Phase 3).
    p.add_argument(
        "--synthetic-per-round", type=int, default=0,
        help=(
            "If > 0, generate this many synthetic terminal-trace rows per round via "
            "synth_corpus.generate_synthetic_rows and append them to the round's "
            "iterative corpus JSONL alongside real rollouts.  When 0 (default), no "
            "synthetic rows are generated and behavior is byte-identical to the "
            "pre-Phase-3 code."
        ),
    )
    p.add_argument(
        "--multi-turn-floor", type=float, default=0.10,
        help=(
            "Fraction of synthetic rows to render as sft_full (multi-turn with "
            "tool_steps).  The remainder are rendered as answer-only (trace = "
            "submit step only).  Only meaningful when --synthetic-per-round > 0.  "
            "Default 0.10 (10%% multi-turn, 90%% answer-only) - the answer-only "
            "rows directly train the visual -> AP-coordinate mapping; the 10%% "
            "multi-turn floor preserves tool-calling competence at inference."
        ),
    )
    p.add_argument(
        "--synthetic-reasoning-mode",
        choices=("region_dump", "thinking_signature"),
        default="region_dump",
        help=(
            "Reasoning target used for synthetic rows. 'region_dump' preserves "
            "the legacy submit_estimate(reasoning=...) trace shape. "
            "'thinking_signature' emits Gemma thinking rows with compact "
            "BrainGlobe geometry signatures and splits multi-turn traces into "
            "one supervised assistant action per row."
        ),
    )
    p.add_argument(
        "--bucketed-shape-sampler", action="store_true",
        help=(
            "Forward --bucketed-shape-sampler to each round's train_sft.py call. "
            "Required when --synthetic-per-round > 0 with a non-trivial mix of "
            "answer-only and multi-turn rows: produces shape-pure micro-batches "
            "instead of mixing the two row shapes (which have ~5x different "
            "image counts and would otherwise blow up loss variance)."
        ),
    )
    p.add_argument(
        "--clahe-augment-fraction", type=float, default=0.0,
        help=(
            "Forward --clahe-augment-fraction to each round's train_sft.py call. "
            "Fraction in [0.0, 1.0] of training rows that get production CLAHE "
            "applied to their slice image at __getitem__ time. CLAHE'd rows "
            "bypass the query embedding cache for their slice image so the "
            "live SigLIP forward sees the augmented pixels. Default 0.0 (off)."
        ),
    )

    # Curriculum (Phase C — see curriculum.weights / curriculum.cli).
    p.add_argument(
        "--curriculum-weights-dir", type=Path, default=None,
        help=(
            "Optional dir holding round_<k>_weights.json. When set, the "
            "rollout sampler weights prompts toward weak bins (using "
            "the previous round's weights file) and after each round's "
            "slicebench eval the driver refreshes the next round's "
            "weights file from per_coord_bin MAE."
        ),
    )
    p.add_argument(
        "--curriculum-alpha", type=float, default=1.0,
        help="Sharpness of the weight update; w proportional to (mae_bin/baseline)**alpha.",
    )
    p.add_argument(
        "--curriculum-max-weight-change", type=float, default=3.0,
        help="Max multiplicative change vs prev_weights per round.",
    )
    p.add_argument(
        "--curriculum-floor-fraction", type=float, default=0.1,
        help="Lower bound on any weight as a fraction of baseline (1.0).",
    )
    p.add_argument(
        "--curriculum-smoothing", type=float, default=0.5,
        help="EMA smoothing factor against prev_weights before the cap.",
    )

    # Seen-section ledger (Phase 4).
    p.add_argument(
        "--use-seen-ledger", action="store_true", default=False,
        help=(
            "Enable the bin-aware sampler v2 with a persistent seen-section "
            "ledger.  Sections are never re-sampled once they appear in the "
            "ledger.  Default False (byte-identical to pre-Phase-4 behavior)."
        ),
    )
    p.add_argument(
        "--ledger-path", type=Path,
        default=None,
        help=(
            "Path to the seen-section ledger JSONL.  Defaults to "
            "<output-dir>/seen_ledger.jsonl.  Only used when "
            "--use-seen-ledger is set."
        ),
    )
    p.add_argument(
        "--reset-ledger", action="store_true", default=False,
        help=(
            "Delete the ledger file before loading on startup.  Use when "
            "the manifest has changed and you want to start fresh without "
            "manually deleting the file."
        ),
    )

    # Adaptive accept/reject threshold (Phase 5).
    p.add_argument(
        "--adaptive-accept", action="store_true", default=False,
        help=(
            "Enable adaptive accept/reject threshold for the filter phase.  "
            "When set, each round's filter uses the quantile_cutoff_mm from a "
            "per-run error buffer instead of the static --threshold-pct.  "
            "During warmup (fewer than --adaptive-accept-warmup-n observations) "
            "all rollouts pass (cutoff = inf).  Default False; behavior is "
            "byte-identical to pre-Phase-5 when this flag is absent."
        ),
    )
    p.add_argument(
        "--adaptive-accept-quantile", type=float, default=0.95,
        help=(
            "Which percentile of recent abs_err_mm to use as the accept "
            "threshold when --adaptive-accept is set.  Default 0.95."
        ),
    )
    p.add_argument(
        "--adaptive-accept-warmup-n", type=int, default=50,
        help=(
            "Minimum number of error observations required before the adaptive "
            "threshold tightens.  Below this count all rollouts pass.  "
            "Default 50."
        ),
    )
    args = p.parse_args(argv)
    if not 0.0 <= args.clahe_augment_fraction <= 1.0:
        p.error(
            f"--clahe-augment-fraction must be in [0.0, 1.0]; got "
            f"{args.clahe_augment_fraction}"
        )
    return args


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _check_vllm_alive(base_url: str) -> bool:
    """Probe ``/v1/models``; returns True on 200."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.warning("vLLM probe of %s failed: %s", url, exc)
        return False


def _plane_extent_mm(atlas_name: str, plane: str) -> float:
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)  # type: ignore[arg-type]
    if plane == "sagittal":
        pos_hi = pos_hi / 2.0
    return float(pos_hi - pos_lo)


def _sample_specs(
    allocation: list[Any],
    *,
    n_prompts: int,
    rollouts_per_prompt: int,
    seed: int,
    section_weights: dict[str, float] | None = None,
) -> list[RolloutSpec]:
    rng = random.Random(seed)
    pool = list(allocation)
    if section_weights:
        # Curriculum-biased sampling: prompts in weak bins get up-weighted.
        # Sections without a weight in the map fall back to 1.0.
        weights = [float(section_weights.get(ex.section_id, 1.0)) for ex in pool]
        # Edge case — all-zero weights would crash random.choices; fall back
        # to uniform shuffle.
        if any(w > 0 for w in weights):
            n = min(n_prompts, len(pool))
            # random.choices samples WITH replacement; dedup so we don't run
            # the same (section_id) repeatedly within one round.
            seen: set[str] = set()
            picked: list[Any] = []
            attempts = 0
            cap = max(50 * n, 10_000)
            while len(picked) < n and attempts < cap:
                ex = rng.choices(pool, weights=weights, k=1)[0]
                if ex.section_id in seen:
                    attempts += 1
                    continue
                seen.add(ex.section_id)
                picked.append(ex)
                attempts += 1
        else:
            rng.shuffle(pool)
            picked = pool[: min(n_prompts, len(pool))]
    else:
        rng.shuffle(pool)
        picked = pool[: min(n_prompts, len(pool))]
    extent_cache: dict[tuple[str, str], float] = {}
    specs: list[RolloutSpec] = []
    for ex in picked:
        key = (ex.atlas_name, ex.plane)
        if key not in extent_cache:
            extent_cache[key] = _plane_extent_mm(ex.atlas_name, ex.plane)
        extent = extent_cache[key]
        for gen_idx in range(rollouts_per_prompt):
            specs.append(RolloutSpec(
                section_id=ex.section_id,
                subject_id=ex.subject_id,
                image_path=ex.image_path,
                atlas_name=ex.atlas_name,
                plane=ex.plane,
                truth_mm=ex.ap_mm,
                plane_extent_mm=extent,
                generation_idx=gen_idx,
            ))
    return specs


def _sample_specs_v2(
    *,
    allocation: list[Any],
    bin_weights: dict[str, float],
    section_bins: dict[str, tuple[str, str] | int],
    seen_ledger: SeenLedger,
    n_target: int,
    rng: random.Random,
) -> list[Any]:
    """Breadth-first bin-aware sampler v2.

    Enumerate bins sorted by weight descending (high weight = weak bin = sampled
    first).  For each bin, sample uniformly from sections in that bin that have
    not yet appeared in ``seen_ledger``.  Advance to the next bin when the
    current bin is exhausted.  If all bins exhaust before ``n_target`` is
    reached, wrap to a second pass (ledger preserved — the caller decides
    whether to reset).

    Parameters
    ----------
    allocation:
        Full list of ``SingleSliceExample`` objects (the RLVR pool).
    bin_weights:
        ``{bin_label: weight}`` mapping used to determine priority order.
        Any string key is accepted; examples whose bin is absent fall back
        to weight 1.0.
    section_bins:
        ``{section_id: (plane, bin_label) | int}`` — the bin for each section.
    seen_ledger:
        The persistent seen-section ledger.  Used to skip already-seen sections
        AND to mark returned sections as seen before returning.
    n_target:
        Number of ``AllocationExample`` objects to return.
    rng:
        Seeded ``random.Random`` instance for reproducibility.

    Returns
    -------
    list of ``AllocationExample`` objects, length <= ``n_target``.
    """
    # Build a per-bin pool: bin_label -> list[example]
    # Resolve each section to a bin_label string for grouping.
    _LABELS = ("q1", "q2", "q3", "q4", "q5")

    def _bin_label(section_id: str) -> str:
        b = section_bins.get(section_id)
        if b is None:
            return "_unknown"
        if isinstance(b, int):
            if 0 <= b < len(_LABELS):
                return _LABELS[b]
            return f"q{b + 1}"
        # tuple (plane, label) — use the label part only for grouping
        return str(b[1]) if len(b) >= 2 else "_unknown"

    by_bin: dict[str, list[Any]] = {}
    for ex in allocation:
        label = _bin_label(ex.section_id)
        by_bin.setdefault(label, []).append(ex)

    # Sort bins by weight descending (highest weight first = weakest bin first).
    all_labels = sorted(
        by_bin.keys(),
        key=lambda lb: bin_weights.get(lb, 1.0),
        reverse=True,
    )

    collected: list[Any] = []
    # Track IDs already collected within this call to prevent duplicates when
    # the second pass runs (caller marks sections seen externally, not inside
    # this function, so the ledger alone isn't sufficient for within-call dedup).
    _this_call_ids: set[str] = set()

    def _is_available(section_id: str) -> bool:
        return (
            not seen_ledger.is_seen(section_id)
            and section_id not in _this_call_ids
        )

    def _sample_from_pass(labels: list[str]) -> None:
        """One pass: drain bins in priority order, each bin fully before advancing."""
        for lb in labels:
            if len(collected) >= n_target:
                break
            unseen = [ex for ex in by_bin[lb] if _is_available(ex.section_id)]
            if not unseen:
                continue
            # Shuffle for uniform sampling within the bin.
            rng.shuffle(unseen)
            for ex in unseen:
                if len(collected) >= n_target:
                    break
                if _is_available(ex.section_id):
                    collected.append(ex)
                    _this_call_ids.add(ex.section_id)

    # First pass: bins in weight-descending order.
    _sample_from_pass(all_labels)

    if len(collected) < n_target:
        # Second pass (ledger preserved — caller decides about reset).
        # This handles the case where first-pass exhausted all bins before
        # reaching n_target.
        second_pass_pool = [
            ex for lb in all_labels for ex in by_bin[lb]
            if _is_available(ex.section_id)
        ]
        if second_pass_pool:
            rng.shuffle(second_pass_pool)
            for ex in second_pass_pool:
                if len(collected) >= n_target:
                    break
                if _is_available(ex.section_id):
                    collected.append(ex)
                    _this_call_ids.add(ex.section_id)
        else:
            # Truly all sections seen — log and return what we have.
            logger.warning(
                "_sample_specs_v2: all %d sections in ledger; "
                "returning %d (requested %d).",
                seen_ledger.seen_count(), len(collected), n_target,
            )

    return collected


def _score_rollouts(rollouts: list[Any]) -> list[dict[str, Any]]:
    from slicebench.score import score_predictions  # noqa: PLC0415

    predictions: list[dict[str, Any]] = []
    for r in rollouts:
        predicted = (
            float(r.submitted_position_mm)
            if r.error is None and r.submitted_position_mm is not None
            else None
        )
        predictions.append({
            "section_id": r.spec.section_id,
            "subject_id": r.spec.subject_id,
            "atlas": r.spec.atlas_name,
            "plane": r.spec.plane,
            "truth_mm": float(r.spec.truth_mm),
            "predicted_mm": predicted,
            "plane_extent_mm": float(r.spec.plane_extent_mm),
            "generation_idx": r.spec.generation_idx,
            "rollout_error": r.error,
        })
    return score_predictions(predictions)


def _kept_rollouts_for_filter(
    scored: list[dict[str, Any]],
    *,
    filter_mode: str,
    threshold_pct: float,
    adaptive_buffer: Any | None = None,
    adaptive_quantile: float = 0.95,
    adaptive_warmup_n: int = 50,
) -> list[dict[str, Any]]:
    """Return the subset of scored rollouts to keep for this round.

    When ``adaptive_buffer`` is not None (i.e. ``--adaptive-accept`` is set),
    the accept threshold for the ``threshold-accept`` and ``both`` modes is
    derived from the quantile of recent absolute errors in the buffer.  During
    warmup (fewer than ``adaptive_warmup_n`` observations) all rollouts pass.
    The ``best-of-n`` path is unaffected — it selects the single best rollout
    per section regardless of absolute error.

    When ``adaptive_buffer`` is None (default), behavior is byte-identical to
    the pre-Phase-5 code path.
    """
    scored = [r for r in scored if r.get("predicted_mm") is not None]

    if adaptive_buffer is not None:
        # Adaptive threshold: use quantile_cutoff_mm from the shared buffer.
        from adaptive.schedule import AdaptiveSchedule  # noqa: PLC0415
        _sched = AdaptiveSchedule(
            warmup_min_observations=adaptive_warmup_n,
        )
        cutoff_mm = _sched.quantile_cutoff_mm(adaptive_buffer, quantile=adaptive_quantile)

        def _adaptive_accept(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if cutoff_mm == float("inf"):
                return rollouts
            return [r for r in rollouts if float(r["abs_err_mm"]) <= cutoff_mm]

        if filter_mode == "best-of-n":
            # best-of-n is error-rank-based; adaptive cutoff doesn't add value here.
            return filter_mod.best_of_n(scored)
        if filter_mode == "threshold-accept":
            return _adaptive_accept(scored)
        # "both": take best-of-n union adaptive-accept
        bo = filter_mod.best_of_n(scored)
        th = _adaptive_accept(scored)
        seen: set[tuple[str, int]] = set()
        out: list[dict[str, Any]] = []
        for row in bo + th:
            key = (row["section_id"], int(row["generation_idx"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    # --- static path (default, byte-identical to pre-Phase-5) ---
    if filter_mode == "best-of-n":
        return filter_mod.best_of_n(scored)
    if filter_mode == "threshold-accept":
        return filter_mod.threshold_accept(scored, threshold_pct=threshold_pct)
    bo = filter_mod.best_of_n(scored)
    th = filter_mod.threshold_accept(scored, threshold_pct=threshold_pct)
    seen_s: set[tuple[str, int]] = set()
    out_s: list[dict[str, Any]] = []
    for row in bo + th:
        key = (row["section_id"], int(row["generation_idx"]))
        if key in seen_s:
            continue
        seen_s.add(key)
        out_s.append(row)
    return out_s


def _stage_query_for_row(
    *,
    image_path: Path,
    section_id: str,
    atlas_name: str,
    plane: str,
    apply_clahe: bool,
    out_dir: Path,
    repo_root: Path,
) -> str:
    try:
        rel = image_path.resolve().relative_to(repo_root)
    except ValueError:
        rel = image_path
    return trace_format.stage_query_image(
        str(rel).replace("\\", "/"),
        out_dir,
        section_id,
        atlas_name=atlas_name,
        plane=plane,
        apply_clahe=apply_clahe,
    )


def _validate_corpus(unioned_jsonl: Path) -> int:
    """Walk every row through ``sft.dataset._validate_row``."""
    sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))
    from sft.dataset import _validate_row  # noqa: PLC0415

    n = 0
    with unioned_jsonl.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            _validate_row(row, lineno, unioned_jsonl.parent)
            n += 1
    return n


# ──────────────────────────────────────────────────────────────────────────
# Per-phase implementations
# ──────────────────────────────────────────────────────────────────────────

def _phase_rollouts(
    *,
    args: argparse.Namespace,
    round_idx: int,
    artifacts_root: Path,
    model_str: str,
) -> tuple[list[Any], list[Any]]:
    """Sample prompts + run rollouts. Returns (specs, rollouts)."""
    from rlvr.dataset import load_rlvr_allocation  # noqa: PLC0415
    allocation = load_rlvr_allocation(
        args.allocation_root, repo_root=_REPO, planes=tuple(args.planes),
    )
    if not allocation:
        raise RuntimeError(
            "Allocation empty; check --allocation-root and --planes."
        )

    # Curriculum: bias prompt sampling toward weak bins using the previous
    # round's weights file (computed from the previous round's slicebench
    # summary by ``_phase_curriculum_update`` below). Round 0 has no
    # prior summary so we fall back to uniform sampling.
    section_weights: dict[str, float] | None = None
    if args.curriculum_weights_dir is not None and round_idx > 0:
        weights_path = args.curriculum_weights_dir / f"round_{round_idx - 1}_weights.json"
        if weights_path.is_file():
            from curriculum.weights import read_weights_json  # noqa: PLC0415

            section_weights = read_weights_json(weights_path)
            logger.info(
                "[round %d] curriculum: biasing prompts via %s (n=%d)",
                round_idx, weights_path, len(section_weights),
            )
        else:
            logger.info(
                "[round %d] curriculum: %s missing; uniform sampling.",
                round_idx, weights_path,
            )

    # Vary the seed per round so we don't keep replaying the same prompts.
    seen_ledger: SeenLedger | None = getattr(args, "_seen_ledger", None)
    if seen_ledger is not None and args.use_seen_ledger:
        # v2: bin-aware sampler with persistent ledger.
        rng_v2 = random.Random(args.seed + 1000 * round_idx)
        # Build bin_weights and section_bins from the curriculum weights file
        # (if available), otherwise use uniform weights.
        bin_weights_v2: dict[str, float] = {}
        section_bins_v2: dict[str, tuple[str, str] | int] = {}
        if section_weights:
            # section_weights is {section_id: weight}; map to per-section bins
            # using the curriculum bins module.
            try:
                from curriculum.bins import compute_section_bins  # noqa: PLC0415
                sb = compute_section_bins(allocation)
                _LABELS_V2 = ("q1", "q2", "q3", "q4", "q5")
                plane_by_sid = {ex.section_id: str(ex.plane) for ex in allocation}
                for sid, qi in sb.items():
                    plane = plane_by_sid.get(sid)
                    if plane and 0 <= qi < len(_LABELS_V2):
                        section_bins_v2[sid] = (plane, _LABELS_V2[qi])
                    else:
                        section_bins_v2[sid] = qi
                # Aggregate per-bin weights: average section_weights per (plane, label).
                bin_accum: dict[str, list[float]] = {}
                for sid, bkey in section_bins_v2.items():
                    if isinstance(bkey, tuple):
                        lb = str(bkey[1])
                    else:
                        lb = f"q{bkey + 1}" if isinstance(bkey, int) else "_unknown"
                    w = section_weights.get(sid, 1.0)
                    bin_accum.setdefault(lb, []).append(w)
                bin_weights_v2 = {lb: sum(ws) / len(ws) for lb, ws in bin_accum.items()}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[round %d] v2 sampler: could not compute bin_weights: %s; "
                    "falling back to uniform bin_weights.",
                    round_idx, exc,
                )
        picked_v2 = _sample_specs_v2(
            allocation=allocation,
            bin_weights=bin_weights_v2,
            section_bins=section_bins_v2,
            seen_ledger=seen_ledger,
            n_target=args.prompts_per_round,
            rng=rng_v2,
        )
        # Mark every picked section as seen *at sample time*, not only after
        # acceptance.  Without this, a section whose rollout fails the quality
        # filter would be eligible for re-selection next round — undermining
        # the "breadth over depth" guarantee of the ledger.  The append-time
        # mark (line ~893) is idempotent on already-seen ids, so the duplicate
        # call is just an extra audit-trail entry per accepted section.
        for ex in picked_v2:
            seen_ledger.mark_seen(ex.section_id, round_idx, source="real_rollout")
        # Expand picked examples into RolloutSpecs (mirrors _sample_specs logic).
        extent_cache_v2: dict[tuple[str, str], float] = {}
        specs: list[RolloutSpec] = []
        for ex in picked_v2:
            key = (ex.atlas_name, ex.plane)
            if key not in extent_cache_v2:
                extent_cache_v2[key] = _plane_extent_mm(ex.atlas_name, ex.plane)
            extent = extent_cache_v2[key]
            for gen_idx in range(args.rollouts_per_prompt):
                specs.append(RolloutSpec(
                    section_id=ex.section_id,
                    subject_id=ex.subject_id,
                    image_path=ex.image_path,
                    atlas_name=ex.atlas_name,
                    plane=ex.plane,
                    truth_mm=ex.ap_mm,
                    plane_extent_mm=extent,
                    generation_idx=gen_idx,
                ))
    else:
        specs = _sample_specs(
            allocation,
            n_prompts=args.prompts_per_round,
            rollouts_per_prompt=args.rollouts_per_prompt,
            seed=args.seed + 1000 * round_idx,
            section_weights=section_weights,
        )
    logger.info(
        "[round %d] sampled %d prompts × %d rollouts = %d specs",
        round_idx, len({s.section_id for s in specs}),
        args.rollouts_per_prompt, len(specs),
    )
    rollouts = run_rollouts(
        specs,
        artifacts_root=artifacts_root,
        model=model_str,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        max_retries=args.max_retries,
        apply_clahe=args.apply_clahe,
        media_resolution=args.media_resolution,
        concurrency=args.concurrency,
    )
    return specs, rollouts


def _phase_append_iterative(
    *,
    args: argparse.Namespace,
    round_idx: int,
    rollouts: list[Any],
    kept: list[dict[str, Any]],
    model_str: str,
) -> tuple[Path, dict[str, int]]:
    """Convert kept rollouts → SFT rows, append to iterative_dir/round_k.jsonl.

    Returns (jsonl path, stats dict).
    """
    iterative_jsonl = args.iterative_corpus_dir / f"round_{round_idx}.jsonl"
    iterative_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rollout_by_key: dict[tuple[str, int], Any] = {
        (r.spec.section_id, r.spec.generation_idx): r for r in rollouts
    }

    # Dedup-on-resume: read existing rows out of the round's jsonl so an
    # interrupted append doesn't double-write any (subject_id, gen_idx).
    existing_keys: set[tuple[str, int]] = set()
    if iterative_jsonl.exists():
        with iterative_jsonl.open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                q = row.get("quality") or {}
                gen_idx = int(q.get("generation_idx", -1))
                existing_keys.add((row["subject_id"], gen_idx))

    n_kept = 0
    n_dropped_trace = 0
    n_dropped_dup = 0
    rows_to_write: list[dict[str, Any]] = []
    for scored_row in kept:
        key = (scored_row["section_id"], int(scored_row["generation_idx"]))
        rollout = rollout_by_key.get(key)
        if rollout is None or rollout.error is not None:
            n_dropped_trace += 1
            continue
        try:
            query_rel = _stage_query_for_row(
                image_path=rollout.spec.image_path,
                section_id=rollout.spec.section_id,
                atlas_name=rollout.spec.atlas_name,
                plane=rollout.spec.plane,
                apply_clahe=args.apply_clahe,
                out_dir=args.iterative_corpus_dir,
                repo_root=_REPO,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "[round %d] skipping %s gen=%d: query stage failed: %s",
                round_idx, rollout.spec.section_id,
                rollout.spec.generation_idx, exc,
            )
            n_dropped_trace += 1
            continue
        row = trace_format.build_row(
            section_id=rollout.spec.section_id,
            subject_id=rollout.spec.subject_id,
            atlas_name=rollout.spec.atlas_name,
            plane=rollout.spec.plane,
            query_image_rel=query_rel,
            events=rollout.events,
            submitted_position_mm=float(scored_row["predicted_mm"]),
            sub_canonical_mm=float(scored_row["predicted_mm"]),
            abs_err_mm=float(scored_row["abs_err_mm"]),
            model=model_str,
            iteration_round=round_idx,
            out_dir=args.iterative_corpus_dir,
            run_dir=rollout.run_dir,
            apply_clahe=args.apply_clahe,
            max_fetch_calls=args.max_fetch_calls,
            max_total_images=args.max_total_images,
        )
        if row is None:
            n_dropped_trace += 1
            continue
        row["quality"]["generation_idx"] = int(rollout.spec.generation_idx)
        dedup_key = (row["subject_id"], int(rollout.spec.generation_idx))
        if dedup_key in existing_keys:
            n_dropped_dup += 1
            continue
        existing_keys.add(dedup_key)
        rows_to_write.append(row)

    seen_ledger: SeenLedger | None = getattr(args, "_seen_ledger", None)
    with iterative_jsonl.open("a", encoding="utf-8") as fh:
        for row in rows_to_write:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
            if seen_ledger is not None and getattr(args, "use_seen_ledger", False):
                sid = row.get("quality", {}).get("section_id", "")
                if sid:
                    seen_ledger.mark_seen(sid, round_idx, source="real_rollout")
    logger.info(
        "[round %d] wrote %d rows → %s (dropped: bad_trace=%d, dup=%d)",
        round_idx, n_kept, iterative_jsonl, n_dropped_trace, n_dropped_dup,
    )
    return iterative_jsonl, {
        "n_kept": n_kept,
        "n_dropped_trace": n_dropped_trace,
        "n_dropped_dup": n_dropped_dup,
    }


def _count_synthetic_rows(jsonl_path: Path) -> int:
    """Count rows in ``jsonl_path`` whose ``quality.source`` is
    ``"synthetic_terminal"``.  Returns 0 for missing files (round just
    started, no rows yet).  Used by ``_phase_synthetic`` to detect
    crash-resume mid-round and avoid duplicate synthetic-row writes.
    """
    if not jsonl_path.is_file():
        return 0
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("quality", {}).get("source") == "synthetic_terminal":
                n += 1
    return n


def _read_synthetic_section_ids(jsonl_path: Path) -> list[str]:
    """Return ``section_id`` of every ``synthetic_terminal`` row in the file.

    Used by ``_phase_synthetic`` to retroactively mark seen on crash-resume:
    rows were written to disk before the prior attempt could call
    ``seen_ledger.mark_seen``, so we replay the mark-seen step from the file.
    """
    if not jsonl_path.is_file():
        return []
    out: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("quality", {}).get("source") != "synthetic_terminal":
                continue
            sid = row.get("section_id") or row.get("quality", {}).get("section_id")
            if sid:
                out.append(str(sid))
    return out


def _phase_synthetic(
    *,
    args: argparse.Namespace,
    round_idx: int,
    iterative_jsonl: Path,
) -> dict[str, int]:
    """Generate synthetic terminal-trace rows and append to the iterative JSONL.

    Called immediately after ``_phase_append_iterative`` when
    ``--synthetic-per-round > 0``.  When the flag is 0 this function is never
    called, keeping the default code path byte-identical.

    Returns a stats dict with keys ``n_synth_written``.
    """
    from iSFT.synth_corpus import (  # noqa: PLC0415
        SectionSpec,
        generate_synthetic_rows,
        load_specs_from_allocation,
        weighted_sample_without_replacement,
        write_synthetic_jsonl,
    )

    seen_ledger: SeenLedger | None = getattr(args, "_seen_ledger", None)
    ledger_active = (
        seen_ledger is not None and getattr(args, "use_seen_ledger", False)
    )

    # Resume guard: if the iterative JSONL already contains synthetic_terminal
    # rows for this round, we crashed between _phase_append_iterative and
    # _phase_synthetic on a prior attempt — the "appended" phase boundary
    # only fires after BOTH complete (see _run_round).  Re-running synth
    # generation would produce duplicate rows because write_synthetic_jsonl
    # is an unconditional append.  Count any existing synthetic rows and
    # skip the phase if any are present.
    existing_synth = _count_synthetic_rows(iterative_jsonl)
    if existing_synth > 0:
        # Crash-resume: rows were written to disk but the prior attempt may
        # have crashed before the mark-seen loop ran.  Replay mark-seen from
        # the file so the ledger reflects what's actually on disk.  Idempotent.
        if ledger_active:
            assert seen_ledger is not None  # ledger_active implies non-None
            recovered = _read_synthetic_section_ids(iterative_jsonl)
            already_seen = sum(1 for sid in recovered if seen_ledger.is_seen(sid))
            for sid in recovered:
                seen_ledger.mark_seen(sid, round_idx, source="synthetic")
            logger.info(
                "[round %d] synth: resume-replayed mark-seen for %d section_ids "
                "(%d already in ledger).",
                round_idx, len(recovered), already_seen,
            )
        logger.info(
            "[round %d] synth: %d existing synthetic_terminal rows in %s "
            "(crash-resume); skipping synthetic generation.",
            round_idx, existing_synth, iterative_jsonl,
        )
        return {"n_synth_written": existing_synth, "n_synth_skipped_resume": existing_synth}

    n = args.synthetic_per_round
    rng = random.Random(args.seed + 7777 + 1000 * round_idx)

    # Curriculum: bias synth section selection toward weak bins using prior
    # round's weights file.  Round 0 has no prior weights — uniform sampling.
    # Mirrors the rollout sampler's curriculum path so synth and rollouts
    # bias to the same bin distribution within a round.
    section_weights: dict[str, float] | None = None
    if args.curriculum_weights_dir is not None and round_idx > 0:
        weights_path = (
            args.curriculum_weights_dir / f"round_{round_idx - 1}_weights.json"
        )
        if weights_path.is_file():
            try:
                from curriculum.weights import read_weights_json  # noqa: PLC0415
                section_weights = read_weights_json(weights_path)
                logger.info(
                    "[round %d] synth: biasing section selection via %s (n=%d)",
                    round_idx, weights_path, len(section_weights),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[round %d] synth: curriculum load failed: %s; uniform.",
                    round_idx, exc,
                )

    # Seen-ledger filter: keep across-round coverage strict by excluding
    # sections already picked in prior rounds (rollouts or synth).
    seen_ids: set[str] | None = None
    if ledger_active:
        assert seen_ledger is not None
        seen_ids = set(seen_ledger.seen)

    # Load the *full* eligible pool per plane (n=None disables per-plane
    # sampling).  Per-plane weighted sampling would let an over-represented
    # plane crowd weak-bin sections from an under-represented plane — we
    # sample once across the combined pool below so curriculum bias is
    # applied across the union.
    specs: list[SectionSpec] = []
    for plane in args.planes:
        try:
            plane_specs = load_specs_from_allocation(
                args.allocation_root, plane, None, rng, repo_root=_REPO,
                seen_ids=seen_ids,
                section_weights=None,
            )
            specs.extend(plane_specs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[round %d] synth: load_specs_from_allocation failed for plane=%s: %s",
                round_idx, plane, exc,
            )

    if not specs:
        logger.warning(
            "[round %d] synth: no specs loaded; skipping synthetic row generation.",
            round_idx,
        )
        return {"n_synth_written": 0}

    # Single combined sample across the multi-plane pool.  When curriculum
    # weights are present this is the *only* place they are applied to the
    # synth pool — no per-plane re-sampling above this line.
    specs = weighted_sample_without_replacement(specs, n, rng, section_weights)

    rows = generate_synthetic_rows(
        specs,
        args.multi_turn_floor,
        rng,
        max_total_images=args.max_total_images,
        synthetic_reasoning_mode=getattr(args, "synthetic_reasoning_mode", "region_dump"),
    )

    write_synthetic_jsonl(rows, iterative_jsonl)

    # Mark synthetic sections as seen in the ledger (Phase 4).
    if ledger_active:
        assert seen_ledger is not None
        for spec in specs:
            seen_ledger.mark_seen(spec.section_id, round_idx, source="synthetic")

    logger.info(
        "[round %d] synth: wrote %d synthetic rows → %s",
        round_idx, len(rows), iterative_jsonl,
    )
    return {"n_synth_written": len(rows)}


def _phase_union(
    *,
    args: argparse.Namespace,
    round_idx: int,
) -> tuple[Path, dict[str, int]]:
    """Build the unioned corpus + unify image trees + validate."""
    unioned_jsonl = args.output_dir / f"round_{round_idx}_corpus.jsonl"
    iterative_jsonls = sorted(args.iterative_corpus_dir.glob("round_*.jsonl"))
    # Only include rounds 0..round_idx (don't pull in future rounds if a
    # caller resumes with --start-round in the middle).
    iterative_jsonls = [
        p for p in iterative_jsonls
        if _round_from_filename(p.name) is not None
        and _round_from_filename(p.name) <= round_idx  # type: ignore[operator]
    ]
    stats = path_rewriter.build_unified_corpus(
        base_corpus=args.base_corpus,
        iterative_corpus_dir=args.iterative_corpus_dir,
        iterative_jsonls=iterative_jsonls,
        output_dir=args.output_dir,
        output_jsonl=unioned_jsonl,
        extra_image_roots=[_REPO],
        base_sample_n=args.distilled_sample_n,
        base_sample_seed=args.distilled_sample_seed,
    )
    if not args.no_validate:
        try:
            n_validated = _validate_corpus(unioned_jsonl)
            logger.info(
                "[round %d] validated %d rows in %s",
                round_idx, n_validated, unioned_jsonl,
            )
        except Exception as exc:  # noqa: BLE001 — surface validation issue
            logger.error(
                "[round %d] validation failed: %s; inspect %s",
                round_idx, exc, unioned_jsonl,
            )
            raise
    return unioned_jsonl, stats


def _round_from_filename(name: str) -> int | None:
    """Parse round_<k>.jsonl → k."""
    if not name.startswith("round_") or not name.endswith(".jsonl"):
        return None
    middle = name[len("round_"):-len(".jsonl")]
    if "_" in middle:
        middle = middle.split("_", 1)[0]
    try:
        return int(middle)
    except ValueError:
        return None


def _phase_train(
    *,
    args: argparse.Namespace,
    round_idx: int,
    unioned_jsonl: Path,
    output_adapter_dir: Path,
) -> Path:
    """Invoke train_sft.py inside the training container, return adapter path."""
    repo_root = args.repo_root
    in_container_corpus = _to_container_path(unioned_jsonl, repo_root)
    in_container_out = _to_container_path(output_adapter_dir, repo_root)
    in_container_config = _to_container_path(args.sft_config, repo_root)

    # Curriculum: pass the previous round's weights file (if available)
    # so train_sft.py wraps the dataset in a CurriculumSFTTrainer with
    # WeightedRandomSampler. Round 0 falls through with uniform weights.
    curriculum_arg = ""
    if args.curriculum_weights_dir is not None and round_idx > 0:
        prev_weights = (
            args.curriculum_weights_dir / f"round_{round_idx - 1}_weights.json"
        )
        if prev_weights.is_file():
            in_container_weights = _to_container_path(prev_weights, repo_root)
            curriculum_arg = f"--curriculum-weights {in_container_weights} "

    # Initial-adapter: each round resumes from the SAME anchor adapter
    # (the original SFT adapter), nudging it forward on this round's
    # corpus. This is the "Option D" path that prevents catastrophic
    # forgetting from re-initializing a fresh LoRA on a small corpus.
    # Round k>0 still starts from the SFT anchor, NOT the prior round's
    # adapter — the iterative corpus accumulates instead.
    initial_adapter_arg = ""
    if args.sft_initial_adapter is not None:
        in_container_adapter = _to_container_path(args.sft_initial_adapter, repo_root)
        initial_adapter_arg = f"--initial-adapter {in_container_adapter} "

    # Atlas-embedding splice: skip SigLIP for cached atlas reference images.
    # Saves ~30-40 min/round at production batch sizes. Path is host-side at
    # CLI parse time; translate to in-container view before passing to
    # train_sft.py (which runs inside the container).
    atlas_cache_arg = ""
    if args.atlas_embedding_cache is not None:
        in_container_cache = _to_container_path(args.atlas_embedding_cache, repo_root)
        atlas_cache_arg = f"--atlas-embedding-cache {in_container_cache} "

    # Query-embedding splice: parallel mechanism for cached per-row slice
    # embeddings. Combine with --atlas-embedding-cache for maximum SigLIP-skip
    # coverage. Path translation mirrors the atlas case.
    query_cache_arg = ""
    if args.query_embedding_cache is not None:
        in_container_query_cache = _to_container_path(
            args.query_embedding_cache, repo_root,
        )
        query_cache_arg = f"--query-embedding-cache {in_container_query_cache} "

    bucketed_arg = "--bucketed-shape-sampler " if args.bucketed_shape_sampler else ""

    clahe_aug_arg = ""
    if float(getattr(args, "clahe_augment_fraction", 0.0)) > 0.0:
        clahe_aug_arg = (
            f"--clahe-augment-fraction {float(args.clahe_augment_fraction)} "
        )

    train_shell_cmd = (
        "set -euo pipefail && "
        # Avoid the VRAM-fragmentation step cliff at ~step 500. See
        # project memory feedback_sft_io_slowdown_epoch2: this is the
        # canonical fix for the SFT-1716 plateau. Cheap on shorter runs,
        # mandatory on overnight ones.
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        # Disable Unsloth's vLLM-in-standby integration. It is designed
        # for trainers that share the GPU with a live vLLM; iSFT shuts
        # vLLM down before training begins. Leaving it on costs us the
        # `expandable_segments` setting (Unsloth force-removes it) with
        # no benefit. Confirmed 2026-05-11 after the first ckpt300
        # smoke cliffed at step ~12 with the warning:
        #   "UNSLOTH_VLLM_STANDBY is on, but requires expandable_segments
        #    to be off. We will remove expandable_segments."
        "export UNSLOTH_VLLM_STANDBY=0 && "
        "cd /workspace/LangSlice && "
        # Guarded editable install — skip when the package is already
        # importable. With the persistent training container (default flow)
        # the guard fires and the install is a no-op, saving ~30-60 sec
        # per campaign. Falls back to a real install only on the rare cold
        # container start.
        "( python -c 'import langslice_harness' >/dev/null 2>&1 "
        "  || python -m pip install --no-deps -e . >/dev/null 2>&1 ) && "
        "cd /workspace/LangSlice/models/langslice-gemma-4/training && "
        f"python -m sft.train_sft "
        f"--config {in_container_config} "
        f"--dataset {in_container_corpus} "
        f"--output-dir {in_container_out} "
        f"{curriculum_arg}"
        f"{initial_adapter_arg}"
        f"{atlas_cache_arg}"
        f"{query_cache_arg}"
        f"{bucketed_arg}"
        f"{clahe_aug_arg}"
        "--report-to none "
        "--skip-agent-eval"
    )
    cmd = _docker_exec_training_cmd(args.training_container_name, train_shell_cmd)
    logger.info("[round %d] $ %s", round_idx, " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)  # noqa: S603
    return output_adapter_dir


def _phase_eval(
    *,
    args: argparse.Namespace,
    round_idx: int,
    eval_out_dir: Path,
    served_name: str | None = None,
) -> Path:
    """Invoke slicebench tiny eval; persist summary.json next to it.

    ``served_name`` overrides the model alias slicebench routes to. In merge
    mode this is always ``args.vllm_served_name``. In LoRA mode it is the
    hot-loaded adapter name returned by ``_lora_eval_name(round_idx)``.
    """
    repo_root = args.repo_root
    in_container_out = _to_container_path(eval_out_dir, repo_root)
    served = served_name or args.vllm_served_name
    model_arg = f"litellm-proxy:{served}"
    eval_concurrency = getattr(args, "eval_concurrency", getattr(args, "concurrency", 16))
    # In-container slicebench can't reach the host's `127.0.0.1:8000` (that's
    # container loopback, not host). vLLM lives on the same docker-compose
    # default network under service name `vllm`, so route via service DNS.
    # The `LANGSLICE_LITELLM_PROXY_BASE` env var is what
    # `langslice_harness.model_resolver` reads when resolving
    # `litellm-proxy:<alias>` to a LiteLLM wrapper.
    eval_shell_cmd = (
        "set -euo pipefail && "
        "cd /workspace/LangSlice && "
        # Same guard as the train_sft step — skip the editable install
        # when langslice_harness is already importable. Fires in the
        # persistent training container; falls back only on cold start.
        "( python -c 'import langslice_harness' >/dev/null 2>&1 "
        "  || python -m pip install --no-deps -e . >/dev/null 2>&1 ) && "
        f"python -m slicebench.run --model {model_arg} "
        f"--bench {args.eval_bench} --out {in_container_out} "
        f"--num-generations {args.eval_num_generations} "
        f"--concurrency {eval_concurrency} && "
        f"python -m slicebench.score {in_container_out}"
    )
    cmd = _docker_exec_training_cmd(
        args.training_container_name,
        eval_shell_cmd,
        extra_env={
            "LANGSLICE_LITELLM_PROXY_BASE": "http://vllm:8000/v1",
            "LANGSLICE_LITELLM_PROXY_KEY": "dummy",
        },
    )
    logger.info("[round %d] $ %s", round_idx, " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)  # noqa: S603
    summary_src = eval_out_dir / "summary.json"
    summary_dst = eval_out_dir.parent / f"slicebench_summary_round_{round_idx}.json"
    if summary_src.is_file():
        try:
            shutil.copy2(summary_src, summary_dst)
        except OSError as exc:
            logger.warning("Could not copy summary.json: %s", exc)
    return summary_src


def _phase_curriculum_update(
    *,
    args: argparse.Namespace,
    round_idx: int,
    slicebench_dir: Path,
) -> Path | None:
    """Compute round_<k>_weights.json from the round's slicebench summary.

    Reads ``per_coord_bin`` MAE from ``slicebench_dir/summary.json``,
    runs ``compute_section_bins`` over the full RLVR allocation, and
    writes the new weights to
    ``args.curriculum_weights_dir/round_<k>_weights.json``. The next
    round's prompt sampler + SFT retrain pick it up automatically.

    Returns the weights path on success, or None if the slicebench
    summary doesn't exist (e.g. eval was skipped).
    """
    if args.curriculum_weights_dir is None:
        return None
    summary_path = slicebench_dir / "summary.json"
    if not summary_path.is_file():
        logger.info(
            "[round %d] curriculum-update: %s missing; skipping.",
            round_idx, summary_path,
        )
        return None

    from curriculum.bins import compute_section_bins  # noqa: PLC0415
    from curriculum.weights import (  # noqa: PLC0415
        compute_weights,
        read_per_bin_mae,
        read_weights_json,
        write_weights_json,
    )
    from rlvr.dataset import load_rlvr_allocation  # noqa: PLC0415

    per_bin_mae = read_per_bin_mae(summary_path)
    if not per_bin_mae:
        logger.warning(
            "[round %d] curriculum-update: empty per_coord_bin in %s; skipping.",
            round_idx, summary_path,
        )
        return None

    allocation = load_rlvr_allocation(
        args.allocation_root, repo_root=_REPO, planes=tuple(args.planes),
    )
    if not allocation:
        logger.warning(
            "[round %d] curriculum-update: empty allocation; skipping.",
            round_idx,
        )
        return None
    section_bin_idx = compute_section_bins(allocation)
    # ``compute_section_bins`` returns ``{section_id: int}`` — pair each
    # int with the matching plane so per_bin_mae lookups are plane-aware.
    plane_by_section: dict[str, str] = {ex.section_id: str(ex.plane) for ex in allocation}
    _LABELS = ("q1", "q2", "q3", "q4", "q5")
    section_bins: dict[str, tuple[str, str] | int] = {}
    for sid, qi in section_bin_idx.items():
        plane = plane_by_section.get(sid)
        if plane is None or qi < 0 or qi >= len(_LABELS):
            section_bins[sid] = qi
        else:
            section_bins[sid] = (plane, _LABELS[qi])

    args.curriculum_weights_dir.mkdir(parents=True, exist_ok=True)
    prev_path = args.curriculum_weights_dir / f"round_{round_idx - 1}_weights.json"
    prev_weights = (
        read_weights_json(prev_path)
        if round_idx > 0 and prev_path.is_file()
        else None
    )
    new_weights = compute_weights(
        per_bin_mae,
        section_bins,
        alpha=args.curriculum_alpha,
        max_weight_change=args.curriculum_max_weight_change,
        prev_weights=prev_weights,
        floor_fraction=args.curriculum_floor_fraction,
        smoothing=args.curriculum_smoothing,
    )
    out_path = args.curriculum_weights_dir / f"round_{round_idx}_weights.json"
    metadata = {
        "round": round_idx,
        "alpha": args.curriculum_alpha,
        "max_weight_change": args.curriculum_max_weight_change,
        "floor_fraction": args.curriculum_floor_fraction,
        "smoothing": args.curriculum_smoothing,
        "n_per_bin_mae": len(per_bin_mae),
        "n_sections": len(section_bins),
        "had_prev_weights": prev_weights is not None,
        "slicebench_summary": str(summary_path),
    }
    write_weights_json(out_path, new_weights, metadata=metadata)
    logger.info(
        "[round %d] curriculum-update: wrote %d weights -> %s",
        round_idx, len(new_weights), out_path,
    )
    return out_path


def _verify_training_container_running(name: str) -> None:
    """Fail fast if the persistent training container isn't running.

    Called at iterate.py startup (unless --skip-container-check is set).
    Without it, the first `docker exec` would surface a cryptic
    'No such container' from deep in a phase, which is hard to diagnose.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"Pre-flight check failed: could not run `docker ps` ({exc}). "
            f"Is Docker Desktop running?"
        ) from exc
    if result.stdout.strip() != name:
        raise SystemExit(
            f"Persistent training container '{name}' is not running. "
            "Start the configured training container before running.\n"
            f"(or pass --skip-container-check to bypass this check)"
        )


def _docker_exec_training_cmd(
    container_name: str,
    shell_cmd: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Build a `docker exec` command targeting the persistent training container.

    Equivalent to the old `docker compose run --rm training bash -lc ...`
    pattern, but reuses a running container so Unsloth's import + Gemma's
    weight load (the ~5-7 min cold-start tax) is paid ONCE per container
    lifetime instead of per subprocess. Combined with the langslice_harness
    editable-install guard, this turns 30-60 sec of per-round overhead into
    near-zero.

    ``extra_env`` keys become `-e KEY=value` flags. Used by the slicebench
    eval to inject LANGSLICE_LITELLM_PROXY_{BASE,KEY} for in-container
    routing to the vllm service via compose-default-network DNS.
    """
    cmd = ["docker", "exec"]
    for k, v in (extra_env or {}).items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend([container_name, "bash", "-lc", shell_cmd])
    return cmd


def _to_container_path(host_path: Path, repo_root: Path) -> str:
    """Map a host path inside ``repo_root`` to its in-container equivalent."""
    host = Path(host_path).resolve() if not Path(host_path).is_absolute() else Path(host_path)
    try:
        rel = host.relative_to(Path(repo_root).resolve())
        return f"/workspace/LangSlice/{str(rel).replace(os.sep, '/')}"
    except ValueError:
        # Already absolute & outside repo: assume the caller passed a
        # container-side path directly.
        return str(host_path).replace(os.sep, "/")


# ──────────────────────────────────────────────────────────────────────────
# vLLM lifecycle wrappers (Option A: merge + restart per round)
# ──────────────────────────────────────────────────────────────────────────

def _resolve_round_model_path(
    *, args: argparse.Namespace, round_idx: int, run_dir: Path,
) -> Path:
    """Return the merged-bf16 path vLLM should serve for this round.

    Round 0 uses ``--base-checkpoint`` directly (already merged). For
    round k>0 we merge the previous round's LoRA adapter into a new
    bf16 checkpoint under ``run_dir/round_{k-1}_merged_bf16``.
    """
    if round_idx == 0:
        return args.base_checkpoint
    prev_adapter = run_dir / f"round_{round_idx - 1}_adapter"
    merged_path = run_dir / f"round_{round_idx - 1}_merged_bf16"
    if merged_path.is_dir() and (merged_path / "config.json").is_file():
        logger.info("[round %d] using cached merged ckpt: %s", round_idx, merged_path)
        return merged_path
    if not prev_adapter.is_dir():
        raise FileNotFoundError(
            f"Cannot prepare vLLM for round {round_idx}: prior adapter "
            f"{prev_adapter} does not exist."
        )
    base_in_container = _to_container_path(args.base_checkpoint, args.repo_root)
    adapter_in_container = _to_container_path(prev_adapter, args.repo_root)
    merged_in_container = _to_container_path(merged_path, args.repo_root)
    vllm_lifecycle.merge_lora_to_bf16(
        base_checkpoint=base_in_container,
        adapter_path=adapter_in_container,
        output_path=merged_in_container,
        repo_root=args.repo_root,
        training_container_name=args.training_container_name,
        workdir=run_dir,
    )
    return merged_path


def _bring_up_vllm_for_round(
    *,
    args: argparse.Namespace,
    round_idx: int,
    run_dir: Path,
) -> Path | None:
    """Spawn vLLM with the round's merged adapter; return override path.

    In LoRA mode (``args.vllm_lora_mode``) we don't merge; instead vLLM is
    started with the SFT base + ``--enable-lora`` and we hot-swap the
    previous round's adapter (if any) via REST. The override path written
    in this case stays usable for ``compose_down_vllm`` later.
    """
    if not args.manage_vllm:
        if not args.skip_vllm_check and not _check_vllm_alive(args.vllm_url):
            logger.error(
                "vLLM not reachable at %s and --manage-vllm not set. "
                "Either pass --manage-vllm or start vllm yourself: "
                "docker compose -f docker-compose.training.yml --profile serve up -d vllm",
                args.vllm_url,
            )
            raise RuntimeError("vLLM not reachable")
        return None

    if args.vllm_lora_mode:
        # LoRA mode: serve the static SFT base + --enable-lora once. Adapters
        # for round k>0 are hot-loaded after the base is up. The same override
        # file is reused across rollout/eval phases of every round (the only
        # thing that changes round-to-round is the active adapter name, which
        # lives in REST state, not in compose).
        return _bring_up_vllm_lora_base(
            args=args, round_idx=round_idx, run_dir=run_dir,
            for_eval=False,
        )

    merged_path = _resolve_round_model_path(args=args, round_idx=round_idx, run_dir=run_dir)
    override = run_dir / f"round_{round_idx}_compose.override.yml"
    vllm_lifecycle.write_compose_override(
        output_path=override,
        model_path=_to_container_path(merged_path, args.repo_root),
        served_model_name=args.vllm_served_name,
        max_model_len=args.vllm_max_model_len,
        gpu_memory_utilization=args.vllm_gpu_mem_util,
    )
    vllm_lifecycle.compose_up_vllm(
        base_compose=args.vllm_base_compose,
        override=override,
        cwd=args.repo_root,
    )
    if not vllm_lifecycle.wait_for_vllm(
        args.vllm_url, timeout_s=args.vllm_startup_timeout,
    ):
        raise RuntimeError(f"vLLM did not become ready within {args.vllm_startup_timeout}s")
    return override


def _bring_up_vllm_lora_base(
    *,
    args: argparse.Namespace,
    round_idx: int,
    run_dir: Path,
    for_eval: bool,
    eval_adapter_dir: Path | None = None,
) -> Path:
    """Bring up vLLM with the static SFT base + ``--enable-lora``.

    After the base is ready, hot-load the appropriate round's LoRA adapter
    (if any). The model name we register and route rollouts/eval to follows
    a fixed convention:

    - ``round_<k>_rollout_adapter`` → loaded from
      ``run_dir/round_<k-1>_adapter`` (the previous round's trained
      adapter; round 0 has no prior adapter so we route to the bare base
      under ``args.vllm_served_name``).
    - ``round_<k>_eval_adapter`` → loaded from
      ``run_dir/round_<k>_adapter`` (the just-trained adapter for this round).

    The function never merges. Returns the compose override path so the
    standard tear-down helper still works.
    """
    base_model_path = (
        args.vllm_base_model_path
        if getattr(args, "vllm_base_model_path", None)
        else _to_container_path(args.base_checkpoint, args.repo_root)
    )
    override = run_dir / "lora_base_compose.override.yml"
    if not override.is_file():
        vllm_lifecycle.write_compose_override(
            output_path=override,
            model_path=base_model_path,
            served_model_name=args.vllm_served_name,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            enable_lora=True,
            max_lora_rank=args.vllm_lora_max_rank,
            max_loras=args.vllm_lora_max_loras,
        )
    vllm_lifecycle.compose_up_vllm(
        base_compose=args.vllm_base_compose,
        override=override,
        cwd=args.repo_root,
    )
    if not vllm_lifecycle.wait_for_vllm(
        args.vllm_url, timeout_s=args.vllm_startup_timeout,
    ):
        raise RuntimeError(
            f"vLLM did not become ready within {args.vllm_startup_timeout}s (lora-mode base)"
        )

    if for_eval:
        if eval_adapter_dir is None:
            raise ValueError("eval_adapter_dir required when for_eval=True")
        adapter_name = _lora_eval_name(round_idx)
        adapter_path = _to_container_path(eval_adapter_dir, args.repo_root)
        _hot_load_with_eviction(
            args=args, adapter_name=adapter_name, adapter_path=adapter_path,
        )
    else:
        # Rollout phase: load the *previous* round's trained adapter
        # (already on disk if round_idx > 0). Round 0 rollouts run against
        # the bare base so we skip the load.
        if round_idx > 0:
            prev_adapter = run_dir / f"round_{round_idx - 1}_adapter"
            if not prev_adapter.is_dir():
                raise FileNotFoundError(
                    f"Cannot route rollouts for round {round_idx} in lora-mode: "
                    f"prior adapter {prev_adapter} does not exist."
                )
            adapter_name = _lora_rollout_name(round_idx)
            adapter_path = _to_container_path(prev_adapter, args.repo_root)
            _hot_load_with_eviction(
                args=args, adapter_name=adapter_name, adapter_path=adapter_path,
            )
    return override


def _lora_rollout_name(round_idx: int) -> str:
    """Convention for the model name routed to during rollouts of round k."""
    return f"round_{round_idx}_rollout_adapter"


def _lora_eval_name(round_idx: int) -> str:
    """Convention for the model name routed to during slicebench eval of round k."""
    return f"round_{round_idx}_eval_adapter"


def _hot_load_with_eviction(
    *,
    args: argparse.Namespace,
    adapter_name: str,
    adapter_path: str,
) -> None:
    """Load ``adapter_name``; evict oldest non-base adapters if over cap.

    vLLM's ``--max-loras`` is a hard cap on simultaneously-resident
    adapters. Before loading a new one, we list registered models and, if
    we're at the cap, unload the oldest ones whose names follow our
    ``round_<k>_(rollout|eval)_adapter`` convention. The base model is
    never touched.
    """
    base_url = args.vllm_url
    cap = max(1, int(args.vllm_lora_max_loras))
    try:
        loaded = vllm_lifecycle.list_loaded_models(base_url=base_url)
    except vllm_lifecycle.VLLMLoRAError as exc:
        # Probe failed → no eviction; load attempt below will surface the issue.
        logger.warning("Could not list loaded models for eviction: %s", exc)
        loaded = []
    # Already loaded? Treat as a hit.
    if adapter_name in loaded:
        logger.info("LoRA adapter %r already loaded; skipping load.", adapter_name)
        return
    # Filter to our adapter naming convention so we don't clobber unrelated adapters.
    convention = [
        m for m in loaded
        if m.startswith("round_") and m.endswith("_adapter")
    ]
    # Sort by round index (stable for two same-round names).
    def _round_of(name: str) -> int:
        try:
            return int(name.split("_", 2)[1])
        except (IndexError, ValueError):
            return -1
    convention.sort(key=_round_of)
    # Number of slots already used by our convention adapters.
    while len(convention) + 1 > cap and convention:
        victim = convention.pop(0)
        try:
            vllm_lifecycle.unload_lora_adapter(base_url=base_url, lora_name=victim)
        except vllm_lifecycle.VLLMLoRAError as exc:
            logger.warning("Eviction of %r failed (continuing): %s", victim, exc)
    vllm_lifecycle.load_lora_adapter(
        base_url=base_url, lora_name=adapter_name, lora_path=adapter_path,
    )


def _tear_down_vllm_for_round(
    *,
    args: argparse.Namespace,
    override: Path | None,
) -> None:
    if not args.manage_vllm or override is None:
        return
    vllm_lifecycle.compose_down_vllm(
        base_compose=args.vllm_base_compose,
        override=override,
        cwd=args.repo_root,
    )


# ──────────────────────────────────────────────────────────────────────────
# Round orchestration
# ──────────────────────────────────────────────────────────────────────────

def _run_round(
    *,
    args: argparse.Namespace,
    round_idx: int,
    run_state: state_mod.RunState,
) -> dict[str, Any]:
    """Drive one round through every phase, honoring resume state."""
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = run_dir / f"round_{round_idx}_rollout_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # In LoRA mode, rollouts for round k>0 are routed to the previous
    # round's hot-loaded adapter (round_<k-1>_rollout_adapter); round 0
    # rollouts run against the bare base under args.model_alias.
    if args.vllm_lora_mode and round_idx > 0:
        rollout_alias = _lora_rollout_name(round_idx)
    else:
        rollout_alias = args.model_alias
    model_str = f"litellm-proxy:{rollout_alias}"

    summary: dict[str, Any] = {
        "round": round_idx,
        "model": model_str,
        "filter_mode": args.filter_mode,
    }

    # Bring up vLLM (skipped if --manage-vllm not set).
    override_path = _bring_up_vllm_for_round(
        args=args, round_idx=round_idx, run_dir=run_dir,
    )

    try:
        # Phase: sampled+rollouts (combined — we don't separately persist specs
        # because rollouts.json already encodes (section_id, gen_idx) keys
        # which the dedup-on-resume relies on).
        rollouts_pickle = run_dir / f"round_{round_idx}_rollouts.json"
        if not run_state.is_phase_done(round_idx, "rollouts"):
            specs, rollouts = _phase_rollouts(
                args=args,
                round_idx=round_idx,
                artifacts_root=artifacts_root,
                model_str=model_str,
            )
            # Persist a lightweight rollouts manifest. We can't pickle the
            # full RolloutResult (events are dicts but run_dir is Path), but
            # we can dump enough to skip re-running on resume.
            rollouts_pickle.write_text(
                json.dumps(
                    [_rollout_to_dict(r) for r in rollouts],
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary["n_rollouts_total"] = len(rollouts)
            summary["n_rollouts_failed"] = sum(1 for r in rollouts if r.error is not None)
            run_state.mark_phase(
                round_idx, "rollouts",
                rollouts_manifest=str(rollouts_pickle),
            )
            state_mod.save_state(run_state, args.output_dir)
        else:
            rollouts = _load_rollouts_from_manifest(rollouts_pickle)

        # Phase: scored
        scored_path = run_dir / f"round_{round_idx}_scored.json"
        if not run_state.is_phase_done(round_idx, "scored"):
            scored = _score_rollouts(rollouts)
            scored_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
            run_state.mark_phase(round_idx, "scored", scored_path=str(scored_path))
            state_mod.save_state(run_state, args.output_dir)
        else:
            scored = json.loads(scored_path.read_text(encoding="utf-8"))

        # Feed scored rollouts into the adaptive accept buffer (Phase 5).
        # This runs whether or not --adaptive-accept is set so resumption from
        # "filtered" still has a populated buffer on the second attempt.
        _adaptive_buf = getattr(args, "_adaptive_buffer", None)
        if _adaptive_buf is not None:
            for _row in scored:
                _abs_err = _row.get("abs_err_mm")
                _extent = _row.get("plane_extent_mm")
                if _abs_err is not None and _extent is not None and float(_extent) > 0:
                    _adaptive_buf.append((
                        float(_abs_err),
                        float(_extent),
                        str(_row.get("plane", "")),
                        str(_row.get("atlas", "")),
                        str(_row.get("section_id", "")),
                    ))
            logger.debug(
                "Round %d: adaptive buffer now has %d observations.",
                round_idx, len(_adaptive_buf),
            )

        # Phase: filtered
        filtered_path = run_dir / f"round_{round_idx}_filtered.json"
        if not run_state.is_phase_done(round_idx, "filtered"):
            kept = _kept_rollouts_for_filter(
                scored,
                filter_mode=args.filter_mode,
                threshold_pct=args.threshold_pct,
                adaptive_buffer=_adaptive_buf,
                adaptive_quantile=getattr(args, "adaptive_accept_quantile", 0.95),
                adaptive_warmup_n=getattr(args, "adaptive_accept_warmup_n", 50),
            )
            filtered_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
            summary["n_rollouts_kept"] = len(kept)
            run_state.mark_phase(round_idx, "filtered", filtered_path=str(filtered_path))
            state_mod.save_state(run_state, args.output_dir)
        else:
            kept = json.loads(filtered_path.read_text(encoding="utf-8"))

        # Phase: appended
        if not run_state.is_phase_done(round_idx, "appended"):
            iterative_jsonl, append_stats = _phase_append_iterative(
                args=args,
                round_idx=round_idx,
                rollouts=rollouts,
                kept=kept,
                model_str=model_str,
            )
            # Phase: synthetic (only when --synthetic-per-round > 0; default is
            # 0 so this block is never entered and behavior is byte-identical).
            synth_stats: dict[str, int] = {}
            if getattr(args, "synthetic_per_round", 0) > 0:
                synth_stats = _phase_synthetic(
                    args=args,
                    round_idx=round_idx,
                    iterative_jsonl=iterative_jsonl,
                )
            summary.update(append_stats)
            summary.update(synth_stats)
            summary["iterative_jsonl"] = str(iterative_jsonl)
            run_state.mark_phase(
                round_idx, "appended",
                iterative_jsonl=str(iterative_jsonl),
                **append_stats,
            )
            state_mod.save_state(run_state, args.output_dir)

        # Phase: unioned
        if not run_state.is_phase_done(round_idx, "unioned"):
            unioned_jsonl, union_stats = _phase_union(args=args, round_idx=round_idx)
            summary["unioned_corpus_jsonl"] = str(unioned_jsonl)
            summary["union_stats"] = union_stats
            run_state.mark_phase(
                round_idx, "unioned",
                unioned_jsonl=str(unioned_jsonl),
                union_stats=union_stats,
            )
            state_mod.save_state(run_state, args.output_dir)
        else:
            unioned_jsonl = run_dir / f"round_{round_idx}_corpus.jsonl"

        # Tear down vLLM before retraining (training container needs the GPU).
        _tear_down_vllm_for_round(args=args, override=override_path)
        override_path = None

        # Phase: trained
        adapter_dir = run_dir / f"round_{round_idx}_adapter"
        if args.skip_retrain:
            logger.info(
                "[round %d] --skip-retrain set; unioned corpus at %s. Stopping.",
                round_idx, unioned_jsonl,
            )
            summary["skipped_retrain"] = True
            return summary
        if not run_state.is_phase_done(round_idx, "trained"):
            adapter_dir.mkdir(parents=True, exist_ok=True)
            _phase_train(
                args=args,
                round_idx=round_idx,
                unioned_jsonl=unioned_jsonl,
                output_adapter_dir=adapter_dir,
            )
            summary["adapter_dir"] = str(adapter_dir)
            run_state.mark_phase(round_idx, "trained", adapter_dir=str(adapter_dir))
            state_mod.save_state(run_state, args.output_dir)

        # Bring vLLM back up serving the new adapter for slicebench eval.
        eval_out = run_dir / f"round_{round_idx}_slicebench"
        if not args.skip_eval:
            override_path = _bring_up_vllm_for_round_for_eval(
                args=args, round_idx=round_idx, run_dir=run_dir,
                adapter_dir=adapter_dir,
            )
            try:
                if not run_state.is_phase_done(round_idx, "evaluated"):
                    eval_out.mkdir(parents=True, exist_ok=True)
                    eval_served_name = (
                        _lora_eval_name(round_idx)
                        if args.vllm_lora_mode else None
                    )
                    _phase_eval(
                        args=args, round_idx=round_idx, eval_out_dir=eval_out,
                        served_name=eval_served_name,
                    )
                    summary["slicebench_dir"] = str(eval_out)
                    run_state.mark_phase(
                        round_idx, "evaluated", slicebench_dir=str(eval_out),
                    )
                    state_mod.save_state(run_state, args.output_dir)
            finally:
                _tear_down_vllm_for_round(args=args, override=override_path)
                override_path = None

        # CURRICULUM HOOK: refresh round_<k>_weights.json from this
        # round's slicebench per_coord_bin MAE so the next round's
        # prompt sampler + SFT retrain can up-weight weak bins.
        if (
            args.curriculum_weights_dir is not None
            and not args.skip_eval
            and not run_state.is_phase_done(round_idx, "curriculum")
        ):
            weights_path = _phase_curriculum_update(
                args=args, round_idx=round_idx, slicebench_dir=eval_out,
            )
            if weights_path is not None:
                summary["curriculum_weights_path"] = str(weights_path)
                run_state.mark_phase(
                    round_idx, "curriculum",
                    curriculum_weights_path=str(weights_path),
                )
                state_mod.save_state(run_state, args.output_dir)

        run_state.mark_phase(round_idx, "done")
        state_mod.save_state(run_state, args.output_dir)
        return summary
    finally:
        _tear_down_vllm_for_round(args=args, override=override_path)


def _bring_up_vllm_for_round_for_eval(
    *,
    args: argparse.Namespace,
    round_idx: int,
    run_dir: Path,
    adapter_dir: Path,
) -> Path | None:
    """Like ``_bring_up_vllm_for_round`` but for the just-trained adapter."""
    if not args.manage_vllm:
        return None
    if args.vllm_lora_mode:
        return _bring_up_vllm_lora_base(
            args=args, round_idx=round_idx, run_dir=run_dir,
            for_eval=True, eval_adapter_dir=adapter_dir,
        )
    merged_path = run_dir / f"round_{round_idx}_merged_bf16"
    if not merged_path.is_dir():
        base_in_container = _to_container_path(args.base_checkpoint, args.repo_root)
        adapter_in_container = _to_container_path(adapter_dir, args.repo_root)
        merged_in_container = _to_container_path(merged_path, args.repo_root)
        vllm_lifecycle.merge_lora_to_bf16(
            base_checkpoint=base_in_container,
            adapter_path=adapter_in_container,
            output_path=merged_in_container,
            repo_root=args.repo_root,
            training_container_name=args.training_container_name,
            workdir=run_dir,
        )
    override = run_dir / f"round_{round_idx}_eval_compose.override.yml"
    vllm_lifecycle.write_compose_override(
        output_path=override,
        model_path=_to_container_path(merged_path, args.repo_root),
        served_model_name=args.vllm_served_name,
        max_model_len=args.vllm_max_model_len,
        gpu_memory_utilization=args.vllm_gpu_mem_util,
    )
    vllm_lifecycle.compose_up_vllm(
        base_compose=args.vllm_base_compose,
        override=override,
        cwd=args.repo_root,
    )
    if not vllm_lifecycle.wait_for_vllm(
        args.vllm_url, timeout_s=args.vllm_startup_timeout,
    ):
        raise RuntimeError("vLLM did not become ready within timeout (eval)")
    return override


# Lightweight rollout serialization for resume.

def _rollout_to_dict(r: Any) -> dict[str, Any]:
    """Serialize enough of a RolloutResult to skip re-running on resume.

    We keep ``events`` so trace conversion can run from disk; ``run_dir``
    is preserved as a string so trace_format can resolve atlas image paths.
    """
    return {
        "spec": {
            "section_id": r.spec.section_id,
            "subject_id": r.spec.subject_id,
            "image_path": str(r.spec.image_path),
            "atlas_name": r.spec.atlas_name,
            "plane": r.spec.plane,
            "truth_mm": float(r.spec.truth_mm),
            "plane_extent_mm": float(r.spec.plane_extent_mm),
            "generation_idx": int(r.spec.generation_idx),
        },
        "submitted_position_mm": (
            float(r.submitted_position_mm)
            if r.submitted_position_mm is not None else None
        ),
        "reasoning": r.reasoning,
        "events": r.events,
        "run_dir": str(r.run_dir),
        "error": r.error,
    }


def _load_rollouts_from_manifest(path: Path) -> list[Any]:
    """Reverse of ``_rollout_to_dict``."""
    from iSFT.rollout import RolloutResult, RolloutSpec  # noqa: PLC0415

    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[Any] = []
    for r in payload:
        spec = RolloutSpec(
            section_id=r["spec"]["section_id"],
            subject_id=r["spec"]["subject_id"],
            image_path=Path(r["spec"]["image_path"]),
            atlas_name=r["spec"]["atlas_name"],
            plane=r["spec"]["plane"],
            truth_mm=r["spec"]["truth_mm"],
            plane_extent_mm=r["spec"]["plane_extent_mm"],
            generation_idx=r["spec"]["generation_idx"],
        )
        out.append(RolloutResult(
            spec=spec,
            submitted_position_mm=r["submitted_position_mm"],
            reasoning=r["reasoning"],
            events=r["events"],
            run_dir=Path(r["run_dir"]),
            error=r["error"],
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def _assert_no_double_lora(args: argparse.Namespace) -> None:
    """Refuse to start a run that would double-apply the SFT LoRA adapter.

    The double-LoRA bug pattern: ``--vllm-lora-mode`` is the canonical
    iterative serving mode and is typically paired with
    ``configs/sft_iterative.toml`` (which uses a merged-bf16 SFT base —
    i.e. the base model already contains the SFT weights). If
    ``--sft-initial-adapter`` is also passed in that combo, train_sft.py
    will load the SFT adapter on top of the already-SFT-merged base,
    silently double-applying the LoRA delta and producing a degraded
    model with no warning. Caught a real regression once; see project
    memory ``feedback_isft_double_lora_application``.
    """
    if args.sft_initial_adapter is not None and args.vllm_lora_mode:
        raise RuntimeError(
            "Refusing to start: --sft-initial-adapter + --vllm-lora-mode is "
            "the double-LoRA bug pattern. The vLLM LoRA serving mode is "
            "designed to pair with sft_iterative.toml (merged-bf16 base, no "
            "initial adapter). Either drop --sft-initial-adapter, or switch "
            "to sft_default.toml + the merge-restart serving flow. See "
            "models/langslice-gemma-4/training/configs/sft_iterative.toml "
            "for the safe iterative defaults."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    if args.rounds < 1:
        logger.error("--rounds must be >= 1 (got %d)", args.rounds)
        return 2
    _assert_no_double_lora(args)
    if not args.skip_container_check and not args.skip_retrain:
        _verify_training_container_running(args.training_container_name)

    # Wire --vllm-url to the env vars that langslice_harness.model_resolver
    # reads when resolving "litellm-proxy:<alias>" to a LiteLlm wrapper.
    # Without this, rollouts try to hit the default proxy at port 4000
    # (which has nothing listening) instead of our vLLM at args.vllm_url.
    os.environ.setdefault("LANGSLICE_LITELLM_PROXY_BASE", args.vllm_url)
    os.environ.setdefault("LANGSLICE_LITELLM_PROXY_KEY", "dummy")
    logger.info(
        "Routing rollouts via LANGSLICE_LITELLM_PROXY_BASE=%s",
        os.environ["LANGSLICE_LITELLM_PROXY_BASE"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.iterative_corpus_dir.mkdir(parents=True, exist_ok=True)

    # Adaptive accept buffer (Phase 5).
    # Constructed once per run so error observations accumulate across rounds.
    # Attached to args so _run_round can read it without extra parameters.
    if args.adaptive_accept:
        from adaptive.schedule import make_error_buffer as _make_error_buffer  # noqa: PLC0415
        args._adaptive_buffer = _make_error_buffer()  # type: ignore[attr-defined]  # noqa: SLF001
        logger.info(
            "Adaptive accept enabled: quantile=%.2f, warmup_n=%d.",
            args.adaptive_accept_quantile, args.adaptive_accept_warmup_n,
        )
    else:
        args._adaptive_buffer = None  # type: ignore[attr-defined]  # noqa: SLF001

    # Seen-section ledger (Phase 4).
    # Attach to args so all phase functions can access it without extra params.
    if args.use_seen_ledger:
        ledger_path = args.ledger_path or (args.output_dir / "seen_ledger.jsonl")
        if args.reset_ledger and ledger_path.is_file():
            ledger_path.unlink()
            logger.info("--reset-ledger: deleted %s", ledger_path)
        ledger = SeenLedger(ledger_path, manifest_root=args.allocation_root)
        try:
            ledger.load()
        except StaleLedgerError as exc:
            logger.error("Stale ledger detected: %s", exc)
            raise
        logger.info(
            "Seen-section ledger loaded: %d sections seen so far (%s)",
            ledger.seen_count(), ledger_path,
        )
        args._seen_ledger = ledger  # type: ignore[attr-defined]  # noqa: SLF001
    else:
        args._seen_ledger = None  # type: ignore[attr-defined]  # noqa: SLF001

    run_id = args.run_id or args.output_dir.name
    run_state = state_mod.init_or_resume(
        args.output_dir, run_id=run_id, rounds_total=args.rounds,
    )
    logger.info(
        "Run id=%s rounds_total=%d resume_round=%d resume_phase=%s",
        run_state.run_id, run_state.rounds_total,
        run_state.round, run_state.phase,
    )

    start = max(args.start_round, run_state.round)
    summaries: list[dict[str, Any]] = []
    for round_idx in range(start, args.rounds):
        logger.info("════════ Round %d / %d ════════", round_idx, args.rounds - 1)
        summary = _run_round(args=args, round_idx=round_idx, run_state=run_state)
        summaries.append(summary)
        if round_idx + 1 < args.rounds:
            run_state.advance_round(summary=summary)
            state_mod.save_state(run_state, args.output_dir)

    final_summary = {
        "run_id": run_state.run_id,
        "rounds_total": args.rounds,
        "rounds_completed": run_state.round + (1 if run_state.phase == "done" else 0),
        "rounds": summaries,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(final_summary, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Run summary: %s", json.dumps(final_summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
