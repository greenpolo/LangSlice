"""Convert ADK rollout output into a langslice-native SFT JSONL row.

Reuses ``_convert_trace`` / ``_classify_quality`` / ``_trim_trace`` from
``_local/trace_collection/build_sft_corpus.py`` so iterative-corpus rows
are byte-for-byte schema-identical to the distilled corpus rows that the
SFT trainer already validates.

Two extra audit fields land in ``quality``:

    quality.source           = "expert_iteration"
    quality.iteration_round  = <round index, 0 for the MVP>

These are additive and ignored by the SFT validator; a downstream curator
can stratify or drop iterative rows by inspecting them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from langslice_harness.atlas.core import get_in_plane_long_edge, load_atlas
from langslice_harness.harness.estimation.trace_collection import (
    plane_rescue_threshold_mm,
    plane_tolerance_mm,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUILD_SFT_CORPUS_PATH = (
    _REPO_ROOT / "_local" / "trace_collection" / "build_sft_corpus.py"
)


def _load_build_sft_corpus_module() -> Any:
    """Import the private ``_local`` module via importlib.

    The build_sft_corpus script lives outside the installable package tree
    (it's under ``_local/``, which is gitignored as per repo policy). We
    deliberately import-by-path so this file works regardless of whether
    the script is on ``PYTHONPATH``.

    ``build_sft_corpus.py`` also reads ``.env`` at import time. When .env
    is absent (e.g. fresh container, CI), we synthesize an empty one so
    the module import doesn't crash; nothing in the helper functions we
    actually call needs an API key.
    """
    env_path = _REPO_ROOT / ".env"
    created_env = False
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
        created_env = True
    try:
        spec = importlib.util.spec_from_file_location(
            "langslice_build_sft_corpus", _BUILD_SFT_CORPUS_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load build_sft_corpus from {_BUILD_SFT_CORPUS_PATH}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        return module
    finally:
        if created_env:
            try:
                env_path.unlink()
            except OSError:
                pass


_BUILD_SFT_CORPUS = None


def _build_sft_corpus():  # noqa: ANN202 — dynamic module
    global _BUILD_SFT_CORPUS
    if _BUILD_SFT_CORPUS is None:
        _BUILD_SFT_CORPUS = _load_build_sft_corpus_module()
    return _BUILD_SFT_CORPUS


# ──────────────────────────────────────────────────────────────────────────
# Helpers from build_sft_corpus, re-exported for unit-test convenience
# ──────────────────────────────────────────────────────────────────────────

def convert_trace(
    events: list[dict[str, Any]],
    *,
    run_dir: Path,
    atlas_name: str,
    plane: str,
    out_dir: Path,
) -> list[dict[str, Any]] | None:
    """Thin wrapper around build_sft_corpus._convert_trace for direct reuse."""
    return _build_sft_corpus()._convert_trace(
        events,
        run_dir=run_dir,
        atlas_name=atlas_name,
        plane=plane,
        out_dir=out_dir,
    )


def classify_quality(
    *,
    model: str | None,
    err: float,
    tol: float,
    rescue: float | None,
    sub_canonical_mm: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return _build_sft_corpus()._classify_quality(
        model=model,
        err=err,
        tol=tol,
        rescue=rescue,
        sub_canonical_mm=sub_canonical_mm,
        events=events,
    )


def trim_trace(
    trace: list[dict[str, Any]],
    *,
    max_fetch_calls: int = 2,
    max_total_images: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _build_sft_corpus()._trim_trace(
        trace,
        max_fetch_calls=max_fetch_calls,
        max_total_images=max_total_images,
    )


def stage_query_image(
    src_rel: str,
    out_dir: Path,
    sid: str,
    *,
    atlas_name: str,
    plane: str,
    apply_clahe: bool,
    run_dir: Path | None = None,
) -> str:
    return _build_sft_corpus()._stage_query_image(
        src_rel,
        out_dir,
        sid,
        atlas_name=atlas_name,
        plane=plane,
        apply_clahe=apply_clahe,
        run_dir=run_dir,
    )


def atlas_version(atlas_name: str) -> str:
    return _build_sft_corpus()._atlas_version(atlas_name)


def user_prompt_text(plane: str, atlas_name: str) -> str:
    return _build_sft_corpus()._user_prompt_text(plane, atlas_name)


def estimate_tokens(
    *,
    n_query_images: int,
    n_atlas_images: int,
    n_fetch_calls: int,
    reasoning: str,
) -> int:
    return _build_sft_corpus()._estimate_tokens(
        n_query_images=n_query_images,
        n_atlas_images=n_atlas_images,
        n_fetch_calls=n_fetch_calls,
        reasoning=reasoning,
    )


# ──────────────────────────────────────────────────────────────────────────
# Driver-facing entrypoint
# ──────────────────────────────────────────────────────────────────────────

def build_row(
    *,
    section_id: str,
    subject_id: str,
    atlas_name: str,
    plane: str,
    query_image_rel: str,
    events: list[dict[str, Any]],
    submitted_position_mm: float,
    sub_canonical_mm: float,
    abs_err_mm: float,
    model: str,
    iteration_round: int,
    out_dir: Path,
    run_dir: Path,
    apply_clahe: bool,
    max_fetch_calls: int = 2,
    max_total_images: int = 12,
) -> dict[str, Any] | None:
    """Construct one SFT-validatable row from a rollout's recorder state.

    Returns ``None`` when the trace lacks a terminal ``submit_estimate`` or
    when the trace cannot be reconstructed (caller should drop the rollout).

    Caller is responsible for staging the query image (so the ``out_dir``
    layout matches the distilled corpus). ``query_image_rel`` is the
    out_dir-relative path returned by ``stage_query_image``.
    """
    trace = convert_trace(
        events,
        run_dir=run_dir,
        atlas_name=atlas_name,
        plane=plane,
        out_dir=out_dir,
    )
    if not trace or "submit" not in trace[-1]:
        return None

    tol = plane_tolerance_mm(atlas_name, plane)
    rescue = plane_rescue_threshold_mm(atlas_name, plane)

    quality = classify_quality(
        model=model,
        err=abs_err_mm,
        tol=tol,
        rescue=rescue,
        sub_canonical_mm=sub_canonical_mm,
        events=events,
    )
    quality["preprocessing"] = "clahe" if apply_clahe else "raw"
    long_edge = get_in_plane_long_edge(load_atlas(atlas_name), plane=plane)
    quality["query_long_edge_px"] = long_edge

    if quality["accuracy"] in {"tight", "in_tolerance"}:
        quality["acceptance_tier"] = "strict"
    elif quality["accuracy"] == "rescued":
        quality["acceptance_tier"] = "rescued"
    else:
        quality["acceptance_tier"] = "out_of_tolerance"
    quality["rescue_threshold_mm"] = round(rescue, 4)

    # Audit-trail fields the spec calls out for iterative rows.
    quality["source"] = "expert_iteration"
    quality["iteration_round"] = iteration_round
    quality["submitted_position_mm"] = round(float(submitted_position_mm), 4)

    trace, trim_meta = trim_trace(
        trace,
        max_fetch_calls=max_fetch_calls,
        max_total_images=max_total_images,
    )
    quality["trim"] = trim_meta

    reasoning = ""
    if trace and "submit" in trace[-1]:
        reasoning = trace[-1]["submit"].get("args", {}).get("reasoning", "") or ""
    n_atlas = sum(
        len(s.get("tool_result", {}).get("image_paths") or [])
        for s in trace if "tool_call" in s
    )
    n_fetch = sum(1 for s in trace if "tool_call" in s)
    quality["estimated_tokens"] = estimate_tokens(
        n_query_images=1,
        n_atlas_images=n_atlas,
        n_fetch_calls=n_fetch,
        reasoning=reasoning,
    )

    return {
        "bucket": 1,
        "atlas_name": atlas_name,
        "atlas_version": atlas_version(atlas_name),
        "plane": plane,
        "subject_id": subject_id,
        # Phase 6: propagate section_id so train_sft.py's WeightedRandomSampler
        # can key curriculum weights by section_id rather than subject_id.
        # The iSFT real-rollout path provides section_id directly from the
        # allocation; real rollouts should always pass a non-empty value here.
        "section_id": section_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [query_image_rel],
        "user_prompt_text": user_prompt_text(plane, atlas_name),
        "trace": trace,
        "quality": quality,
    }
