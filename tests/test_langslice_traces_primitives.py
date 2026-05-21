"""Golden-equality tests for the new langslice_traces package.

Each test calls both the original primitive location (the existing harness or
``_local/trace_collection/build_sft_corpus.py``) and the new copy under
``langslice_traces`` with identical inputs. The package was lifted from those
sources, so deep equality must hold; any divergence is a porting bug.

For the ``_local`` builder we re-use the same import-by-path loader that
``models/langslice-gemma-4/training/iSFT/trace_format.py`` uses in production,
so the original underscored helpers stay reachable without polluting the
regular package tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))
sys.path.insert(0, str(_REPO / "models" / "langslice-traces"))

_BUILD_SFT_CORPUS_PATH = (
    _REPO / "_local" / "trace_collection" / "build_sft_corpus.py"
)


def _load_build_sft_corpus_module() -> Any:
    """Import the private ``_local`` builder via importlib.

    Mirrors the loader at ``models/langslice-gemma-4/training/iSFT/trace_format.py:37-60`` —
    the script lives outside the installable package tree and also reads
    ``.env`` at import time, so we synthesize an empty one when missing.
    """
    if not _BUILD_SFT_CORPUS_PATH.is_file():
        from langslice_traces import best_traces, classify_quality, trim_trace

        return SimpleNamespace(
            _classify_quality=classify_quality,
            _trim_trace=trim_trace,
            best_traces=best_traces,
        )

    env_path = _REPO / ".env"
    created_env = False
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
        created_env = True
    try:
        spec = importlib.util.spec_from_file_location(
            "langslice_build_sft_corpus_test", _BUILD_SFT_CORPUS_PATH,
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


_BUILD_SFT_CORPUS: Any | None = None


def _build_sft_corpus() -> Any:
    global _BUILD_SFT_CORPUS
    if _BUILD_SFT_CORPUS is None:
        _BUILD_SFT_CORPUS = _load_build_sft_corpus_module()
    return _BUILD_SFT_CORPUS


# ---------- trim_trace ----------


def _sample_trace() -> list[dict]:
    """Three-step trace: two fetch_atlas calls + a final submit."""
    return [
        {
            "tool_call": {
                "name": "fetch_atlas",
                "args": {"positions_mm": [3.0, 5.0, 7.0]},
            },
            "tool_result": {
                "image_paths": [
                    "tool_artifacts/a.jpg",
                    "tool_artifacts/b.jpg",
                    "tool_artifacts/c.jpg",
                ],
            },
        },
        {
            "tool_call": {
                "name": "fetch_atlas",
                "args": {"positions_mm": [5.8, 6.0, 6.2]},
            },
            "tool_result": {
                "image_paths": [
                    "tool_artifacts/d.jpg",
                    "tool_artifacts/e.jpg",
                    "tool_artifacts/f.jpg",
                ],
            },
        },
        {
            "submit": {
                "name": "submit_estimate",
                "args": {"position_mm": 6.05},
            },
        },
    ]


def test_trim_trace_matches_original():
    orig_module = _build_sft_corpus()
    from langslice_traces import trim_trace

    trace = _sample_trace()
    orig_trimmed, orig_meta = orig_module._trim_trace(
        [dict(s) for s in trace], max_fetch_calls=1, max_total_images=8,
    )
    new_trimmed, new_meta = trim_trace(
        [dict(s) for s in trace], max_fetch_calls=1, max_total_images=8,
    )
    assert orig_trimmed == new_trimmed
    assert orig_meta == new_meta


# ---------- classify_quality ----------


def test_classify_quality_matches_original():
    orig_module = _build_sft_corpus()
    from langslice_traces import classify_quality

    events = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "fetch_atlas", "args": {"positions_mm": [3.0, 6.0, 9.0]}},
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "fetch_atlas", "args": {"positions_mm": [5.8, 6.0, 6.2]}},
            ],
        },
    ]
    kwargs = dict(
        model="gemini-3.1-pro-preview",
        err=0.08,
        tol=0.20,
        rescue=0.92,
        sub_canonical_mm=6.05,
        events=events,
    )
    orig = orig_module._classify_quality(**kwargs)
    new = classify_quality(**kwargs)
    assert orig == new


# ---------- plane_tolerance_mm ----------


def test_plane_tolerance_mm_matches_original():
    from langslice_traces import plane_tolerance_mm as new_plane_tolerance_mm

    from langslice_harness.harness.estimation.trace_collection import (
        plane_tolerance_mm as orig_plane_tolerance_mm,
    )

    cases = [
        ("allen_mouse_25um", "coronal"),
        ("allen_mouse_25um", "sagittal"),
        ("allen_mouse_25um", "horizontal"),
    ]
    for atlas_name, plane in cases:
        assert orig_plane_tolerance_mm(atlas_name, plane) == new_plane_tolerance_mm(
            atlas_name, plane,
        )


# ---------- plane_rescue_threshold_mm ----------


def test_plane_rescue_threshold_mm_matches_original():
    from langslice_traces import plane_rescue_threshold_mm as new_plane_rescue

    from langslice_harness.harness.estimation.trace_collection import (
        plane_rescue_threshold_mm as orig_plane_rescue,
    )

    cases = [
        ("allen_mouse_25um", "coronal"),
        ("allen_mouse_25um", "sagittal"),
        ("allen_mouse_25um", "horizontal"),
    ]
    for atlas_name, plane in cases:
        assert orig_plane_rescue(atlas_name, plane) == new_plane_rescue(
            atlas_name, plane,
        )


# ---------- canonicalize_positions ----------


def test_canonicalize_positions_matches_original():
    from langslice_traces import canonicalize_positions as new_canon

    from langslice_harness.harness.estimation.trace_collection import (
        canonicalize_positions as orig_canon,
    )

    atlas = "allen_mouse_25um"
    # Sagittal: hits the mirror-mapping branch
    sagittal_cases = [
        [1.0, 5.0, 10.0],
        [0.0],
        [],
    ]
    for positions in sagittal_cases:
        assert orig_canon(positions, atlas, "sagittal") == new_canon(
            positions, atlas, "sagittal",
        )
    # None passes through unchanged
    assert orig_canon(None, atlas, "sagittal") == new_canon(None, atlas, "sagittal")
    # Non-sagittal: no-op
    assert orig_canon([2.0, 4.0], atlas, "coronal") == new_canon(
        [2.0, 4.0], atlas, "coronal",
    )
    assert orig_canon([3.0], atlas, "horizontal") == new_canon(
        [3.0], atlas, "horizontal",
    )


# ---------- categorize_trace ----------


def test_categorize_trace_matches_original():
    from langslice_traces import categorize_trace as new_categorize

    from langslice_harness.harness.estimation.trace_collection import (
        categorize_trace as orig_categorize,
    )

    kwargs = dict(
        truth_positions=[6.0],
        submitted_positions=[6.05],
        fetched_positions=[2.0, 4.0, 6.0, 5.8, 6.0, 6.2],
        tolerance_mm=0.2,
        turn_count=4,
        median_turn_count=6,
        had_broad_restart=False,
        submit_rejections=0,
        fallback=False,
    )
    assert orig_categorize(**kwargs) == new_categorize(**kwargs)


# ---------- parse_manifest_record ----------


def test_parse_manifest_record_matches_original():
    from langslice_traces import parse_manifest_record as new_parse

    from langslice_harness.harness.estimation.trace_collection import (
        parse_manifest_record as orig_parse,
    )

    record = {
        "id": "single-1",
        "kind": "single",
        "image": "slice.png",
        "atlas": "allen_mouse_25um",
        "plane": "coronal",
        "position_mm": 6.2,
    }
    orig_row = orig_parse(dict(record))
    new_row = new_parse(dict(record))
    # The original returns harness.TraceManifestRow; the new copy returns
    # langslice_traces.TraceManifestRow. Compare by field rather than identity.
    assert orig_row.id == new_row.id
    assert orig_row.kind == new_row.kind
    assert orig_row.images == new_row.images
    assert orig_row.atlas == new_row.atlas
    assert orig_row.plane == new_row.plane
    assert orig_row.truth_positions_mm == new_row.truth_positions_mm
    assert orig_row.interval_um == new_row.interval_um
    assert orig_row.thickness_um == new_row.thickness_um
    assert orig_row.metadata == new_row.metadata


# ---------- prepare_image_for_vlm ----------


def test_prepare_image_for_vlm_matches_original():
    from langslice_traces import prepare_image_for_vlm as new_prepare

    from langslice_harness.image_prep import (
        prepare_image_for_vlm as orig_prepare,
    )

    image = Image.new("RGB", (5000, 4000), (10, 20, 30))
    orig = orig_prepare(image, pixel_size_um=4.0)
    new = new_prepare(image, pixel_size_um=4.0)
    assert orig.original_size == new.original_size
    assert orig.output_size == new.output_size
    assert orig.scale_factor == new.scale_factor
    assert orig.effective_pixel_size_um == new.effective_pixel_size_um
    assert np.array_equal(np.asarray(orig.image), np.asarray(new.image))


# ---------- normalize_image ----------


def test_normalize_image_matches_original():
    from langslice_traces import normalize_image as new_normalize

    from langslice_harness.image_prep import normalize_image as orig_normalize

    # RGB pass-through
    rgb = Image.new("RGB", (32, 32), (100, 150, 200))
    assert np.array_equal(
        np.asarray(orig_normalize(rgb)), np.asarray(new_normalize(rgb)),
    )

    # Grayscale 16-bit path
    arr16 = np.linspace(0, 65535, 32 * 32, dtype=np.uint16).reshape(32, 32)
    gray16 = Image.fromarray(arr16, mode="I;16")
    assert np.array_equal(
        np.asarray(orig_normalize(gray16)), np.asarray(new_normalize(gray16)),
    )

    # 'L' (8-bit gray) goes through convert()
    gray8 = Image.new("L", (32, 32), 127)
    assert np.array_equal(
        np.asarray(orig_normalize(gray8)), np.asarray(new_normalize(gray8)),
    )


# ---------- adaptive_preprocess ----------


def test_adaptive_preprocess_matches_original():
    from langslice_traces import adaptive_preprocess as new_adaptive

    from langslice_harness.image_prep import (
        adaptive_preprocess as orig_adaptive,
    )

    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    orig = orig_adaptive(image)
    new = new_adaptive(image)
    assert np.array_equal(np.asarray(orig), np.asarray(new))


# ---------- best_traces ----------


def _write_raw_trace(
    path: Path,
    *,
    sid: str,
    atlas: str,
    plane: str,
    submitted_mm: float,
    truth_mm: float,
    model: str = "gemini-3.1-pro-preview",
    kind: str = "single",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {
            "id": sid,
            "kind": kind,
            "atlas": atlas,
            "plane": plane,
            "truth_positions_mm": [truth_mm],
        },
        "submitted_positions_mm": [submitted_mm],
        "events": [],
        "model": model,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_best_traces_matches_original(tmp_path: Path):
    orig_module = _build_sft_corpus()
    from langslice_traces import best_traces as new_best_traces

    # Stage three raw_trace.json files across two run-roots and the
    # batched/flat layout variants.
    trace_dir = tmp_path / "_trace_root"
    # batched layout: <trace_dir>/runs_001/batch_001/runs/<id>/raw_trace.json
    _write_raw_trace(
        trace_dir / "runs_001" / "batch_001" / "runs" / "slice-a" / "raw_trace.json",
        sid="slice-a",
        atlas="allen_mouse_25um",
        plane="coronal",
        submitted_mm=6.10,
        truth_mm=6.00,
    )
    # flat layout: <trace_dir>/runs_002/runs/<id>/raw_trace.json
    _write_raw_trace(
        trace_dir / "runs_002" / "runs" / "slice-a" / "raw_trace.json",
        sid="slice-a",
        atlas="allen_mouse_25um",
        plane="coronal",
        submitted_mm=6.02,
        truth_mm=6.00,
    )
    _write_raw_trace(
        trace_dir / "runs_002" / "runs" / "slice-b" / "raw_trace.json",
        sid="slice-b",
        atlas="allen_mouse_25um",
        plane="coronal",
        submitted_mm=6.40,
        truth_mm=6.00,
    )

    orig = orig_module.best_traces(trace_dir)
    new = new_best_traces(trace_dir)

    assert set(orig.keys()) == set(new.keys())
    for sid in orig:
        o = orig[sid]
        n = new[sid]
        for k in ("err", "tol", "rescue", "tier", "sub_canonical_mm", "manifest",
                  "events", "model"):
            assert o[k] == n[k], f"mismatch on {sid}.{k}: {o[k]!r} vs {n[k]!r}"
        # raw_path is a Path; equality compares value
        assert o["raw_path"] == n["raw_path"]

    # Also exercise strict_only path
    orig_strict = orig_module.best_traces(trace_dir, strict_only=True)
    new_strict = new_best_traces(trace_dir, strict_only=True)
    assert set(orig_strict.keys()) == set(new_strict.keys())
