"""End-to-end smoke for the unified single-turn RL pipeline (Task 6).

This is the integration-level test the plan asks for: it builds the
adaptive curriculum stack end-to-end on a synthetic 50-row manifest and
verifies the wiring works without instantiating the heavy TRL
``GRPOTrainer``. Specifically it checks:

* :class:`ManifestIndex` loads from a synthetic shards/allocations tree
  and ``bulk_load_difficulty`` rehydrates per-section difficulty scores.
* :func:`build_datasets_from_index` (Lane B) yields :class:`SectionState`
  rows that round-trip into Lane B-shaped row specs.
* :class:`CurriculumRepeatingSampler` selects in-band indices over the
  resulting dataset.
* :class:`AdaptiveRewardSchedule` returns the static fallback before
  warmup and quantile-derived values after warmup.
* :class:`AdaRFTCurriculumCallback` ``on_log`` advances ``T`` and (when
  ``write_back=True``) calls :meth:`ManifestIndex.update_difficulty`.
* :class:`_DifficultyPersistCallback` triggers
  :meth:`ManifestIndex.persist_live_difficulty` every N ticks.
* :func:`_select_trainer_cls` produces the correct mixin class for each
  combination without importing TRL (the test stubs the lookup).
* The unified-pipeline CLI parser accepts the documented flag set and
  enforces the cross-flag mutex rules.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from single_turn_rl import adaptive_reward as ar
from single_turn_rl import dataset as ds
from single_turn_rl import section_state as ss
from single_turn_rl import train_grpo as tg
from single_turn_rl.curriculum import (
    AdaRFTCurriculumCallback,
    CurriculumRepeatingSampler,
)
from single_turn_rl.manifest_index import ManifestIndex

# ---------------------------------------------------------------------------
# Synthetic manifest builder (mirrors test_section_state)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _build_50_row_manifest(tmp_path: Path) -> Path:
    """Build a 50-row synthetic manifest under ``tmp_path``.

    Layout: 1 plane (coronal) × 2 datasets × 5 subjects × 5 sections each =
    50 rows. Positions are a deterministic ramp so the band sampler has
    clean candidates to choose from.
    """
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"
    plane = "coronal"
    n_datasets = 2
    n_subjects = 5
    n_sections = 5
    for d_idx in range(n_datasets):
        dataset = f"ds_{d_idx}"
        rows: list[dict[str, Any]] = []
        for s_idx in range(n_subjects):
            for sec_idx in range(n_sections):
                pos = 0.5 + 0.25 * (s_idx * n_sections + sec_idx)
                rows.append(
                    {
                        "image_path": (
                            f"data/datasets/{plane}/{dataset}/subj_{s_idx}/"
                            f"sec_{sec_idx}.png"
                        ),
                        "dataset": dataset,
                        "subject_id": f"subj_{d_idx}_{s_idx}",
                        "section_id": f"{dataset}_subj{s_idx}_sec{sec_idx}",
                        "species": "mouse",
                        "atlas": "allen_mouse_25um",
                        "orientation": plane,
                        "slice_axis": "ap",
                        "position_mm": pos,
                        "imaging": "brightfield",
                        "staining": "Nissl",
                        "exclude_from_training": False,
                    }
                )
        _write_jsonl(shards_root / plane / f"{dataset}.jsonl", rows)

    alloc_root = manifest_root / "allocations"
    # Allocate every section to the rlvr split so build_datasets_from_index
    # has rows to yield under its default split filter.
    rlvr_rows: list[dict[str, Any]] = []
    for d_idx in range(n_datasets):
        dataset = f"ds_{d_idx}"
        for s_idx in range(n_subjects):
            for sec_idx in range(n_sections):
                rlvr_rows.append(
                    {
                        "section_id": f"{dataset}_subj{s_idx}_sec{sec_idx}",
                        "dataset": dataset,
                        "added_by": "smoke",
                        "added_at": "2026-05-10T00:00:00+00:00",
                    }
                )
    _write_jsonl(alloc_root / plane / "rlvr.jsonl", rlvr_rows)
    _write_jsonl(alloc_root / plane / "sft.jsonl", [])
    _write_jsonl(alloc_root / plane / "eval.jsonl", [])
    return manifest_root


def _stub_atlas_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading the real BrainGlobe atlas during the smoke."""
    monkeypatch.setattr(
        ss,
        "_atlas_valid_range_mm",
        lambda atlas, plane: {
            ("allen_mouse_25um", "coronal"): (0.0, 13.2),
            ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
            ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
        }[(atlas, plane)],
    )
    # The dataset-level row builder also looks up the atlas long-edge —
    # stub that too so we don't need the SFT-corpus fixtures.
    monkeypatch.setattr(ds, "_atlas_in_plane_long_edge", lambda _atlas, _plane: 320)
    monkeypatch.setattr(ds, "species_from_atlas_name", lambda _name: "mouse")
    monkeypatch.setattr(ds, "canonicalize_atlas_name", lambda name: name)


def _seed_query_files(base: Path, index: ManifestIndex, plane: str) -> None:
    for section in index.query(plane=plane):
        target = base / section.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


def _seed_atlas_files(base: Path, slate: ss.CanonicalSlate) -> None:
    for rel in slate.image_paths:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.fixture
def synthetic_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[ManifestIndex, Path]:
    _stub_atlas_range(monkeypatch)
    manifest_root = _build_50_row_manifest(tmp_path)
    index = ManifestIndex.from_manifest_root(manifest_root, repo_root=tmp_path)
    _seed_query_files(tmp_path, index, "coronal")
    # Seed the canonical slate's atlas image stubs so iter_section_states
    # doesn't drop everything via the require_atlas_on_disk guard.
    slate = ss.build_canonical_slate("allen_mouse_25um", "coronal", n_positions=5)
    _seed_atlas_files(tmp_path, slate)
    return index, manifest_root


@pytest.fixture(autouse=True)
def _isolate_recent_errors():
    """Each test starts with an empty rolling-error deque (and clears after)."""
    ar.clear_recent_errors()
    yield
    ar.clear_recent_errors()


# ---------------------------------------------------------------------------
# 1. ManifestIndex loads + bulk_load_difficulty round-trips
# ---------------------------------------------------------------------------


def test_manifest_index_loads_and_seeds(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    index, manifest_root = synthetic_index
    assert len(index) == 50
    # All in the rlvr split per the allocation seed.
    rlvr_sections = index.query(split="rlvr")
    assert len(rlvr_sections) == 50

    # Write a seed file mirroring the seed-difficulty training tool output.
    seed_rows = []
    for section in rlvr_sections[:10]:
        seed_rows.append(
            {
                "plane": section.plane,
                "dataset": section.dataset,
                "section_id": section.section_id,
                "difficulty_score": 0.42,
                "source": "slicebench_seed",
            }
        )
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({"rows": seed_rows}), encoding="utf-8")
    n_loaded = index.bulk_load_difficulty(seed_path)
    assert n_loaded == 10
    seeded = index.live_difficulty(
        rlvr_sections[0].plane, rlvr_sections[0].dataset, rlvr_sections[0].section_id,
    )
    assert seeded is not None
    assert seeded.score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 2. Lane B dataset builder yields SectionState-derived rows
# ---------------------------------------------------------------------------


def test_build_datasets_from_index_yields_lane_b_rows(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    index, _ = synthetic_index
    train, evald = ds.build_datasets_from_index(
        manifest_index=index,
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        n_positions=5,
        eval_holdout_every=5,
        seed=0,
    )
    assert len(train) > 0
    # Every spec carries the new Lane B fields.
    for spec in train._specs:  # noqa: SLF001
        assert "dataset" in spec
        assert spec["dataset"].startswith("ds_")
        # Lane B specs: difficulty_score is None at cold start (no seed).
        assert spec["difficulty_score"] is None
        # Slate paths landed in the spec (renamed to atlas_image_paths in
        # the spec for shape parity with Lane A).
        assert len(spec["atlas_image_paths"]) == 5
        assert len(spec["fetched_positions_mm"]) == 5
    # 10 distinct subjects (2 datasets × 5 subjects); eval_holdout_every=5 → 2
    # eval subjects (every 5th in sorted order). Each has 5 sections → 10 rows.
    assert evald is not None
    assert len(evald) > 0
    assert len(evald) <= len(train) // 2 + len(train)  # sanity upper bound
    # No subject-level leakage between train and eval.
    train_subjects = {s["subject_id"] for s in train._specs}  # noqa: SLF001
    eval_subjects = {s["subject_id"] for s in evald._specs}  # noqa: SLF001
    assert train_subjects.isdisjoint(eval_subjects)


# ---------------------------------------------------------------------------
# 3. RowDataset row builder produces a row from a SectionState
# ---------------------------------------------------------------------------


def test_row_builder_consumes_section_state_round_trip(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    """``_spec_from_section_state`` produces a spec that ``_build_row_from_spec``
    can fully consume — verified without decoding actual images by stubbing
    the image-loader helpers."""
    index, _ = synthetic_index
    section = index.query(plane="coronal", split="rlvr")[0]
    slate = ss.build_canonical_slate("allen_mouse_25um", "coronal", n_positions=3)
    state = ss.SectionState.from_section(section, slate=slate, difficulty_score=0.7)
    spec = ds._spec_from_section_state(state, repo_root=tmp_path)
    assert spec["section_id"] == section.section_id
    assert spec["dataset"] == section.dataset
    assert spec["difficulty_score"] == pytest.approx(0.7)
    assert spec["atlas_image_paths"] == slate.image_paths
    assert spec["fetched_positions_mm"] == slate.positions_mm


# ---------------------------------------------------------------------------
# 4. CurriculumRepeatingSampler returns in-band indices over Lane B specs
# ---------------------------------------------------------------------------


def test_curriculum_sampler_selects_in_band_over_lane_b(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    index, _ = synthetic_index
    # Stamp difficulty on a band of sections so rung 1 succeeds.
    sections = index.query(plane="coronal", split="rlvr")
    for sec in sections[:20]:
        index.update_difficulty(sec.plane, sec.dataset, sec.section_id, 0.5, "test")
    for sec in sections[20:30]:
        index.update_difficulty(sec.plane, sec.dataset, sec.section_id, 0.9, "test")

    train, _ = ds.build_datasets_from_index(
        manifest_index=index,
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        n_positions=3,
        eval_holdout_every=0,
        seed=0,
    )
    sampler = CurriculumRepeatingSampler(
        train,
        num_generations=2,
        batch_size=4,
        initial_T=0.5,
        band_width=0.1,
        max_per_subject_in_batch=2,
        generator=torch.Generator().manual_seed(0),
    )
    indices = list(iter(sampler))
    assert len(indices) == 4
    # Every picked index must lie in [T - band, T + band] OR be a cold-start
    # row (which the sampler treats as if its difficulty == T).
    for idx in indices:
        d = sampler._difficulty_for_index(idx)  # noqa: SLF001
        assert 0.4 <= d <= 0.6, f"idx {idx} d={d} outside band"


# ---------------------------------------------------------------------------
# 5. AdaptiveRewardSchedule warmup + steady state
# ---------------------------------------------------------------------------


def test_adaptive_schedule_returns_static_before_warmup() -> None:
    schedule = ar.AdaptiveRewardSchedule(
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_observations=50,
        static_fallback=(0.05, 0.15),
    )
    # Empty buffer → fallback.
    assert schedule.current() == (0.05, 0.15)


def test_adaptive_schedule_returns_quantile_after_warmup() -> None:
    schedule = ar.AdaptiveRewardSchedule(
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_observations=10,
        static_fallback=(0.05, 0.15),
    )
    # Plant 10 observations with abs_err = axis_span * 0.02 (so error_frac = 0.02).
    for _ in range(10):
        ar.record_error(
            abs_err_mm=0.02 * 13.2,
            axis_span_mm=13.2,
            plane="coronal",
            dataset="ds_0",
            section_id="ds_0_subj0_sec0",
        )
    sigma_frac, cutoff_frac = schedule.current()
    # Median quantile = 0.02 → clamped to [0.005, 0.05].
    assert sigma_frac == pytest.approx(0.02, abs=1e-9)
    assert cutoff_frac >= sigma_frac


# ---------------------------------------------------------------------------
# 6. AdaRFTCurriculumCallback advances T and writes back when wired
# ---------------------------------------------------------------------------


def test_adaRFT_callback_surfaces_metrics_to_log_history(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    """The callback must write adarft/* keys into both ``logs`` and
    ``state.log_history[-1]`` so trackio/wandb pick them up.

    Without this surface, operators could only see curriculum starvation
    via the ``logger.info`` line — useless for trackio/wandb dashboards.
    """
    index, _ = synthetic_index
    train, _ = ds.build_datasets_from_index(
        manifest_index=index,
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        n_positions=3,
        eval_holdout_every=0,
        seed=0,
    )
    sampler = CurriculumRepeatingSampler(
        train,
        num_generations=2,
        batch_size=4,
        initial_T=0.5,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        band_width=0.5,
        max_per_subject_in_batch=8,
        generator=torch.Generator().manual_seed(0),
    )
    schedule = ar.AdaptiveRewardSchedule(min_observations=1)
    callback = AdaRFTCurriculumCallback(
        sampler=sampler,
        manifest_index=index,
        schedule=schedule,
        write_back=False,
    )
    logs = {"reward": 0.8}
    state = SimpleNamespace(log_history=[{"reward": 0.8, "step": 1}])
    callback.on_log(args=None, state=state, control=None, logs=logs)

    expected_keys = {
        "adarft/T",
        "adarft/raw_step",
        "adarft/smoothed_step",
        "adarft/ladder_level",
        "adarft/normalized_reward",
    }
    # Metrics must be visible to the live wandb/trackio path (the ``logs`` dict).
    assert expected_keys.issubset(set(logs.keys()))
    # And to log_history readers (the appended entry).
    assert expected_keys.issubset(set(state.log_history[-1].keys()))
    # Sanity: T was advanced and surfaced; reward (0.8) > target (0.5) ⇒ T > 0.5.
    assert state.log_history[-1]["adarft/T"] > 0.5
    # Normalized reward clamps to [0, 1]; 0.8 passes through unchanged.
    assert state.log_history[-1]["adarft/normalized_reward"] == pytest.approx(0.8)


def test_adaRFT_callback_advances_T_and_writes_back(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end wiring: callback's ``on_log`` advances sampler.T and pushes
    per-section difficulty back into the manifest index."""
    index, _ = synthetic_index
    train, _ = ds.build_datasets_from_index(
        manifest_index=index,
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        n_positions=3,
        eval_holdout_every=0,
        seed=0,
    )
    sampler = CurriculumRepeatingSampler(
        train,
        num_generations=2,
        batch_size=4,
        initial_T=0.5,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        band_width=0.5,  # wide enough to always succeed at rung 1
        max_per_subject_in_batch=8,
        generator=torch.Generator().manual_seed(0),
    )
    schedule = ar.AdaptiveRewardSchedule(
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
        min_observations=1,
        static_fallback=(0.05, 0.15),
    )

    # Plant a per-section observation so the write-back path produces
    # something to write.
    sample_section = train._specs[0]  # noqa: SLF001
    ar.record_error(
        abs_err_mm=0.01 * 13.2,
        axis_span_mm=13.2,
        plane=sample_section["plane"],
        dataset=sample_section["dataset"],
        section_id=sample_section["section_id"],
    )

    callback = AdaRFTCurriculumCallback(
        sampler=sampler,
        manifest_index=index,
        schedule=schedule,
        write_back=True,
    )
    initial_T = sampler.T
    state = SimpleNamespace(log_history=[{"reward": 1.0, "step": 1}])
    callback.on_log(args=None, state=state, control=None)
    # T moved up because reward (1.0) > target (0.5).
    assert sampler.T > initial_T
    # And the manifest index now carries a live-difficulty observation for
    # the planted section.
    written = index.live_difficulty(
        sample_section["plane"],
        sample_section["dataset"],
        sample_section["section_id"],
    )
    assert written is not None


def test_adaRFT_callback_skips_write_back_when_disabled(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    index, _ = synthetic_index
    train, _ = ds.build_datasets_from_index(
        manifest_index=index,
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        n_positions=3,
        eval_holdout_every=0,
        seed=0,
    )
    sampler = CurriculumRepeatingSampler(
        train,
        num_generations=2,
        batch_size=4,
        initial_T=0.5,
        band_width=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    schedule = ar.AdaptiveRewardSchedule(min_observations=1)

    sample_section = train._specs[0]  # noqa: SLF001
    ar.record_error(
        abs_err_mm=0.01 * 13.2,
        axis_span_mm=13.2,
        plane=sample_section["plane"],
        dataset=sample_section["dataset"],
        section_id=sample_section["section_id"],
    )

    callback = AdaRFTCurriculumCallback(
        sampler=sampler,
        manifest_index=index,
        schedule=schedule,
        write_back=False,
    )
    state = SimpleNamespace(log_history=[{"reward": 1.0}])
    callback.on_log(args=None, state=state, control=None)
    # T moved (sampler.update was called) but no live_difficulty entry yet.
    assert (
        index.live_difficulty(
            sample_section["plane"],
            sample_section["dataset"],
            sample_section["section_id"],
        )
        is None
    )


# ---------------------------------------------------------------------------
# 7. _DifficultyPersistCallback persists every N ticks
# ---------------------------------------------------------------------------


def test_difficulty_persist_callback_fires_on_cadence(
    synthetic_index: tuple[ManifestIndex, Path], tmp_path: Path,
) -> None:
    index, _ = synthetic_index
    # Plant some difficulty so persist has something to write.
    sec = index.query(plane="coronal", split="rlvr")[0]
    index.update_difficulty(sec.plane, sec.dataset, sec.section_id, 0.42, "test")

    callback = tg._DifficultyPersistCallback(
        manifest_index=index, every_n_ticks=3,
    )
    state = SimpleNamespace(log_history=[])
    # Ticks 1 and 2 don't fire; tick 3 does.
    sidecar = index._default_live_difficulty_path()  # noqa: SLF001
    assert not sidecar.exists()
    callback.on_log(args=None, state=state, control=None)
    callback.on_log(args=None, state=state, control=None)
    assert not sidecar.exists()
    callback.on_log(args=None, state=state, control=None)
    assert sidecar.is_file()


# ---------------------------------------------------------------------------
# 8. CLI mutex enforcement
# ---------------------------------------------------------------------------


def test_cli_mutex_adaptive_curriculum_conflicts_with_static_weights(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    weights_path = tmp_path / "weights.json"
    weights_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        tg._parse_args(
            [
                "--config", str(cfg_path),
                "--sft-model", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--curriculum-mode", "adaptive",
                "--curriculum-weights", str(weights_path),
            ]
        )


def test_cli_mutex_static_weights_requires_weights(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        tg._parse_args(
            [
                "--config", str(cfg_path),
                "--sft-model", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--curriculum-mode", "static_weights",
            ]
        )


def test_cli_mutex_terminal_states_required_with_terminal_data_source(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        tg._parse_args(
            [
                "--config", str(cfg_path),
                "--sft-model", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--data-source", "terminal_states",
            ]
        )


def test_cli_mutex_index_data_source_rejects_static_weights_curriculum(
    tmp_path: Path,
) -> None:
    """``--data-source index`` + ``--curriculum-mode static_weights`` is invalid.

    static_weights expects a WeightedRowDataset built only by the
    terminal-states path; combining it with index-driven data would crash
    later at an isinstance assert with a bare AssertionError. The CLI
    mutex must surface a usable error here.
    """
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    weights_path = tmp_path / "weights.json"
    weights_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        tg._parse_args(
            [
                "--config", str(cfg_path),
                "--sft-model", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--data-source", "index",
                "--curriculum-mode", "static_weights",
                "--curriculum-weights", str(weights_path),
            ]
        )


def test_cli_defaults_unified_pipeline_modes(tmp_path: Path) -> None:
    """Documented defaults: adaptive curriculum, adaptive reward, index data.

    The data-source default is ``index`` (Lane B): RL feeds from the full
    ~25k+ RLVR allocation via ManifestIndex. The terminal_states path stays
    available for runs against external trace-factory artifacts.
    """
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    args = tg._parse_args(
        [
            "--config", str(cfg_path),
            "--sft-model", str(tmp_path),
            "--output-dir", str(tmp_path),
        ]
    )
    assert args.curriculum_mode == "adaptive"
    assert args.reward_mode == "adaptive"
    assert args.data_source == "index"
    assert args.manifest_root == Path("data/manifest")


# ---------------------------------------------------------------------------
# 9. _resolve_adaptive_cfg merges CLI > TOML > defaults
# ---------------------------------------------------------------------------


def test_resolve_adaptive_cfg_cli_overrides_toml_overrides_default() -> None:
    # SimpleNamespace structurally satisfies argparse.Namespace's getattr-based
    # contract; cast through Any so basedpyright doesn't flag the
    # type-mismatch (the function body only reads attributes, never calls
    # Namespace-specific methods).
    args = cast(
        Any,
        SimpleNamespace(
            adaptive_target_reward=None,  # use TOML
            adaptive_alpha=4.0,           # CLI override
            adaptive_eta=None,
            adaptive_ema_decay=None,
            adaptive_band_width=None,
            max_per_subject_in_batch=None,
            adaptive_sigma_quantile=None,
            adaptive_cutoff_quantile=None,
            adaptive_min_sigma_frac=None,
            adaptive_max_sigma_frac=None,
            adaptive_min_observations=None,
            persist_difficulty_every_n_ticks=0,  # CLI override
        ),
    )
    toml_adaptive = {"target_reward": 0.7}
    out = tg._resolve_adaptive_cfg(args, toml_adaptive)
    assert out["target_reward"] == pytest.approx(0.7)         # TOML
    assert out["alpha"] == pytest.approx(4.0)                  # CLI
    assert out["eta"] == pytest.approx(0.05)                   # default
    assert out["persist_difficulty_every_n_ticks"] == 0        # CLI
    # Type coercion: persist key is an int.
    assert isinstance(out["persist_difficulty_every_n_ticks"], int)


# ---------------------------------------------------------------------------
# 10. _select_trainer_cls mixin layering (no TRL needed — the lookup is
# stubbed because pulling TRL into the smoke test would defeat the
# purpose of the "DOES NOT instantiate the actual TRL GRPOTrainer" rule)
# ---------------------------------------------------------------------------


class _FakeGRPOTrainer:
    """Stand-in for ``trl.GRPOTrainer`` — accepts any kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeCurriculumGRPOTrainer(_FakeGRPOTrainer):
    pass


def _stub_trl_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grpo_cls: type = _FakeGRPOTrainer,
    curriculum_cls: type = _FakeCurriculumGRPOTrainer,
) -> None:
    """Patch ``trl.GRPOTrainer`` and ``curriculum.sampler.CurriculumGRPOTrainer``
    inside ``_select_trainer_cls``'s import scope."""
    import sys

    fake_trl = SimpleNamespace(
        GRPOTrainer=grpo_cls,
        GRPOConfig=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    fake_curriculum_sampler_module = SimpleNamespace(
        CurriculumGRPOTrainer=curriculum_cls,
    )
    monkeypatch.setitem(
        sys.modules, "curriculum.sampler", fake_curriculum_sampler_module,
    )


def test_select_trainer_vanilla(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    cls = tg._select_trainer_cls(
        adarft_enabled=False,
        static_curriculum_enabled=False,
        atlas_cache=None,
    )
    assert cls is _FakeGRPOTrainer


def test_select_trainer_static_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    cls = tg._select_trainer_cls(
        adarft_enabled=False,
        static_curriculum_enabled=True,
        atlas_cache=None,
    )
    assert cls is _FakeCurriculumGRPOTrainer


def test_select_trainer_adarft_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    cls = tg._select_trainer_cls(
        adarft_enabled=True,
        static_curriculum_enabled=False,
        atlas_cache=None,
    )
    # AdaRFT mixin layered on top of vanilla GRPOTrainer.
    assert issubclass(cls, _FakeGRPOTrainer)
    assert cls is not _FakeGRPOTrainer
    # Constructing it stashes the curriculum sampler.
    instance = cls(curriculum_sampler="sentinel")
    assert instance._curriculum_sampler == "sentinel"  # type: ignore[attr-defined]


def test_select_trainer_splice_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    fake_cache = MagicMock()
    cls = tg._select_trainer_cls(
        adarft_enabled=False,
        static_curriculum_enabled=False,
        atlas_cache=fake_cache,
    )
    assert issubclass(cls, _FakeGRPOTrainer)
    assert cls is not _FakeGRPOTrainer
    # Instantiating verifies __init__ pops atlas_cache and wraps processing_class.
    fake_processor = MagicMock()
    instance = cls(atlas_cache=fake_cache, processing_class=fake_processor)
    assert instance._atlas_cache is fake_cache  # type: ignore[attr-defined]
    # processing_class must have been replaced with a sidecar-emitting proxy.
    assert instance.kwargs["processing_class"] is not fake_processor  # type: ignore[attr-defined]


def test_select_trainer_splice_plus_adarft(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    fake_cache = MagicMock()
    cls = tg._select_trainer_cls(
        adarft_enabled=True,
        static_curriculum_enabled=False,
        atlas_cache=fake_cache,
    )
    assert issubclass(cls, _FakeGRPOTrainer)
    # Both mixins applied.
    assert cls is not _FakeGRPOTrainer
    # Instantiating verifies both the splice and adarft __init__ paths fire.
    fake_processor = MagicMock()
    instance = cls(
        atlas_cache=fake_cache,
        processing_class=fake_processor,
        curriculum_sampler="sentinel",
    )
    assert instance._atlas_cache is fake_cache  # type: ignore[attr-defined]
    assert instance._curriculum_sampler == "sentinel"  # type: ignore[attr-defined]
    assert instance.kwargs["processing_class"] is not fake_processor  # type: ignore[attr-defined]


def test_select_trainer_splice_plus_static(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_trl_imports(monkeypatch)
    fake_cache = MagicMock()
    cls = tg._select_trainer_cls(
        adarft_enabled=False,
        static_curriculum_enabled=True,
        atlas_cache=fake_cache,
    )
    # Splice wraps CurriculumGRPOTrainer, which is _FakeCurriculumGRPOTrainer here.
    assert issubclass(cls, _FakeCurriculumGRPOTrainer)
    assert cls is not _FakeCurriculumGRPOTrainer
    # Instantiating verifies the splice __init__ fired (atlas_cache stashed,
    # processing_class wrapped) while static-curriculum base is in the MRO.
    fake_processor = MagicMock()
    instance = cls(atlas_cache=fake_cache, processing_class=fake_processor)
    assert instance._atlas_cache is fake_cache  # type: ignore[attr-defined]
    assert instance.kwargs["processing_class"] is not fake_processor  # type: ignore[attr-defined]


def test_select_trainer_rejects_double_curriculum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI mutex catches this earlier; the helper still defends defensively."""
    _stub_trl_imports(monkeypatch)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tg._select_trainer_cls(
            adarft_enabled=True,
            static_curriculum_enabled=True,
            atlas_cache=None,
        )


# ---------------------------------------------------------------------------
# 11. "no NaN rewards" — adaptive terminal reward is always finite
# ---------------------------------------------------------------------------


def test_adaptive_terminal_reward_never_returns_nan() -> None:
    """Every branch of the adaptive reward returns a finite float.

    Covers: clean parse + in-range hit, out-of-range numeric, and format
    failure (unparseable completion). This is the literal "no NaN rewards"
    assertion from verification item 6.
    """
    reward_fn = ar.make_adaptive_terminal_reward(
        min_observations=50,  # stays in static-fallback mode for this call
        static_fallback=(0.05, 0.15),
    )

    # Four completions exercising all code branches:
    #   [0] clean parse, in-range (bell path)
    #   [1] clean parse, out-of-range
    #   [2] clean parse, in-range but at the boundary (edge of bell)
    #   [3] format failure (unparseable)
    completions = [
        '{"position_mm": 6.0}',   # in-range: GT=6.0, range (0.0, 13.2)
        '{"position_mm": 99.0}',  # out-of-range
        '{"position_mm": 0.01}',  # barely in-range, far from GT
        "not json",               # parse failure → format_penalty
    ]
    ground_truth_mm = [6.0, 6.0, 6.0, 6.0]
    valid_range_mm = [
        (0.0, 13.2),
        (0.0, 13.2),
        (0.0, 13.2),
        (0.0, 13.2),
    ]

    rewards = reward_fn(
        completions=completions,
        ground_truth_mm=ground_truth_mm,
        valid_range_mm=valid_range_mm,
    )

    assert len(rewards) == 4
    for i, r in enumerate(rewards):
        assert math.isfinite(r), (
            f"reward[{i}] is not finite: {r!r}"
        )


# ---------------------------------------------------------------------------
# 12. "sigma narrows over the run" — AdaptiveRewardSchedule tracks improvement
# ---------------------------------------------------------------------------


def test_adaptive_sigma_narrows_as_errors_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sigma shrinks when the rolling buffer transitions from large to small errors.

    Phase 1: fill the deque with large error fracs (0.10) — sigma lands near max.
    Phase 2: overflow with small error fracs (0.01) — sigma should be strictly
    smaller. This is the literal "sigma narrows over the run" assertion from
    verification item 6.

    The _isolate_recent_errors autouse fixture clears the buffer before and
    after every test, so this test doesn't need its own monkeypatch teardown.
    """
    schedule = ar.AdaptiveRewardSchedule(
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_observations=50,
        static_fallback=(0.05, 0.15),
    )

    axis_span_mm = 10.0  # constant; vary abs_err_mm to control error_frac

    # Phase 1: 60 observations with error_frac = 0.03 (large, within [min, max]
    # so the clamp does not flatten the difference). The median lands at 0.03,
    # which is between min_sigma_frac (0.005) and max_sigma_frac (0.05).
    for i in range(60):
        ar.record_error(
            abs_err_mm=0.03 * axis_span_mm,
            axis_span_mm=axis_span_mm,
            plane="coronal",
            dataset="ds_0",
            section_id=f"sec_{i}",
        )
    sigma_frac_a, cutoff_frac_a = schedule.current()

    # Phase 2: 60 more observations with error_frac = 0.007 (small, above
    # min_sigma_frac so the lower clamp doesn't hide the difference either).
    # The deque maxlen is 200 by default, so 120 total fit; the small-error
    # batch dominates the median once added.
    for i in range(60):
        ar.record_error(
            abs_err_mm=0.007 * axis_span_mm,
            axis_span_mm=axis_span_mm,
            plane="coronal",
            dataset="ds_0",
            section_id=f"sec_{i}",
        )
    sigma_frac_b, cutoff_frac_b = schedule.current()

    # Sigma must narrow as errors shrink.
    assert sigma_frac_b < sigma_frac_a, (
        f"Expected sigma to narrow: phase-1 sigma={sigma_frac_a}, "
        f"phase-2 sigma={sigma_frac_b}"
    )

    # Both readings must stay within the configured bounds.
    assert schedule.min_sigma_frac <= sigma_frac_a <= schedule.max_sigma_frac
    assert schedule.min_sigma_frac <= sigma_frac_b <= schedule.max_sigma_frac
