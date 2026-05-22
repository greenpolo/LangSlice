"""Unit tests for ``single_turn_rl.train_grpo`` plumbing.

Heavy TRL/Unsloth imports are deferred inside ``main()`` so these tests can
exercise the pure-Python helpers (sidecar builder, processor wrapper,
curriculum reward log, atlas-cache coverage validator) without instantiating
the trainer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from langslice_training.rl.single_turn import dataset as ds
from langslice_training.rl.single_turn import train_grpo as tg
from langslice_training.rl.single_turn.terminal_states import TerminalState

# --- Test fixtures --------------------------------------------------------


def _stub_atlas_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_atlas_in_plane_long_edge", lambda _name, _plane: 320)
    monkeypatch.setattr(ds, "species_from_atlas_name", lambda _name: "mouse")
    monkeypatch.setattr(ds, "canonicalize_atlas_name", lambda name: name)


def _spec(
    *,
    section_id: str = "sec_a",
    subject_id: str = "subj_1",
    atlas_name: str = "allen_mouse_25um",
    plane: str = "coronal",
    ground_truth_mm: float = 4.37,
) -> dict[str, Any]:
    state = TerminalState(
        section_id=section_id,
        subject_id=subject_id,
        atlas_name=atlas_name,
        plane=plane,  # type: ignore[arg-type]
        valid_range_mm=(0.0, 13.2),
        ground_truth_mm=ground_truth_mm,
        query_image_path=f"queries/{section_id}.jpg",
        atlas_image_paths=("atlas/x/coronal/4.00mm.jpg",),
        fetched_positions_mm=(4.0,),
        source="sft_corpus:strict",
        quality={},
    )
    return ds._spec_from_state(state, repo_root=Path("/repo"))


# --- _atlas_pairs_in_dataset / _validate_atlas_cache_coverage ------------


def test_atlas_pairs_in_dataset_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_atlas_helpers(monkeypatch)
    specs = [
        _spec(section_id="a", subject_id="s1", atlas_name="A1", plane="coronal"),
        _spec(section_id="b", subject_id="s2", atlas_name="A1", plane="coronal"),
        _spec(section_id="c", subject_id="s3", atlas_name="A2", plane="sagittal"),
    ]
    rd = ds.RowDataset(specs)
    assert tg._atlas_pairs_in_dataset(rd) == {("A1", "coronal"), ("A2", "sagittal")}


def test_validate_atlas_cache_coverage_warns_on_missing_pair(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    rd = ds.RowDataset([
        _spec(section_id="a", subject_id="s1", atlas_name="A1", plane="coronal"),
        _spec(section_id="b", subject_id="s2", atlas_name="A2", plane="sagittal"),
    ])
    cache = MagicMock()
    cache.pairs.return_value = [("A1", "coronal")]  # missing A2/sagittal
    with caplog.at_level(logging.WARNING, logger="single_turn_rl.train_grpo"):
        tg._validate_atlas_cache_coverage(cache, rd)
    assert any("missing" in r.message.lower() for r in caplog.records)


def test_validate_atlas_cache_coverage_silent_when_full(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    rd = ds.RowDataset([_spec(atlas_name="A1", plane="coronal")])
    cache = MagicMock()
    cache.pairs.return_value = [("A1", "coronal"), ("Extra", "sagittal")]
    with caplog.at_level(logging.WARNING, logger="single_turn_rl.train_grpo"):
        tg._validate_atlas_cache_coverage(cache, rd)
    assert not any("missing" in r.message.lower() for r in caplog.records)


# --- _build_sidecars ------------------------------------------------------


class _StubCache:
    """Minimal AtlasEmbeddingCache stub keyed by exact path string."""

    def __init__(self, hits: dict[str, torch.Tensor]) -> None:
        self._hits = hits

    def lookup_by_path(self, path: str) -> torch.Tensor | None:
        return self._hits.get(path)


def test_build_sidecars_returns_none_when_no_cache_hits() -> None:
    cache = _StubCache({})
    out = tg._build_sidecars(
        images=[["pil_a", "pil_b"]],
        paths_per_row=[["q.jpg", "atlas/a.jpg"]],
        cache=cache,
    )
    assert out is None


def test_build_sidecars_emits_correct_mask_and_flat() -> None:
    """Two atlas hits, one query miss → mask=[F,T,T], flat = 2*P×H, pc=[P,P]."""
    hidden = 4
    patches_per = 3
    emb_a = torch.arange(patches_per * hidden, dtype=torch.float32).reshape(patches_per, hidden)
    emb_b = (emb_a + 100).clone()
    cache = _StubCache({
        "atlas/x/coronal/2.00mm.jpg": emb_a,
        "atlas/x/coronal/3.00mm.jpg": emb_b,
    })
    sidecars = tg._build_sidecars(
        images=[["pil_q", "pil_a", "pil_b"]],
        paths_per_row=[["queries/q.jpg",
                        "atlas/x/coronal/2.00mm.jpg",
                        "atlas/x/coronal/3.00mm.jpg"]],
        cache=cache,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    assert mask.tolist() == [False, True, True]
    assert cached_flat.shape == (2 * patches_per, hidden)
    # cached_flat is the concatenation in mask-True order (a then b).
    assert torch.equal(cached_flat[:patches_per], emb_a)
    assert torch.equal(cached_flat[patches_per:], emb_b)
    assert cached_pc.tolist() == [patches_per, patches_per]


def test_build_sidecars_peels_singleton_batch_dim() -> None:
    """Cache files store ``(1, P, hidden)``; sidecar must peel the leading 1."""
    hidden = 4
    patches_per = 3
    emb_3d = torch.arange(patches_per * hidden, dtype=torch.float32).reshape(1, patches_per, hidden)
    cache = _StubCache({"atlas/a.jpg": emb_3d})
    sidecars = tg._build_sidecars(
        images=[["pil_a"]], paths_per_row=[["atlas/a.jpg"]], cache=cache,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    assert mask.tolist() == [True]
    assert cached_flat.shape == (patches_per, hidden)
    assert cached_pc.tolist() == [patches_per]


def test_build_sidecars_raises_on_alignment_mismatch() -> None:
    cache = _StubCache({})
    with pytest.raises(ValueError, match="alignment"):
        tg._build_sidecars(
            images=[["pil_a", "pil_b"]],
            paths_per_row=[["only_one.jpg"]],
            cache=cache,
        )


def test_build_sidecars_handles_multi_row_batches() -> None:
    """Per-row image lists flatten in nested order."""
    hidden = 2
    patches_per = 1
    emb = torch.tensor([[1.0, 2.0]])
    cache = _StubCache({"r1_atlas.jpg": emb, "r2_atlas.jpg": emb})
    sidecars = tg._build_sidecars(
        images=[["pil_q1", "pil_a1"], ["pil_q2", "pil_a2"]],
        paths_per_row=[["r1_q.jpg", "r1_atlas.jpg"], ["r2_q.jpg", "r2_atlas.jpg"]],
        cache=cache,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    assert mask.tolist() == [False, True, False, True]
    assert cached_flat.shape == (2 * patches_per, hidden)


def test_build_sidecars_skips_none_rows() -> None:
    """A row with no images at all (None) just contributes nothing to the flat list."""
    cache = _StubCache({"a.jpg": torch.ones((1, 2))})
    sidecars = tg._build_sidecars(
        images=[None, ["pil_a"]],
        paths_per_row=[[], ["a.jpg"]],
        cache=cache,
    )
    assert sidecars is not None
    mask, _, _ = sidecars
    assert mask.tolist() == [True]


# --- _SidecarEmittingProcessor --------------------------------------------


class _BatchEncoding(dict):
    """Stand-in for transformers.BatchEncoding — supports __setitem__."""


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = MagicMock(name="tokenizer")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _BatchEncoding:
        self.calls.append(dict(kwargs))
        return _BatchEncoding({"pixel_values": torch.zeros((1, 3, 4, 4))})

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return "rendered template"


def test_sidecar_processor_delegates_attribute_access() -> None:
    base = _StubProcessor()
    proxy = tg._SidecarEmittingProcessor(
        base=base, cache=_StubCache({}), paths_provider=lambda: [],
    )
    # tokenizer attribute pass-through
    assert proxy.tokenizer is base.tokenizer
    # method pass-through
    assert proxy.apply_chat_template() == "rendered template"


def test_sidecar_processor_returns_base_output_when_no_images() -> None:
    base = _StubProcessor()
    proxy = tg._SidecarEmittingProcessor(
        base=base, cache=_StubCache({}), paths_provider=lambda: [],
    )
    out = proxy(text="some prompt")
    assert "precomputed_image_mask" not in out
    assert "precomputed_cached_flat" not in out


def test_sidecar_processor_returns_base_output_when_no_hits() -> None:
    base = _StubProcessor()
    proxy = tg._SidecarEmittingProcessor(
        base=base, cache=_StubCache({}), paths_provider=lambda: [["q.jpg"]],
    )
    out = proxy(images=[["pil_q"]], text="prompt")
    assert "precomputed_image_mask" not in out


def test_sidecar_processor_appends_sidecars_when_hits_exist() -> None:
    base = _StubProcessor()
    emb = torch.ones((1, 2, 4))
    proxy = tg._SidecarEmittingProcessor(
        base=base, cache=_StubCache({"atlas/a.jpg": emb}),
        paths_provider=lambda: [["q.jpg", "atlas/a.jpg"]],
    )
    out = proxy(images=[["pil_q", "pil_a"]], text="prompt")
    assert "precomputed_image_mask" in out
    assert "precomputed_cached_flat" in out
    assert "precomputed_cached_patch_counts" in out
    assert out["precomputed_image_mask"].tolist() == [False, True]


def test_sidecar_processor_swallows_alignment_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If paths/images misalign, the proxy logs a warning and returns base output."""
    base = _StubProcessor()
    proxy = tg._SidecarEmittingProcessor(
        base=base, cache=_StubCache({"atlas/a.jpg": torch.ones((1, 2, 4))}),
        paths_provider=lambda: [["only_one.jpg"]],  # 1 path
    )
    with caplog.at_level(logging.WARNING, logger="single_turn_rl.train_grpo"):
        out = proxy(images=[["pil_a", "pil_b"]], text="prompt")  # 2 images
    assert "precomputed_image_mask" not in out


# --- _wrap_reward_with_curriculum_log -------------------------------------


class _RecordingLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


def _passthrough_reward(
    completions=None, prompts=None, ground_truth_mm=None, valid_range_mm=None, **kwargs,
) -> list[float]:
    return [0.0] * len(completions or [])


def test_curriculum_log_writes_one_row_per_rollout() -> None:
    log = _RecordingLog()
    wrapped = tg._wrap_reward_with_curriculum_log(
        _passthrough_reward,
        log=log,
        section_bins={"sec_a": 2, "sec_b": 4},
        round_idx=3,
    )
    wrapped(
        completions=[
            '<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>',
            '<|tool_call>call:submit_estimate{position_mm:6.0}<tool_call|>',
        ],
        ground_truth_mm=[5.0, 5.0],
        valid_range_mm=[(0.0, 10.0), (0.0, 10.0)],
        section_id=["sec_a", "sec_b"],
        plane=["coronal", "sagittal"],
    )
    assert len(log.rows) == 2
    # exact-hit row
    a = log.rows[0]
    assert a["round"] == 3
    assert a["section_id"] == "sec_a"
    assert a["bin_idx"] == 2
    assert a["plane"] == "coronal"
    assert a["abs_err_mm"] == pytest.approx(0.0)
    assert a["accuracy_pct"] == pytest.approx(100.0)
    # 1mm-off row
    b = log.rows[1]
    assert b["section_id"] == "sec_b"
    assert b["bin_idx"] == 4
    assert b["abs_err_mm"] == pytest.approx(1.0)
    assert b["accuracy_pct"] == pytest.approx(90.0)


def test_curriculum_log_parse_failure_yields_full_plane_error() -> None:
    log = _RecordingLog()
    wrapped = tg._wrap_reward_with_curriculum_log(
        _passthrough_reward, log=log,
        section_bins={"sec_a": 0}, round_idx=0,
    )
    wrapped(
        completions=["totally not json"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=["sec_a"],
        plane=["coronal"],
    )
    assert len(log.rows) == 1
    assert log.rows[0]["abs_err_mm"] == pytest.approx(10.0)  # full plane span
    assert log.rows[0]["accuracy_pct"] == pytest.approx(0.0)


def test_curriculum_log_out_of_range_yields_at_least_full_span_error() -> None:
    log = _RecordingLog()
    wrapped = tg._wrap_reward_with_curriculum_log(
        _passthrough_reward, log=log,
        section_bins={"sec_a": 0}, round_idx=0,
    )
    wrapped(
        completions=[
            '<|tool_call>call:submit_estimate{position_mm:100.0}<tool_call|>',
        ],  # well outside [0, 10]
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=["sec_a"],
        plane=["coronal"],
    )
    assert len(log.rows) == 1
    assert log.rows[0]["abs_err_mm"] >= 10.0


def test_curriculum_log_unknown_section_id_marks_bin_minus_one() -> None:
    log = _RecordingLog()
    wrapped = tg._wrap_reward_with_curriculum_log(
        _passthrough_reward, log=log,
        section_bins={"sec_known": 1}, round_idx=0,
    )
    wrapped(
        completions=['<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=["sec_unknown"],
        plane=["coronal"],
    )
    assert len(log.rows) == 1
    assert log.rows[0]["bin_idx"] == -1


def test_curriculum_log_skips_rows_with_empty_section_id() -> None:
    log = _RecordingLog()
    wrapped = tg._wrap_reward_with_curriculum_log(
        _passthrough_reward, log=log,
        section_bins={}, round_idx=0,
    )
    wrapped(
        completions=['<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=[""],  # empty
        plane=["coronal"],
    )
    assert log.rows == []


def test_curriculum_log_returns_underlying_reward_scores() -> None:
    log = _RecordingLog()

    def underlying(completions=None, **kw):  # noqa: ANN001
        # Distinct scores per row so we can verify pass-through.
        return [0.42 * (i + 1) for i in range(len(completions or []))]

    wrapped = tg._wrap_reward_with_curriculum_log(
        underlying, log=log, section_bins={}, round_idx=0,
    )
    out = wrapped(
        completions=[
            '<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>',
            '<|tool_call>call:submit_estimate{position_mm:6.0}<tool_call|>',
        ],
        ground_truth_mm=[5.0, 5.0],
        valid_range_mm=[(0.0, 10.0), (0.0, 10.0)],
        section_id=["a", "b"],
        plane=["coronal", "coronal"],
    )
    assert out == [pytest.approx(0.42), pytest.approx(0.84)]
