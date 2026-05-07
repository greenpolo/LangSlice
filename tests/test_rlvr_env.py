"""Unit tests for the RLVR scaffold (spec §11 verification 1 + 3).

Covers:
- ``rlvr.env.LangSliceEstimateEnv`` tool surface, dict-shaped responses,
  prod-shape signatures, defensive done-fence;
- ``rlvr.atlas_grid`` boundary clamp;
- ``rlvr.dataset.split_subjects_for_holdout`` strict subject-level holdout;
- ``rlvr.train_grpo`` GRPOConfig built with ``stop_tool_names``;
- ``rlvr.train_grpo`` Phase-B adapter resume (heavy deps mocked).

These tests stub out ``AtlasGrid`` so they don't require BrainGlobe atlases
or the slow pre-render step — env behavior (clamp, dedupe, cap, kind
matching, ground-truth privacy) is independent of which slices come back.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from rlvr import env as env_mod
from rlvr.atlas_grid import GRID_STEP_MM, GridRange, _grid_position_mm, _quantize
from rlvr.dataset import SingleSliceExample, split_subjects_for_holdout
from rlvr.env import (
    DEDUPE_TOL_MM,
    MAX_POSITIONS_PER_FETCH,
    LangSliceEstimateEnv,
    _public_tool_method_names,
)


class _StubAtlasGrid:
    """Minimal AtlasGrid stand-in: returns a tiny image and snaps to 0.05 mm."""

    def __init__(self, pos_lo: float = 0.0, pos_hi: float = 13.2) -> None:
        self._pos_lo = pos_lo
        self._pos_hi = pos_hi

    def range_mm(self, atlas_name: str, plane: str) -> Any:  # noqa: ARG002
        return GridRange(pos_lo=self._pos_lo, pos_hi=self._pos_hi)

    def get_slice(
        self, atlas_name: str, plane: str, position_mm: float  # noqa: ARG002
    ) -> tuple[Image.Image, float]:
        snapped = round(position_mm / 0.05) * 0.05
        return Image.new("L", (4, 4), color=int(snapped * 10) % 256), snapped


@pytest.fixture
def single_env() -> LangSliceEstimateEnv:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    e.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(5.5,),
        kind="single",
        prompt=[],
        image=None,
        subject_id="M01",
    )
    return e


@pytest.fixture
def group_env() -> LangSliceEstimateEnv:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    e.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(2.5, 3.0, 3.5, 4.0),
        kind="group",
        prompt=[],
        images=[],
        subject_id="M02",
    )
    return e


# --- tool surface ----------------------------------------------------------


def test_only_three_tools_exposed() -> None:
    """``reset`` and any leading-underscore method must NOT be auto-exposed."""
    assert _public_tool_method_names() == (
        "fetch_atlas",
        "submit_estimate",
        "submit_group_estimate",
    )


def test_tool_signatures_match_production_shape() -> None:
    """All three tools accept the prod-shape kwargs (positions/position_mm,
    reasoning, optional tool_context). Production tools live in
    ``src/langslice_harness/harness/estimation/tools.py``.
    """
    sig_fetch = inspect.signature(LangSliceEstimateEnv.fetch_atlas)
    assert list(sig_fetch.parameters) == ["self", "positions_mm", "tool_context"]

    sig_submit = inspect.signature(LangSliceEstimateEnv.submit_estimate)
    assert list(sig_submit.parameters) == ["self", "position_mm", "reasoning", "tool_context"]

    sig_group = inspect.signature(LangSliceEstimateEnv.submit_group_estimate)
    assert list(sig_group.parameters) == ["self", "positions_mm", "reasoning", "tool_context"]


def test_no_public_method_returns_ground_truth() -> None:
    """No public method may have a return annotation that leaks ground truth."""
    instance = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    instance.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(5.5,),
        kind="single",
    )
    for name in _public_tool_method_names():
        method = getattr(instance, name)
        sig = inspect.signature(method)
        # The signature must not mention ``ground_truth`` anywhere.
        assert "ground_truth" not in str(sig).lower(), name
    fetched = instance.fetch_atlas([5.5])
    flattened = " ".join(
        b.get("text", "") for b in fetched["content"] if b.get("type") == "text"
    )
    assert "ground_truth" not in flattened.lower()


# --- fetch_atlas validation ------------------------------------------------


def test_fetch_atlas_returns_dict_with_content_blocks(single_env: LangSliceEstimateEnv) -> None:
    """Production parity: tools return dicts; the multimodal blocks live under ``content``."""
    resp = single_env.fetch_atlas([2.0])
    assert isinstance(resp, dict)
    assert resp["status"] == "ok"
    assert resp["positions_mm"] == [2.0]
    image_blocks = [b for b in resp["content"] if b["type"] == "image"]
    assert len(image_blocks) == 1


def test_fetch_atlas_clamps_to_valid_range(single_env: LangSliceEstimateEnv) -> None:
    resp = single_env.fetch_atlas([-1.0, 20.0])
    text = " ".join(b["text"] for b in resp["content"] if b["type"] == "text")
    assert "clamped" in text
    image_blocks = [b for b in resp["content"] if b["type"] == "image"]
    assert len(image_blocks) == 2


def test_fetch_atlas_dedupes_close_positions(single_env: LangSliceEstimateEnv) -> None:
    resp = single_env.fetch_atlas([5.50, 5.50 + DEDUPE_TOL_MM / 2])
    image_blocks = [b for b in resp["content"] if b["type"] == "image"]
    assert len(image_blocks) == 1
    text = " ".join(b["text"] for b in resp["content"] if b["type"] == "text")
    assert "deduped" in text


def test_fetch_atlas_caps_at_max_positions(single_env: LangSliceEstimateEnv) -> None:
    requested = [0.5 + i * 0.5 for i in range(MAX_POSITIONS_PER_FETCH + 3)]
    resp = single_env.fetch_atlas(requested)
    image_blocks = [b for b in resp["content"] if b["type"] == "image"]
    assert len(image_blocks) == MAX_POSITIONS_PER_FETCH
    text = " ".join(b["text"] for b in resp["content"] if b["type"] == "text")
    assert "8-per-call cap" in text


def test_fetch_atlas_image_before_text_per_position(single_env: LangSliceEstimateEnv) -> None:
    """Gemma 4's chat template requires the image block before its caption."""
    resp = single_env.fetch_atlas([2.0, 4.0])
    blocks = resp["content"]
    for i, b in enumerate(blocks):
        if b["type"] == "image":
            assert blocks[i + 1]["type"] == "text"
            assert "Atlas at" in blocks[i + 1]["text"]


def test_fetch_atlas_empty_input_records_malformed(single_env: LangSliceEstimateEnv) -> None:
    resp = single_env.fetch_atlas([])
    assert single_env._state.malformed_tool_calls == 1
    assert resp["status"] == "error"
    assert resp["error"] == "BAD_ARGS"
    assert resp["content"][0]["type"] == "text"


def test_tools_accept_tool_context_kwarg(single_env: LangSliceEstimateEnv) -> None:
    """The kwarg exists for prod parity but is ignored."""
    sentinel = object()
    resp = single_env.fetch_atlas([2.0], tool_context=sentinel)
    assert resp["status"] == "ok"


# --- submit kind matching --------------------------------------------------


def test_submit_estimate_rejects_when_kind_is_group(group_env: LangSliceEstimateEnv) -> None:
    resp = group_env.submit_estimate(3.0, "guess")
    assert resp["status"] == "error" and resp["error"] == "WRONG_KIND"
    assert "single" in resp["message"]
    assert group_env._state.submitted_positions_mm is None
    assert group_env._state.malformed_tool_calls == 1


def test_submit_group_estimate_rejects_when_kind_is_single(
    single_env: LangSliceEstimateEnv,
) -> None:
    resp = single_env.submit_group_estimate([5.5, 6.0], "guess")
    assert resp["status"] == "error" and resp["error"] == "WRONG_KIND"
    assert "group" in resp["message"]
    assert single_env._state.submitted_positions_mm is None
    assert single_env._state.malformed_tool_calls == 1


def test_submit_estimate_records_and_marks_done(single_env: LangSliceEstimateEnv) -> None:
    resp = single_env.submit_estimate(5.55, "anterior commissure visible")
    assert resp["status"] == "ok"
    assert resp["position_mm"] == pytest.approx(5.55)
    assert single_env._state.submitted_positions_mm == (5.55,)
    assert single_env._state.submitted_kind == "single"
    assert single_env._state.done is True


def test_submit_group_estimate_records_all_positions(group_env: LangSliceEstimateEnv) -> None:
    resp = group_env.submit_group_estimate([2.5, 3.05, 3.55, 4.0], "matched landmarks")
    assert resp["status"] == "ok"
    assert resp["positions_mm"] == [2.5, 3.05, 3.55, 4.0]
    assert group_env._state.submitted_positions_mm == (2.5, 3.05, 3.55, 4.0)
    assert group_env._state.submitted_kind == "group"
    assert group_env._state.done is True


# --- defensive done-fence (P1(b)) ------------------------------------------


def test_post_done_fetch_atlas_rejected(single_env: LangSliceEstimateEnv) -> None:
    """Once submitted, fetch_atlas must refuse and count as malformed."""
    single_env.submit_estimate(5.5, "ok")
    assert single_env._state.done is True
    assert single_env._state.malformed_tool_calls == 0
    resp = single_env.fetch_atlas([3.0])
    assert resp["status"] == "error" and resp["error"] == "ALREADY_DONE"
    assert single_env._state.malformed_tool_calls == 1


def test_post_done_second_submit_rejected(single_env: LangSliceEstimateEnv) -> None:
    single_env.submit_estimate(5.5, "first")
    resp = single_env.submit_estimate(5.6, "second")
    assert resp["status"] == "error" and resp["error"] == "ALREADY_DONE"
    # First submission is still the recorded one.
    assert single_env._state.submitted_positions_mm == (5.5,)


def test_post_done_group_submit_rejected(group_env: LangSliceEstimateEnv) -> None:
    group_env.submit_group_estimate([2.5, 3.0, 3.5, 4.0], "first")
    resp = group_env.submit_group_estimate([1.0, 2.0, 3.0, 4.0], "second")
    assert resp["status"] == "error" and resp["error"] == "ALREADY_DONE"


# --- ground-truth privacy --------------------------------------------------


def test_ground_truth_not_in_any_tool_response(single_env: LangSliceEstimateEnv) -> None:
    fetch = single_env.fetch_atlas([5.4, 5.5, 5.6])
    submit = single_env.submit_estimate(5.5, "guess")
    fetch_text = " ".join(b["text"] for b in fetch["content"] if b["type"] == "text")
    assert "ground_truth" not in fetch_text.lower()
    assert "ground_truth" not in str(submit).lower()


def test_reset_validates_kind_and_positions() -> None:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5,),
            kind="invalid",
        )
    with pytest.raises(ValueError, match="single-slice kind expects exactly 1"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5, 6.0),
            kind="single",
        )
    with pytest.raises(ValueError, match="group kind expects >=2"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5,),
            kind="group",
        )
    with pytest.raises(ValueError, match="valid_range_mm must be increasing"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(5.0, 5.0),
            ground_truth_positions_mm=(5.5,),
            kind="single",
        )


def test_reset_rejects_atlas_grid_without_pair() -> None:
    class _EmptyGrid:
        def range_mm(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyError("no pair")

    e = LangSliceEstimateEnv(atlas_grid=_EmptyGrid())  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        e.reset(
            atlas_name="bogus",
            plane="coronal",
            valid_range_mm=(0.0, 1.0),
            ground_truth_positions_mm=(0.5,),
            kind="single",
        )


def test_module_exports_what_callers_import() -> None:
    assert env_mod.LangSliceEstimateEnv is LangSliceEstimateEnv
    assert env_mod.MAX_POSITIONS_PER_FETCH == 8
    assert env_mod.DEDUPE_TOL_MM == 0.02


# --- atlas_grid boundary (P1(d)) -------------------------------------------


def test_atlas_grid_clamps_hi_index_inward() -> None:
    """``build_atlas_grid`` must never ask the atlas to render past pos_hi.

    Previously it used ``int(round(pos_hi / 0.05))`` which can round up past
    the last valid index. The fix clamps the rounded index inward when its
    mm position exceeds ``pos_hi``.
    """
    # Synthesize a pos_hi whose nearest 0.05 mm grid point lies *above* it:
    # pos_hi = 13.225 → round(13.225/0.05) = 264 → mm = 13.20 (in-range; OK).
    # pos_hi = 13.275 → round(13.275/0.05) = 266 → mm = 13.30 (over!) → must clamp to 265.
    pos_hi = 13.275

    rendered_positions: list[float] = []

    def fake_load_atlas(name):  # noqa: ANN001, ARG001
        return object()

    def fake_get_position_range_mm(atlas, *, plane):  # noqa: ANN001, ARG001
        return 0.0, pos_hi

    def fake_get_reference_slice(atlas, position_mm, *, plane):  # noqa: ANN001, ARG001
        # Mirror production: reject any position past pos_hi like
        # position_mm_to_index does.
        if position_mm > pos_hi + 1e-9:
            raise ValueError(f"out of range: {position_mm} > {pos_hi}")
        rendered_positions.append(position_mm)
        return Image.new("L", (4, 4))

    with patch("rlvr.atlas_grid.load_atlas", fake_load_atlas), patch(
        "rlvr.atlas_grid.get_position_range_mm", fake_get_position_range_mm
    ), patch("rlvr.atlas_grid.get_reference_slice", fake_get_reference_slice):
        from rlvr.atlas_grid import build_atlas_grid  # noqa: PLC0415

        grid = build_atlas_grid([("allen_mouse_25um", "coronal")])

    # No rendered position may exceed pos_hi — the boundary clamp prevents it.
    assert all(p <= pos_hi + 1e-9 for p in rendered_positions)
    # And the highest rendered position is the LARGEST grid mm <= pos_hi
    # (i.e. the last index whose mm fits inside the atlas).
    expected_hi = _grid_position_mm(_quantize(pos_hi) - 1)  # 265 * 0.05 = 13.25
    assert max(rendered_positions) == pytest.approx(expected_hi)
    # Sanity: the cached grid round-trips lookup at pos_hi.
    img, snapped = grid.get_slice("allen_mouse_25um", "coronal", pos_hi)
    assert isinstance(img, Image.Image)
    assert snapped <= pos_hi + 1e-9


def test_atlas_grid_step_constant() -> None:
    """Make sure the documented grid step is what tests assume."""
    assert GRID_STEP_MM == pytest.approx(0.05)


# --- subject-level holdout (P2(h)) -----------------------------------------


_FAKE_ROW = {"atlas_name": "a", "plane": "coronal", "kind": "single"}
_FAKE_PAIRS = {("a", "coronal")}


def _example(subject_id: str) -> SingleSliceExample:
    return SingleSliceExample(
        image_path=Path(f"/fake/{subject_id}.tif"),
        atlas_name="allen_mouse_25um",
        plane="coronal",
        ap_mm=5.0,
        subject_id=subject_id,
    )


def test_split_subjects_for_holdout_disjoint_and_deterministic() -> None:
    examples = [_example(f"M{i:02d}") for i in range(1, 11)]
    train, holdout = split_subjects_for_holdout(examples, eval_holdout_every=5)
    # Sorted ids → ['M01'..'M10']; every 5th (1-indexed) → indices 4 and 9.
    assert holdout == {"M05", "M10"}
    assert train == {f"M{i:02d}" for i in range(1, 11)} - {"M05", "M10"}
    assert train.isdisjoint(holdout)
    # Determinism: rerun returns the same partition.
    train2, holdout2 = split_subjects_for_holdout(examples, eval_holdout_every=5)
    assert train2 == train and holdout2 == holdout


def test_split_subjects_for_holdout_zero_or_negative_disables() -> None:
    examples = [_example(f"M{i:02d}") for i in range(1, 6)]
    train, holdout = split_subjects_for_holdout(examples, eval_holdout_every=0)
    assert holdout == set()
    assert train == {f"M{i:02d}" for i in range(1, 6)}
    train_neg, holdout_neg = split_subjects_for_holdout(examples, eval_holdout_every=-1)
    assert holdout_neg == set()
    assert train_neg == train


def test_split_subjects_for_holdout_dedupes_subjects() -> None:
    """Multiple slices per subject must collapse to one entry in the split."""
    examples = [_example("M01"), _example("M01"), _example("M02"), _example("M03")]
    train, holdout = split_subjects_for_holdout(examples, eval_holdout_every=3)
    assert train | holdout == {"M01", "M02", "M03"}
    assert train.isdisjoint(holdout)


def test_build_datasets_uses_disjoint_subject_holdout() -> None:
    """Driver dataset assembly must keep held-out subjects out of train rows."""
    from rlvr.train_grpo import build_datasets  # noqa: PLC0415

    examples = [_example(f"M{i:02d}") for i in range(1, 11)]

    def fake_rows(
        *, single_examples, group_examples, single_fraction, seed
    ):  # noqa: ANN001, ARG001
        return [
            {
                "atlas_name": ex.atlas_name,
                "plane": ex.plane,
                "kind": "single",
                "subject_id": ex.subject_id,
            }
            for ex in single_examples
        ]

    with patch("rlvr.train_grpo.load_test_images", return_value=examples), patch(
        "rlvr.train_grpo.make_group_examples", return_value=[]
    ), patch("rlvr.train_grpo.build_rlvr_rows", side_effect=fake_rows), patch(
        "rlvr.train_grpo.to_hf_dataset", side_effect=lambda rows: rows
    ):
        train_ds, eval_ds, train_rows, eval_rows, _atlas_pairs = build_datasets(
            test_images_root=Path("unused"),
            data_cfg={"eval_holdout_every": 5, "single_fraction": 1.0},
            seed=0,
        )

    train_subjects = {row["subject_id"] for row in train_rows}
    eval_subjects = {row["subject_id"] for row in eval_rows}
    assert train_subjects == {f"M{i:02d}" for i in range(1, 11)} - {"M05", "M10"}
    assert eval_subjects == {"M05", "M10"}
    assert train_subjects.isdisjoint(eval_subjects)
    assert len(train_ds) == 8
    assert len(eval_ds) == 2


# --- GRPOConfig stop_tool_names wiring (P1(b) + P4) ------------------------


def test_grpo_default_config_has_stop_tool_names() -> None:
    """Both TOML configs must set stop_tool_names to terminate on submit."""
    import tomllib  # noqa: PLC0415

    for name in ("grpo_default.toml", "grpo_pilot.toml", "grpo_phase_b.toml"):
        path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "langslice-gemma-4"
            / "training"
            / "configs"
            / name
        )
        with path.open("rb") as f:
            cfg = tomllib.load(f)
        assert "stop_tool_names" in cfg["grpo"], name
        assert set(cfg["grpo"]["stop_tool_names"]) == {
            "submit_estimate",
            "submit_group_estimate",
        }, name
        # Confirm max_prompt_length is gone from every config (P1(a)).
        assert "max_prompt_length" not in cfg["grpo"], name


def test_repo_root_launch_module_shows_help() -> None:
    """Canonical repo-root launch must not rely on pytest pythonpath."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "langslice_rlvr", "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--resume-from-adapter" in result.stdout


# --- adapter resume smoke test (P2(g) + P4) --------------------------------


def test_resume_from_adapter_attaches_trainable_peft_adapter() -> None:
    """Phase B: --resume-from-adapter attaches the adapter to the SFT base."""
    import sys  # noqa: PLC0415

    captured: dict[str, Any] = {}

    fake_unsloth = MagicMock()

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return MagicMock(name="model"), MagicMock(name="processor")

    fake_unsloth.FastVisionModel.from_pretrained.side_effect = fake_from_pretrained
    # When resuming, get_peft_model must NOT be called; PEFT loads the adapter.
    fake_unsloth.FastVisionModel.get_peft_model.side_effect = AssertionError(
        "get_peft_model must not be called when resuming from an adapter"
    )

    fake_peft = MagicMock()
    fake_peft_model = MagicMock(name="phase_b_peft_model")
    fake_peft.PeftModel.from_pretrained.return_value = fake_peft_model

    fake_trl = MagicMock()
    fake_trainer = MagicMock()
    fake_trl.GRPOTrainer.return_value = fake_trainer
    fake_trl.GRPOConfig.side_effect = lambda **kw: kw  # passthrough

    sft_path = Path("fake/sft")
    adapter_path = Path("fake/phase_a")
    output_path = Path("fake/phase_b")

    # Stub all the heavy training prep so we never hit BrainGlobe / disk.
    fake_grid = MagicMock()
    fake_train_ds = MagicMock(name="train_dataset")
    fake_eval_ds = MagicMock(name="eval_dataset")

    fake_datasets = (fake_train_ds, fake_eval_ds, [_FAKE_ROW], [], _FAKE_PAIRS)
    with patch.dict(
        sys.modules, {"trl": fake_trl, "unsloth": fake_unsloth, "peft": fake_peft}
    ), patch(
        "rlvr.train_grpo.build_datasets", return_value=fake_datasets
    ), patch("rlvr.train_grpo.build_atlas_grid", return_value=fake_grid):
        from rlvr.train_grpo import main  # noqa: PLC0415

        main(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "models"
                    / "langslice-gemma-4"
                    / "training"
                    / "configs"
                    / "grpo_phase_b.toml"
                ),
                "--sft-model",
                str(sft_path),
                "--resume-from-adapter",
                str(adapter_path),
                "--output-dir",
                str(output_path),
                "--test-images-root",
                "fake/TestImages",
            ]
        )

    # The SFT base is loaded first, then the Phase A adapter is attached as trainable.
    assert captured["model_name"] == str(sft_path)
    fake_peft.PeftModel.from_pretrained.assert_called_once()
    _, adapter_arg = fake_peft.PeftModel.from_pretrained.call_args.args
    assert adapter_arg == str(adapter_path)
    assert fake_peft.PeftModel.from_pretrained.call_args.kwargs["is_trainable"] is True
    # max_seq_length was wired through (P1(e)).
    assert captured["kwargs"].get("max_seq_length") == 4096
    # Trainer was constructed and .train() called.
    fake_trl.GRPOTrainer.assert_called_once()
    assert fake_trl.GRPOTrainer.call_args.kwargs["model"] is fake_peft_model
    fake_trainer.train.assert_called_once()


def test_no_resume_calls_get_peft_model() -> None:
    """Without --resume-from-adapter, the driver must wrap the model in PEFT."""
    import sys  # noqa: PLC0415

    fake_unsloth = MagicMock()
    fake_unsloth.FastVisionModel.from_pretrained.return_value = (
        MagicMock(name="base_model"),
        MagicMock(name="processor"),
    )
    fake_unsloth.FastVisionModel.get_peft_model.return_value = MagicMock(name="peft_model")

    fake_trl = MagicMock()
    fake_trainer = MagicMock()
    fake_trl.GRPOTrainer.return_value = fake_trainer
    fake_trl.GRPOConfig.side_effect = lambda **kw: kw

    sft_path = Path("fake/sft")

    fake_grid = MagicMock()
    fake_train_ds = MagicMock()

    fake_datasets = (fake_train_ds, None, [_FAKE_ROW], [], _FAKE_PAIRS)
    with patch.dict(sys.modules, {"trl": fake_trl, "unsloth": fake_unsloth}), patch(
        "rlvr.train_grpo.build_datasets", return_value=fake_datasets
    ), patch("rlvr.train_grpo.build_atlas_grid", return_value=fake_grid):
        from rlvr.train_grpo import main  # noqa: PLC0415

        main(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "models"
                    / "langslice-gemma-4"
                    / "training"
                    / "configs"
                    / "grpo_pilot.toml"
                ),
                "--sft-model",
                str(sft_path),
                "--output-dir",
                "fake/out",
                "--test-images-root",
                "fake/TestImages",
            ]
        )

    fake_unsloth.FastVisionModel.get_peft_model.assert_called_once()


def test_sft_adapter_model_loads_base_then_attaches_trainable_adapter(
    tmp_path: Path,
) -> None:
    """Phase A: --sft-model may be a PEFT adapter produced by SFT."""
    import sys  # noqa: PLC0415

    captured: dict[str, Any] = {}

    fake_unsloth = MagicMock()
    base_model = MagicMock(name="base_model")
    processor = MagicMock(name="processor")

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return base_model, processor

    fake_unsloth.FastVisionModel.from_pretrained.side_effect = fake_from_pretrained
    fake_unsloth.FastVisionModel.get_peft_model.side_effect = AssertionError(
        "SFT adapter path must be attached directly, not replaced by a fresh adapter"
    )

    fake_peft = MagicMock()
    sft_model = MagicMock(name="sft_adapter_model")
    fake_peft.PeftModel.from_pretrained.return_value = sft_model

    fake_trl = MagicMock()
    fake_trainer = MagicMock()
    fake_trl.GRPOTrainer.return_value = fake_trainer
    fake_trl.GRPOConfig.side_effect = lambda **kw: kw

    sft_path = tmp_path / "sft_adapter"
    sft_path.mkdir()
    base_id = "unsloth/gemma-4-E4B-it"
    (sft_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_id}),
        encoding="utf-8",
    )

    fake_grid = MagicMock()
    fake_train_ds = MagicMock(name="train_dataset")
    fake_datasets = (fake_train_ds, None, [_FAKE_ROW], [], _FAKE_PAIRS)

    with patch.dict(
        sys.modules, {"trl": fake_trl, "unsloth": fake_unsloth, "peft": fake_peft}
    ), patch(
        "rlvr.train_grpo.build_datasets", return_value=fake_datasets
    ), patch("rlvr.train_grpo.build_atlas_grid", return_value=fake_grid):
        from rlvr.train_grpo import main  # noqa: PLC0415

        main(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "models"
                    / "langslice-gemma-4"
                    / "training"
                    / "configs"
                    / "grpo_pilot.toml"
                ),
                "--sft-model",
                str(sft_path),
                "--output-dir",
                "fake/out",
                "--test-images-root",
                "fake/TestImages",
            ]
        )

    assert captured["model_name"] == base_id
    fake_peft.PeftModel.from_pretrained.assert_called_once_with(
        base_model,
        str(sft_path),
        is_trainable=True,
    )
    fake_unsloth.FastVisionModel.get_peft_model.assert_not_called()
    fake_trl.GRPOTrainer.assert_called_once()
    assert fake_trl.GRPOTrainer.call_args.kwargs["model"] is sft_model


def test_train_grpo_passes_stop_tool_names_to_grpo_config() -> None:
    """The GRPOConfig built by main() must carry stop_tool_names from the TOML."""
    import sys  # noqa: PLC0415

    captured_config_kwargs: dict[str, Any] = {}

    fake_unsloth = MagicMock()
    fake_unsloth.FastVisionModel.from_pretrained.return_value = (
        MagicMock(),
        MagicMock(),
    )
    fake_unsloth.FastVisionModel.get_peft_model.return_value = MagicMock()

    fake_trl = MagicMock()

    def capture_config(**kwargs):
        captured_config_kwargs.update(kwargs)
        return kwargs

    fake_trl.GRPOConfig.side_effect = capture_config
    fake_trl.GRPOTrainer.return_value = MagicMock()

    sft_path = Path("fake/sft")
    fake_grid = MagicMock()

    fake_datasets = (MagicMock(), None, [_FAKE_ROW], [], _FAKE_PAIRS)
    with patch.dict(sys.modules, {"trl": fake_trl, "unsloth": fake_unsloth}), patch(
        "rlvr.train_grpo.build_datasets", return_value=fake_datasets
    ), patch("rlvr.train_grpo.build_atlas_grid", return_value=fake_grid):
        from rlvr.train_grpo import main  # noqa: PLC0415

        main(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "models"
                    / "langslice-gemma-4"
                    / "training"
                    / "configs"
                    / "grpo_default.toml"
                ),
                "--sft-model",
                str(sft_path),
                "--output-dir",
                "fake/out",
                "--test-images-root",
                "fake/TestImages",
            ]
        )

    assert "stop_tool_names" in captured_config_kwargs
    assert set(captured_config_kwargs["stop_tool_names"]) == {
        "submit_estimate",
        "submit_group_estimate",
    }
    # And max_prompt_length is NOT passed (P1(a)).
    assert "max_prompt_length" not in captured_config_kwargs
