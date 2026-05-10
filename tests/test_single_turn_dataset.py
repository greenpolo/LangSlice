"""Unit tests for ``single_turn_rl.dataset``.

Covers:
* ``__getitem__`` produces Gemma 4 image-before-text user content
  (every ``{type: "image"}`` block is followed by a ``{type: "text"}`` caption).
* PIL images are **embedded** in ``part["image"]`` (TRL VLM GRPO extracts them
  from chat content via ``part["image"]`` — a parallel ``images`` column is
  not used on the no-environment-factory path).
* Public columns include ``image_paths`` for the atlas-cache splice lookup
  + the reward kwargs ``ground_truth_mm`` / ``valid_range_mm``.
* :class:`WeightedRowDataset` exposes ``_weights`` and validates set_weights().
* ``split_subjects_for_holdout`` is deterministic and disjoint.

We monkeypatch BrainGlobe-loading helpers so these tests don't need a real
atlas on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from single_turn_rl import dataset as ds
from single_turn_rl.terminal_states import Plane, TerminalState


def _stub_atlas_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_atlas_in_plane_long_edge", lambda _name, _plane: 320)
    monkeypatch.setattr(ds, "species_from_atlas_name", lambda _name: "mouse")
    monkeypatch.setattr(ds, "canonicalize_atlas_name", lambda name: name)
    monkeypatch.setattr(
        ds, "preprocess_query_image",
        lambda *_a, **_k: "QUERY_IMAGE",  # type: ignore[arg-type, return-value]
    )
    monkeypatch.setattr(
        ds, "load_atlas_reference_image",
        lambda _p: "ATLAS_IMAGE",  # type: ignore[arg-type, return-value]
    )


def _make_state(
    *,
    section_id: str = "sec_a",
    subject_id: str = "subj_1",
    atlas_image_paths: tuple[str, ...] = (
        "atlas/allen_mouse_25um/coronal/4.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/4.50mm.jpg",
    ),
    fetched_positions_mm: tuple[float, ...] = (4.0, 4.5),
    plane: Plane = "coronal",
    valid_range_mm: tuple[float, float] = (0.0, 13.2),
    ground_truth_mm: float = 4.37,
) -> TerminalState:
    return TerminalState(
        section_id=section_id,
        subject_id=subject_id,
        atlas_name="allen_mouse_25um",
        plane=plane,
        valid_range_mm=valid_range_mm,
        ground_truth_mm=ground_truth_mm,
        query_image_path=f"queries/{section_id}.jpg",
        atlas_image_paths=atlas_image_paths,
        fetched_positions_mm=fetched_positions_mm,
        source="sft_corpus:strict",
        quality={"acceptance_tier": "strict"},
    )


def _make_row(state: TerminalState, monkeypatch: pytest.MonkeyPatch) -> dict:
    _stub_atlas_helpers(monkeypatch)
    spec = ds._spec_from_state(state, repo_root=Path("/repo"))
    return ds._build_row_from_spec(spec)


# --- Chat content shape ----------------------------------------------------


def test_user_content_alternates_image_then_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _make_row(_make_state(), monkeypatch)

    user_msg = row["prompt"][1]
    assert user_msg["role"] == "user"
    blocks = user_msg["content"]

    image_indices = [i for i, b in enumerate(blocks) if b.get("type") == "image"]
    for idx in image_indices:
        assert idx + 1 < len(blocks), "image block at end of content"
        assert blocks[idx + 1].get("type") == "text", (
            f"image block at index {idx} not followed by text"
        )
    assert blocks[-1].get("type") == "text"
    assert "Output the JSON object" in blocks[-1]["text"]


def test_pil_images_embedded_in_chat_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRL's _tokenize_prompts reads PIL via ``part["image"]`` — verify it's there."""
    state = _make_state()
    row = _make_row(state, monkeypatch)
    blocks = row["prompt"][1]["content"]

    image_blocks = [b for b in blocks if b.get("type") == "image"]
    # 1 target + N atlas images
    assert len(image_blocks) == 1 + len(state.atlas_image_paths)
    # First image block is the target; remaining are atlas refs.
    assert image_blocks[0]["image"] == "QUERY_IMAGE"
    assert all(b["image"] == "ATLAS_IMAGE" for b in image_blocks[1:])


def test_target_caption_first(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _make_row(_make_state(), monkeypatch)
    blocks = row["prompt"][1]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[1]["type"] == "text"
    assert "TARGET" in blocks[1]["text"]


def test_atlas_captions_carry_position_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(
        atlas_image_paths=("a/2.00mm.jpg", "a/3.50mm.jpg"),
        fetched_positions_mm=(2.0, 3.5),
    )
    row = _make_row(state, monkeypatch)
    blocks = row["prompt"][1]["content"]
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    captions = " ".join(b["text"] for b in text_blocks)
    assert "2.00 mm" in captions
    assert "3.50 mm" in captions


def test_system_prompt_includes_atlas_and_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(valid_range_mm=(0.5, 12.7))
    row = _make_row(state, monkeypatch)
    sys_text = row["prompt"][0]["content"][0]["text"]
    assert "allen_mouse_25um" in sys_text
    assert "mouse" in sys_text
    assert "0.50-12.70 mm" in sys_text
    assert '"position_mm"' in sys_text


# --- Public columns --------------------------------------------------------


def test_row_exposes_reward_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _make_row(_make_state(ground_truth_mm=4.37, valid_range_mm=(0.0, 13.2)), monkeypatch)
    assert row["ground_truth_mm"] == pytest.approx(4.37)
    assert row["valid_range_mm"] == (0.0, 13.2)
    assert row["plane"] == "coronal"
    assert row["atlas_name"] == "allen_mouse_25um"
    assert row["subject_id"] == "subj_1"
    assert row["section_id"] == "sec_a"


def test_row_exposes_image_paths_for_splice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The atlas-cache splice consumes ``image_paths`` to look up cached embeddings."""
    state = _make_state(
        atlas_image_paths=("atlas/x/coronal/2.00mm.jpg", "atlas/x/coronal/3.00mm.jpg"),
    )
    row = _make_row(state, monkeypatch)
    assert "image_paths" in row
    # Order: query first, then atlas refs in trace order.
    assert row["image_paths"] == [
        state.query_image_path,
        *state.atlas_image_paths,
    ]


# --- RowDataset ------------------------------------------------------------


def test_rowdataset_column_names_match_public_set() -> None:
    rd = ds.RowDataset([])
    expected = {
        "prompt", "image_paths", "ground_truth_mm", "valid_range_mm",
        "plane", "atlas_name", "subject_id", "section_id",
        # Task 6 (unified RL pipeline): Lane B specs carry ``dataset`` and
        # ``difficulty_score``. Lane A specs default these to ""/None so the
        # column shape stays uniform across both lanes.
        "dataset", "difficulty_score",
    }
    assert set(rd.column_names) == expected


def test_rowdataset_len_and_index_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    state = _make_state()
    spec = ds._spec_from_state(state, repo_root=Path("/repo"))
    rd = ds.RowDataset([spec])
    assert len(rd) == 1
    item = rd[0]
    assert item["plane"] == "coronal"
    assert item["ground_truth_mm"] == pytest.approx(4.37)
    assert item["image_paths"][0] == state.query_image_path


# --- WeightedRowDataset ----------------------------------------------------


def test_weighted_dataset_default_weights_are_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    specs = [ds._spec_from_state(_make_state(section_id=f"s{i}", subject_id=f"sub{i}"),
                                  repo_root=Path("/r"))
             for i in range(3)]
    wd = ds.WeightedRowDataset(specs)
    assert wd._weights == [1.0, 1.0, 1.0]


def test_weighted_dataset_set_weights_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    specs = [ds._spec_from_state(_make_state(section_id=f"s{i}", subject_id=f"sub{i}"),
                                  repo_root=Path("/r"))
             for i in range(3)]
    wd = ds.WeightedRowDataset(specs)
    wd.set_weights([0.5, 2.0, 1.0])
    assert wd._weights == [0.5, 2.0, 1.0]


def test_weighted_dataset_rejects_wrong_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    wd = ds.WeightedRowDataset([
        ds._spec_from_state(_make_state(), repo_root=Path("/r"))
    ])
    with pytest.raises(ValueError, match="expected 1 values"):
        wd.set_weights([1.0, 2.0])


def test_weighted_dataset_rejects_negative_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    wd = ds.WeightedRowDataset([
        ds._spec_from_state(_make_state(), repo_root=Path("/r"))
    ])
    with pytest.raises(ValueError, match="non-negative"):
        wd.set_weights([-0.1])


def test_weighted_dataset_rejects_all_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    wd = ds.WeightedRowDataset([
        ds._spec_from_state(_make_state(), repo_root=Path("/r")),
        ds._spec_from_state(_make_state(section_id="s2", subject_id="sub2"),
                             repo_root=Path("/r")),
    ])
    with pytest.raises(ValueError, match="at least one weight"):
        wd.set_weights([0.0, 0.0])


# --- specs_to_single_slice_examples (curriculum bin source) ---------------


def test_specs_to_single_slice_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_helpers(monkeypatch)
    specs = [
        ds._spec_from_state(
            _make_state(section_id="alpha", subject_id="A", ground_truth_mm=2.5),
            repo_root=Path("/r"),
        ),
        ds._spec_from_state(
            _make_state(section_id="beta", subject_id="B", ground_truth_mm=8.1),
            repo_root=Path("/r"),
        ),
    ]
    examples = ds.specs_to_single_slice_examples(specs)
    assert len(examples) == 2
    assert examples[0].section_id == "alpha"
    assert examples[0].subject_id == "A"
    assert examples[0].ap_mm == pytest.approx(2.5)
    assert examples[1].section_id == "beta"
    assert examples[1].ap_mm == pytest.approx(8.1)


# --- split_subjects_for_holdout -------------------------------------------


def test_split_subjects_every_nth_assigned_to_eval() -> None:
    specs = [{"subject_id": f"s{i}"} for i in range(10)]
    train, evald = ds.split_subjects_for_holdout(specs, eval_holdout_every=5)
    sorted_ids = sorted({s["subject_id"] for s in specs})
    expected_eval = {sid for i, sid in enumerate(sorted_ids) if (i + 1) % 5 == 0}
    assert evald == expected_eval
    assert train == set(sorted_ids) - expected_eval


def test_split_subjects_disjoint_and_complete() -> None:
    specs = [{"subject_id": f"s{i}"} for i in range(20)]
    train, evald = ds.split_subjects_for_holdout(specs, eval_holdout_every=4)
    assert train.isdisjoint(evald)
    assert train | evald == {f"s{i}" for i in range(20)}


def test_split_subjects_holdout_disabled_when_zero() -> None:
    specs = [{"subject_id": f"s{i}"} for i in range(5)]
    train, evald = ds.split_subjects_for_holdout(specs, eval_holdout_every=0)
    assert evald == set()
    assert train == {f"s{i}" for i in range(5)}


def test_split_subjects_deduplicates_across_specs() -> None:
    specs = [{"subject_id": "shared"}, {"subject_id": "shared"}, {"subject_id": "alone"}]
    train, evald = ds.split_subjects_for_holdout(specs, eval_holdout_every=0)
    assert train == {"shared", "alone"}
    assert evald == set()
