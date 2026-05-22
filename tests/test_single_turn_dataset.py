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
from langslice_training.rl.single_turn import dataset as ds
from langslice_training.rl.single_turn.terminal_states import Plane, TerminalState


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


def test_prompt_has_four_message_multi_turn_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lane A rows render the ADK multi-turn protocol: system + user(target) +
    assistant(fetch_atlas tool_call) + tool(atlas images). The model generates
    the next turn (submit_estimate), which matches slicebench's eval path."""
    row = _make_row(_make_state(), monkeypatch)
    prompt = row["prompt"]
    assert len(prompt) == 4
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert prompt[2]["role"] == "assistant"
    assert prompt[3]["role"] == "tool"


def test_user_message_carries_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user turn has the target image + the slicebench-parity user instruction.
    Atlas images now live in the tool-response message, not the user message."""
    row = _make_row(_make_state(), monkeypatch)
    user_msg = row["prompt"][1]
    blocks = user_msg["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["image"] == "QUERY_IMAGE"
    assert blocks[1]["type"] == "text"
    # User instruction matches slicebench's runner.py prompt verbatim.
    assert "Determine this coronal slice" in blocks[1]["text"]
    assert "AP position" in blocks[1]["text"]
    assert "allen_mouse_25um" in blocks[1]["text"]


def test_assistant_message_has_fetch_atlas_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-rendered assistant turn invokes fetch_atlas with the fetched
    positions as a dict arg (NOT a JSON string) — dict args route through the
    chat template's ``format_argument`` macro and produce the Gemma-native
    compact format that the reward parser expects."""
    state = _make_state(
        atlas_image_paths=("a/2.00mm.jpg", "a/4.50mm.jpg"),
        fetched_positions_mm=(2.0, 4.5),
    )
    row = _make_row(state, monkeypatch)
    assistant_msg = row["prompt"][2]
    assert assistant_msg.get("tool_calls"), "assistant turn missing tool_calls"
    tc = assistant_msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "fetch_atlas"
    args = tc["function"]["arguments"]
    assert isinstance(args, dict), (
        "fetch_atlas arguments must be a dict, not a JSON string — see "
        "chat_template.jinja format_argument macro"
    )
    assert args["positions_mm"] == [2.0, 4.5]


def test_tool_response_carries_atlas_images_and_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atlas images are returned by the synthetic fetch_atlas tool response,
    each followed by a position caption. The final text block is the
    production fetch_atlas summary line."""
    state = _make_state(
        atlas_image_paths=("a/2.00mm.jpg", "a/3.50mm.jpg"),
        fetched_positions_mm=(2.0, 3.5),
    )
    row = _make_row(state, monkeypatch)
    tool_msg = row["prompt"][3]
    assert tool_msg["tool_call_id"]  # must match assistant tool_call id
    blocks = tool_msg["content"]
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) == 2
    assert all(b["image"] == "ATLAS_IMAGE" for b in image_blocks)
    text_joined = " ".join(b["text"] for b in blocks if b.get("type") == "text")
    # The chat template's role=tool path doesn't auto-render images; the
    # tool message must inject <|image|> markers into its text content.
    assert "<|image|>" in text_joined
    assert "2.00 mm" in text_joined
    assert "3.50 mm" in text_joined
    assert "Fetched 2 atlas sections" in text_joined


def test_system_prompt_is_production_single_slice_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The system prompt now delegates to
    ``langslice_harness.harness.estimation.prompts.build_single_slice_prompt``
    — same instruction slicebench feeds the model at eval time."""
    state = _make_state(valid_range_mm=(0.5, 12.7))
    row = _make_row(state, monkeypatch)
    sys_text = row["prompt"][0]["content"][0]["text"]
    assert "allen_mouse_25um" in sys_text
    assert "mouse" in sys_text
    assert "0.50-12.70 mm" in sys_text
    # Production prompt's strategy boilerplate
    assert "fetch_atlas" in sys_text
    assert "submit_estimate" in sys_text


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
            "dataset", "difficulty_score", "ap_bin",
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


# --- build_datasets_from_index Lane A randomized path ----------------------


REPO_ROOT_DS_TEST = Path(__file__).resolve().parent.parent
ATLAS_CACHE_DIR_DS_TEST = REPO_ROOT_DS_TEST / "out" / "atlas_embeddings"
ALLEN_CORONAL_CACHE_DS_TEST = ATLAS_CACHE_DIR_DS_TEST / "allen_mouse_25um_coronal.pt"


_requires_atlas_cache_ds = pytest.mark.skipif(
    not ALLEN_CORONAL_CACHE_DS_TEST.is_file(),
    reason=(
        "build_datasets_from_index Lane-A tests need "
        "out/atlas_embeddings/allen_mouse_25um_coronal.pt — build it with "
        "the atlas embedding cache before running."
    ),
)


def _ds_write_jsonl(path: Path, rows: list[dict]) -> None:
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_json.dumps(row) + "\n")


def _build_lane_a_test_manifest(tmp_path: Path) -> Path:
    """Build a 4-section coronal manifest whose positions span ~1-9mm so
    Lane A's GT-anchored sweep has room on both sides."""
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"
    rows = [
        {
            "image_path": f"data/datasets/coronal/ds_a/subj_{i}/sec_{i:02d}.png",
            "dataset": "ds_a",
            "subject_id": f"subj_{i}",
            "section_id": f"sec_{i:02d}",
            "species": "mouse",
            "atlas": "allen_mouse_25um",
            "orientation": "coronal",
            "slice_axis": "ap",
            "position_mm": pos,
            "imaging": "brightfield",
            "staining": "Nissl",
            "exclude_from_training": False,
        }
        for i, pos in enumerate([2.0, 4.0, 6.0, 8.0])
    ]
    _ds_write_jsonl(shards_root / "coronal" / "ds_a.jsonl", rows)
    alloc_root = manifest_root / "allocations"
    rlvr_rows = [
        {
            "section_id": r["section_id"],
            "dataset": r["dataset"],
            "added_by": "test",
            "added_at": "2026-05-10T00:00:00+00:00",
        }
        for r in rows
    ]
    _ds_write_jsonl(alloc_root / "coronal" / "rlvr.jsonl", rlvr_rows)
    _ds_write_jsonl(alloc_root / "coronal" / "sft.jsonl", [])
    _ds_write_jsonl(alloc_root / "coronal" / "eval.jsonl", [])
    return manifest_root


def _stub_atlas_range_for_ds(monkeypatch: pytest.MonkeyPatch) -> None:
    from langslice_training.rl.single_turn import section_state as ss
    monkeypatch.setattr(
        ss,
        "_atlas_valid_range_mm",
        lambda atlas, plane: {
            ("allen_mouse_25um", "coronal"): (0.0, 13.2),
            ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
            ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
        }[(atlas, plane)],
    )
    monkeypatch.setattr(ds, "_atlas_in_plane_long_edge", lambda _atlas, _plane: 320)


def _seed_query_files_ds(base: Path, manifest_root: Path) -> None:
    from langslice_training.rl.single_turn.manifest_index import ManifestIndex
    idx = ManifestIndex.from_manifest_root(manifest_root, repo_root=base)
    for section in idx.query(plane="coronal"):
        target = base / section.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


@_requires_atlas_cache_ds
def test_build_datasets_from_index_randomized_lane_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_datasets_from_index(randomized=True, strategy="lane_a_prefix",
    ...)`` produces row specs whose ``image_paths`` carry the query plus
    every fetched-prefix atlas tile."""
    pytest.importorskip("torch")
    _stub_atlas_range_for_ds(monkeypatch)

    import random as _random

    from langslice_training.rl.single_turn.manifest_index import ManifestIndex

    manifest_root = _build_lane_a_test_manifest(tmp_path)
    _seed_query_files_ds(tmp_path, manifest_root)
    index = ManifestIndex.from_manifest_root(manifest_root, repo_root=tmp_path)

    train, _ = ds.build_datasets_from_index(
        manifest_index=index,
        plane="coronal",
        split="rlvr",
        repo_root=tmp_path,
        slate_root=tmp_path,
        eval_holdout_every=0,
        seed=0,
        randomized=True,
        strategy="lane_a_prefix",
        atlas_embedding_cache_dir=ATLAS_CACHE_DIR_DS_TEST,
        rng=_random.Random(7),
    )
    assert len(train) >= 1, "expected at least one randomized Lane A spec"
    # Inspect the first spec's spec dict, then materialize the row to confirm
    # the parallel image_paths column is well-formed.
    first_spec = train._specs[0]  # noqa: SLF001
    assert first_spec["dataset"] == "ds_a"
    assert len(first_spec["atlas_image_paths"]) == len(first_spec["fetched_positions_mm"])
    assert len(first_spec["atlas_image_paths"]) >= 2, (
        f"expected the Lane A prefix to carry >=2 atlas tiles, got "
        f"{len(first_spec['atlas_image_paths'])}"
    )
    # Every atlas path is rooted under the canonical layout.
    for p in first_spec["atlas_image_paths"]:
        assert p.startswith("models/langslice-gemma-4/data/atlas/")


def test_build_datasets_from_index_randomized_lane_a_requires_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an atlas_embedding_cache_dir the randomized path must raise."""
    _stub_atlas_range_for_ds(monkeypatch)

    from langslice_training.rl.single_turn.manifest_index import ManifestIndex

    manifest_root = _build_lane_a_test_manifest(tmp_path)
    _seed_query_files_ds(tmp_path, manifest_root)
    index = ManifestIndex.from_manifest_root(manifest_root, repo_root=tmp_path)

    with pytest.raises(ValueError, match="atlas_embedding_cache_dir"):
        ds.build_datasets_from_index(
            manifest_index=index,
            plane="coronal",
            split="rlvr",
            repo_root=tmp_path,
            slate_root=tmp_path,
            eval_holdout_every=0,
            seed=0,
            randomized=True,
            strategy="lane_a_prefix",
            atlas_embedding_cache_dir=None,
        )
