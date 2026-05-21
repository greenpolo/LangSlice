"""Tests for the --clahe-augment-fraction training augmentation knob.

CLAHE augmentation closes the train/inference preprocessing gap: production
applies langslice_harness.image_prep.adaptive_preprocess to every slice;
training did not. The flag opt-in re-applies the same helper to a configurable
random fraction of training rows, and forces the per-row slice image to skip
the precomputed query embedding cache so the live SigLIP forward sees the
augmented pixels.

These tests mock cv2/CLAHE via the production helper, so no working cv2
install is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# Match the import layout used by sibling SFT tests (tests/test_sft_*.py).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from sft import render as render_module  # noqa: E402
from sft.collate import LangSliceCollator  # noqa: E402
from sft.render import RenderedExample, RenderMetadata  # noqa: E402


def _make_metadata() -> RenderMetadata:
    return RenderMetadata(
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="stub_subject",
        system_prompt_kind="single_slice",
    )


def _build_rendered_example(
    image_paths: list[str],
    *,
    n_query_images: int = 1,
    apply_clahe: bool = False,
) -> RenderedExample:
    user_content = [{"type": "image", "image": object()} for _ in image_paths]
    user_content.append({"type": "text", "text": "Estimate the position of this slice."})
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
    ]
    return RenderedExample(
        messages=messages,
        tools=[],
        metadata=_make_metadata(),
        image_paths=image_paths,
        n_query_images=n_query_images,
        apply_clahe=apply_clahe,
    )


class _StubTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    def convert_tokens_to_ids(self, name):
        return -1


class _StubProcessor:
    tokenizer = _StubTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        n_imgs = sum(
            1
            for m in messages
            for block in (m.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "image"
        )
        token_count = max(4, len(messages) * 4)
        ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        attn = torch.ones_like(ids)
        amask = torch.zeros_like(ids)
        amask[:, 4:] = 1
        out = {
            "input_ids": ids,
            "attention_mask": attn,
            "assistant_masks": amask,
        }
        if n_imgs > 0:
            out["pixel_values"] = torch.zeros(n_imgs, 3, 4, 4)
        return out


def _build_synthetic_example(tmp_path: Path):
    """Stage a one-row Example whose query image is a plain solid-color PNG."""
    from sft.dataset import load_examples

    src = _REPO / "tests" / "fixtures" / "sft_traces" / "single_slice_minimal.jsonl"
    dest = tmp_path / "single_slice_minimal.jsonl"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        Image.new("RGB", (16, 16), color=(128, 128, 128)).save(tmp_path / name)
    return load_examples(dest)[0]


# ---------------------------------------------------------------------------
# 1) Per-row coin: fraction=0.0 leaves every row un-augmented.
# ---------------------------------------------------------------------------


def test_clahe_fraction_zero_yields_no_apply_clahe(tmp_path):
    """fraction=0.0 → byte-identical short-circuit; no row gets CLAHE."""
    from sft.render import AtlasMetaCache
    from sft.train_sft import _RenderedDataset

    example = _build_synthetic_example(tmp_path)
    examples = [example] * 100
    # When fraction is 0, the dataset must short-circuit without consulting
    # adaptive_preprocess at all. Patch it to raise so any call would fail.
    with patch.object(
        render_module, "adaptive_preprocess",
        side_effect=AssertionError("adaptive_preprocess called with fraction=0"),
    ):
        ds = _RenderedDataset(
            examples, AtlasMetaCache(),
            clahe_augment_fraction=0.0, seed=42,
        )
        for i in range(100):
            row = ds[i]
            assert row["rendered"].apply_clahe is False


# ---------------------------------------------------------------------------
# 2) Per-row coin: fraction=0.5 → roughly half of 200 rows get CLAHE.
# ---------------------------------------------------------------------------


def test_clahe_fraction_half_yields_approximately_half(tmp_path):
    """fraction=0.5, n=200 → CLAHE count must land in [80, 120]."""
    from sft.render import AtlasMetaCache
    from sft.train_sft import _RenderedDataset

    example = _build_synthetic_example(tmp_path)
    examples = [example] * 200

    sentinel_img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    with patch.object(render_module, "adaptive_preprocess", return_value=sentinel_img):
        ds = _RenderedDataset(
            examples, AtlasMetaCache(),
            clahe_augment_fraction=0.5, seed=42,
        )
        flagged = sum(1 for i in range(200) if ds[i]["rendered"].apply_clahe)
    # Tight enough to catch off-by-large (e.g. nothing fires, or everything
    # fires), loose enough not to flake under different python versions.
    assert 80 <= flagged <= 120, (
        f"expected ~100 CLAHE-flagged rows out of 200 at fraction=0.5, "
        f"got {flagged}"
    )


# ---------------------------------------------------------------------------
# 3) Collator: CLAHE row skips query_cache lookup for the slice but still
#    consults atlas_cache for atlas paths.
# ---------------------------------------------------------------------------


def test_clahe_row_skips_query_cache_lookup():
    """When example.apply_clahe is True, the query cache must NOT be consulted
    for the per-row slice image. Atlas cache lookups still run."""
    mock_atlas_cache = MagicMock()
    mock_atlas_cache.lookup_by_path.return_value = None
    mock_query_cache = MagicMock()
    mock_query_cache.lookup_by_path.return_value = None

    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=mock_atlas_cache,
        query_cache=mock_query_cache,
    )
    slice_path = "data/datasets/coronal/some_dataset/subj/section.jpg"
    atlas_path = "atlas/allen_mouse_25um/coronal/9.60mm.jpg"
    ex = _build_rendered_example(
        [slice_path, atlas_path],
        n_query_images=1,
        apply_clahe=True,
    )
    collator([ex])

    # query_cache should NOT have been consulted for the slice path. Since
    # atlas_path also isn't a query-keyed path, query_cache might still see
    # it as a fallback after the atlas miss — but the slice slot must be
    # skipped entirely.
    query_call_paths = [
        call.args[0] for call in mock_query_cache.lookup_by_path.call_args_list
    ]
    assert slice_path not in query_call_paths, (
        "query_cache.lookup_by_path was called for the slice path on a "
        "CLAHE'd row; the live SigLIP forward must see the augmented pixels."
    )

    # atlas_cache should have been consulted for BOTH paths (atlas wins on
    # collision, runs for every slot regardless of apply_clahe).
    atlas_call_paths = [
        call.args[0] for call in mock_atlas_cache.lookup_by_path.call_args_list
    ]
    assert atlas_path in atlas_call_paths
    assert slice_path in atlas_call_paths


def test_non_clahe_row_consults_query_cache_normally():
    """Sibling guard: apply_clahe=False keeps the existing chain intact."""
    mock_atlas_cache = MagicMock()
    mock_atlas_cache.lookup_by_path.return_value = None
    mock_query_cache = MagicMock()
    mock_query_cache.lookup_by_path.return_value = None

    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=mock_atlas_cache,
        query_cache=mock_query_cache,
    )
    slice_path = "data/datasets/coronal/some_dataset/subj/section.jpg"
    ex = _build_rendered_example(
        [slice_path], n_query_images=1, apply_clahe=False,
    )
    collator([ex])

    query_call_paths = [
        call.args[0] for call in mock_query_cache.lookup_by_path.call_args_list
    ]
    assert slice_path in query_call_paths


# ---------------------------------------------------------------------------
# 4) Render: CLAHE row routes through adaptive_preprocess; non-CLAHE row
#    does not.
# ---------------------------------------------------------------------------


def test_clahe_row_routes_through_adaptive_preprocess(tmp_path):
    """A row rendered with apply_clahe=True must invoke adaptive_preprocess
    exactly once on the slice image. apply_clahe=False must skip it."""
    from sft.render import AtlasMetaCache, render_example

    example = _build_synthetic_example(tmp_path)
    sentinel_img = Image.new("RGB", (16, 16), color=(255, 0, 0))

    with patch.object(
        render_module, "adaptive_preprocess", return_value=sentinel_img,
    ) as mock_clahe:
        rendered = render_example(
            example, atlas_meta_cache=AtlasMetaCache(), apply_clahe=True,
        )
    assert mock_clahe.call_count == 1, (
        f"expected adaptive_preprocess to be called once for the single "
        f"query image with apply_clahe=True, got {mock_clahe.call_count}"
    )
    # The argument must be a PIL Image (the source slice image after the
    # convert("RGB") + copy() in hydrate_image).
    first_arg = mock_clahe.call_args.args[0]
    assert isinstance(first_arg, Image.Image)
    assert rendered.apply_clahe is True
    assert rendered.n_query_images == 1

    with patch.object(
        render_module, "adaptive_preprocess", return_value=sentinel_img,
    ) as mock_clahe:
        rendered = render_example(
            example, atlas_meta_cache=AtlasMetaCache(), apply_clahe=False,
        )
    assert mock_clahe.call_count == 0, (
        f"adaptive_preprocess called {mock_clahe.call_count} times with "
        "apply_clahe=False; the helper must not run on un-augmented rows."
    )
    assert rendered.apply_clahe is False


# ---------------------------------------------------------------------------
# 5) Reproducibility: same seed + fraction → same sequence of apply_clahe.
# ---------------------------------------------------------------------------


def test_clahe_seed_reproducibility(tmp_path):
    """Two _RenderedDataset instances built with the same seed and fraction
    produce identical apply_clahe sequences when walked in the same order."""
    from sft.render import AtlasMetaCache
    from sft.train_sft import _RenderedDataset

    example = _build_synthetic_example(tmp_path)
    examples = [example] * 50
    sentinel_img = Image.new("RGB", (16, 16), color=(255, 0, 0))

    with patch.object(render_module, "adaptive_preprocess", return_value=sentinel_img):
        ds_a = _RenderedDataset(
            examples, AtlasMetaCache(),
            clahe_augment_fraction=0.3, seed=12345,
        )
        ds_b = _RenderedDataset(
            examples, AtlasMetaCache(),
            clahe_augment_fraction=0.3, seed=12345,
        )
        seq_a = [ds_a[i]["rendered"].apply_clahe for i in range(50)]
        seq_b = [ds_b[i]["rendered"].apply_clahe for i in range(50)]
    assert seq_a == seq_b, (
        "two datasets built with the same seed produced divergent CLAHE "
        f"sequences; reproducibility is broken. seq_a={seq_a} seq_b={seq_b}"
    )
    # And a different seed must produce a different sequence (otherwise the
    # RNG isn't actually seeded from --seed).
    with patch.object(render_module, "adaptive_preprocess", return_value=sentinel_img):
        ds_c = _RenderedDataset(
            examples, AtlasMetaCache(),
            clahe_augment_fraction=0.3, seed=99999,
        )
        seq_c = [ds_c[i]["rendered"].apply_clahe for i in range(50)]
    assert seq_a != seq_c, (
        "expected a different seed to produce a different CLAHE sequence "
        "at fraction=0.3"
    )


# ---------------------------------------------------------------------------
# 6) CLI validation: invalid fraction values rejected at parse time.
# ---------------------------------------------------------------------------


def test_train_sft_cli_rejects_negative_clahe_fraction():
    from sft.train_sft import _parse_args

    with pytest.raises(SystemExit):
        _parse_args([
            "--config", "x.toml",
            "--dataset", "x.jsonl",
            "--output-dir", "out",
            "--clahe-augment-fraction", "-0.1",
        ])


def test_train_sft_cli_rejects_above_one_clahe_fraction():
    from sft.train_sft import _parse_args

    with pytest.raises(SystemExit):
        _parse_args([
            "--config", "x.toml",
            "--dataset", "x.jsonl",
            "--output-dir", "out",
            "--clahe-augment-fraction", "1.5",
        ])


def test_train_sft_cli_default_is_zero():
    from sft.train_sft import _parse_args

    args = _parse_args([
        "--config", "x.toml",
        "--dataset", "x.jsonl",
        "--output-dir", "out",
    ])
    assert args.clahe_augment_fraction == 0.0
