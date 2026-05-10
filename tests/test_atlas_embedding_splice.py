"""Tests for embeddings.splice: forward-pre-hook + get_image_features patch.

These tests stub out the Gemma 4 model with a minimal nn.Module that exposes
the same surface (`vision_tower`, `embed_vision`, `get_image_features`,
``forward(**kwargs)``). Real Gemma 4 weights are not needed.

The splice handles VARIABLE per-image patch counts: cached images may each
have different patch counts (different atlas slices have different
resolutions), and uncached images (e.g. queries) typically have yet another
patch count. The fake's ``get_image_features`` mirrors the real model's
flat layout: ``(sum_patches, hidden)`` last_hidden_state.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from embeddings.splice import (
    _CACHED_FLAT_ATTR,
    _CACHED_PATCH_COUNTS_ATTR,
    _INSTALLED_ATTR,
    _MASK_ATTR,
    install_atlas_splice,
)


class _FakeBaseModelOutputWithPooling:
    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


class _FakeGemmaInner(nn.Module):
    """Smallest nn.Module that satisfies ``_find_gemma_inner``'s contract.

    Per-image patch count is derived from ``image_position_ids`` (Gemma 4
    convention: padding patches use ``(-1, -1)``).  The fake produces a
    deterministic last_hidden_state of shape ``(sum_patches, hidden)``
    with row ``j`` set to ``j`` so tests can spot-check positions.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.hidden = hidden
        self.vision_tower = nn.Identity()

        class _EmbedVision(nn.Module):
            def __call__(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
                return inputs_embeds * 2

        self.embed_vision = _EmbedVision()
        self.gif_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> _FakeBaseModelOutputWithPooling:
        self.gif_calls.append((pixel_values.clone(), image_position_ids))
        # Live SigLIP path — match the real Gemma 4 vision tower's flat
        # ``(sum_patches, hidden)`` layout. Per-image patch counts come from
        # image_position_ids' non-padding entries.
        if image_position_ids is None:
            raise RuntimeError("fake model requires image_position_ids")
        real_mask = (image_position_ids >= 0).all(dim=-1)
        per_image_patches = real_mask.sum(dim=-1)
        total = int(per_image_patches.sum().item())
        # row j has all entries = j
        rows = torch.arange(total, dtype=torch.float32).view(total, 1)
        lhs_flat = rows.expand(total, self.hidden).clone()
        pooler = lhs_flat * 2
        return _FakeBaseModelOutputWithPooling(last_hidden_state=lhs_flat, pooler_output=pooler)

    def forward(self, **kwargs: object) -> _FakeBaseModelOutputWithPooling:
        pv = kwargs.get("pixel_values")
        pid = kwargs.get("image_position_ids")
        assert isinstance(pv, torch.Tensor)
        return self.get_image_features(pv, pid)


def _make_top_model() -> _FakeGemmaInner:
    return _FakeGemmaInner()


def _build_pid(per_image_patches: list[int], max_patches: int = 8) -> torch.Tensor:
    """Build an ``(N, max_patches, 2)`` image_position_ids tensor with the given
    per-image patch counts. Real entries are arbitrary >= 0; padding is (-1, -1).
    """
    n = len(per_image_patches)
    pid = torch.full((n, max_patches, 2), -1, dtype=torch.long)
    for i, p in enumerate(per_image_patches):
        pid[i, :p] = 0  # any non-negative value marks "real"
    return pid


def test_install_atlas_splice_idempotent():
    model = _make_top_model()
    install_atlas_splice(model)
    install_atlas_splice(model)
    assert len(model._forward_pre_hooks) == 1
    assert getattr(model, _INSTALLED_ATTR) is True


def test_pre_hook_strips_three_sidecars_from_kwargs():
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.zeros(3, 8, 8)
    pid = _build_pid([4, 4, 4])
    mask = torch.tensor([True, False, True])
    cached_flat = torch.full((4 + 4, 8), 99.0)
    cached_pc = torch.tensor([4, 4], dtype=torch.long)
    model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=mask,
        precomputed_cached_flat=cached_flat,
        precomputed_cached_patch_counts=cached_pc,
    )
    assert torch.equal(getattr(model, _MASK_ATTR), mask)
    assert torch.equal(getattr(model, _CACHED_FLAT_ATTR), cached_flat)
    assert torch.equal(getattr(model, _CACHED_PATCH_COUNTS_ATTR), cached_pc)


def test_no_sidecars_clears_state_and_falls_through():
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.zeros(2, 8, 8)
    pid = _build_pid([3, 3])
    # First call: state populated
    model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=torch.tensor([True, False]),
        precomputed_cached_flat=torch.zeros(3, 8),
        precomputed_cached_patch_counts=torch.tensor([3], dtype=torch.long),
    )
    assert getattr(model, _MASK_ATTR) is not None
    # Second call without sidecars must clear state
    model(pixel_values=pv, image_position_ids=pid)
    assert getattr(model, _MASK_ATTR) is None
    assert getattr(model, _CACHED_FLAT_ATTR) is None
    assert getattr(model, _CACHED_PATCH_COUNTS_ATTR) is None


def test_splice_runs_original_only_on_uncached():
    """mask=[True,False,True,False] → original sees exactly 2 uncached rows."""
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(4, 8, 8)
    pid = _build_pid([3, 5, 3, 5])  # mixed per-image patch counts
    mask = torch.tensor([True, False, True, False])
    # cached image 0 has 3 patches, cached image 2 has 3 patches → flat (6, 8)
    cached_flat = torch.full((6, 8), 99.0)
    cached_pc = torch.tensor([3, 3], dtype=torch.long)
    out = model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=mask,
        precomputed_cached_flat=cached_flat,
        precomputed_cached_patch_counts=cached_pc,
    )
    # Original called exactly once with 2 uncached rows
    assert len(model.gif_calls) == 1
    called_pv, called_pid = model.gif_calls[0]
    assert called_pv.shape[0] == 2
    # Output is FLAT, total patches = 3+5+3+5 = 16
    assert out.last_hidden_state.shape == (16, 8)
    # Image 0 occupies rows 0-3, all should be 99.0 (cached)
    assert torch.allclose(out.last_hidden_state[0:3], cached_flat[0:3])
    # Image 1 occupies rows 3-8, fresh from fake (rows 0-4 of partial → row j has j)
    assert torch.allclose(out.last_hidden_state[3], torch.zeros(8))
    assert torch.allclose(out.last_hidden_state[4], torch.ones(8))
    # Image 2 occupies rows 8-11, all 99.0 (cached)
    assert torch.allclose(out.last_hidden_state[8:11], cached_flat[3:6])
    # Pooler is embed_vision applied to the merged flat
    assert torch.allclose(out.pooler_output, out.last_hidden_state * 2)


def test_splice_all_cached_skips_original_entirely():
    """When mask is all True, original SigLIP must not run (max compute saved)."""
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(3, 8, 8)
    pid = _build_pid([4, 4, 4])
    mask = torch.tensor([True, True, True])
    cached_flat = torch.full((4 + 4 + 4, 8), 5.0)
    cached_pc = torch.tensor([4, 4, 4], dtype=torch.long)
    out = model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=mask,
        precomputed_cached_flat=cached_flat,
        precomputed_cached_patch_counts=cached_pc,
    )
    assert model.gif_calls == []
    assert out.last_hidden_state.shape == (12, 8)
    assert torch.allclose(out.last_hidden_state, cached_flat)
    assert torch.allclose(out.pooler_output, cached_flat * 2)


def test_splice_all_uncached_falls_through():
    """No cached → patch falls through to original on the full batch."""
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(3, 8, 8)
    pid = _build_pid([3, 3, 3])
    mask = torch.tensor([False, False, False])
    cached_flat = torch.empty(0, 8)
    cached_pc = torch.tensor([], dtype=torch.long)
    out = model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=mask,
        precomputed_cached_flat=cached_flat,
        precomputed_cached_patch_counts=cached_pc,
    )
    # Original called once with full batch
    assert len(model.gif_calls) == 1
    assert model.gif_calls[0][0].shape[0] == 3


def test_splice_mask_length_mismatch_raises():
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(3, 8, 8)
    pid = _build_pid([3, 3, 3])
    bad_mask = torch.tensor([True, False])  # len 2, pv has 3
    with pytest.raises(RuntimeError, match="mask length"):
        model(
            pixel_values=pv,
            image_position_ids=pid,
            precomputed_image_mask=bad_mask,
            precomputed_cached_flat=torch.zeros(3, 8),
            precomputed_cached_patch_counts=torch.tensor([3], dtype=torch.long),
        )


def test_splice_uneven_uncached_division_raises():
    """If uncached partial output isn't divisible by N_uncached, splice raises."""
    # Use a fake whose uncached output count is NOT divisible by N_uncached.
    # The default fake derives count from pid; build an asymmetric pid where
    # total_real_patches isn't a multiple of N_uncached.
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(3, 8, 8)
    # 2 uncached images: pid says 3 + 4 = 7 real patches total, but N_uncached=2
    # doesn't divide 7. Splice raises.
    pid = _build_pid([4, 3, 4])  # image 0 cached, 1 and 2 uncached
    mask = torch.tensor([True, False, False])
    cached_flat = torch.full((4, 8), 1.0)
    cached_pc = torch.tensor([4], dtype=torch.long)
    with pytest.raises(RuntimeError, match="not divisible by N_uncached"):
        model(
            pixel_values=pv,
            image_position_ids=pid,
            precomputed_image_mask=mask,
            precomputed_cached_flat=cached_flat,
            precomputed_cached_patch_counts=cached_pc,
        )


def test_splice_requires_image_position_ids_when_active():
    """Splice path needs pid to derive uncached patch counts."""
    model = _make_top_model()
    install_atlas_splice(model)
    pv = torch.randn(2, 8, 8)
    mask = torch.tensor([True, False])
    cached_flat = torch.full((3, 8), 1.0)
    cached_pc = torch.tensor([3], dtype=torch.long)
    with pytest.raises(RuntimeError, match="requires image_position_ids"):
        model(
            pixel_values=pv,
            image_position_ids=None,
            precomputed_image_mask=mask,
            precomputed_cached_flat=cached_flat,
            precomputed_cached_patch_counts=cached_pc,
        )


def test_find_gemma_inner_raises_when_no_match():
    from embeddings.splice import _find_gemma_inner

    bare = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="could not locate the inner Gemma 4 model"):
        _find_gemma_inner(bare)
