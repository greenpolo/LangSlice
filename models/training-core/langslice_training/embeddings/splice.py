"""Forward-side splice: substitute cached SigLIP outputs into Gemma 4's vision path.

Phase 2 of the atlas-embedding cache. The collator emits three sidecars per
batch when ``enable_splice=True`` and at least one cache hit is present:

    - ``precomputed_image_mask``: bool tensor of shape ``(N_total_images,)``
      where True = "this image has a cached embedding".
    - ``precomputed_cached_flat``: tensor of shape
      ``(sum_cached_patches, hidden)`` — every cached SigLIP last_hidden_state
      concatenated along dim 0 in mask-True order.
    - ``precomputed_cached_patch_counts``: int tensor of shape ``(N_cached,)``
      giving the per-image patch count, in mask-True order.

Per-image patch counts vary across atlas slices (different positions render at
different resolutions; query images vs atlas images), so a uniform-shape
sidecar would be wrong. The flat-concat layout sidesteps that.

This module installs:

  1. A forward-pre-hook on the outermost trainable module that strips the
     three sidecars from forward kwargs and stashes them on the inner Gemma 4
     model before forward proceeds.
  2. A monkey-patch of the inner ``Gemma4Model.get_image_features`` that:
       * computes per-uncached-image patch counts from
         ``image_position_ids`` padding,
       * runs the SigLIP vision tower only on the uncached subset,
       * scatters cached and fresh embeddings into a single flat
         ``last_hidden_state`` at correct per-image offsets,
       * runs ``embed_vision`` live on the merged tensor to produce the
         projected ``pooler_output`` the rest of forward expects.

When no sidecars are stashed (eval batch built outside the SFT collator,
``model.generate`` path, splice intentionally disabled) the patch falls
through to the original ``get_image_features`` with no shape change.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MASK_ATTR = "_atlas_splice_mask"
_CACHED_FLAT_ATTR = "_atlas_splice_cached_flat"
_CACHED_PATCH_COUNTS_ATTR = "_atlas_splice_cached_patch_counts"
_INSTALLED_ATTR = "_atlas_splice_installed"


def _find_gemma_inner(top_module: torch.nn.Module) -> torch.nn.Module:
    """Walk the module tree to find the Gemma 4 model that owns the vision pipeline.

    The trainable model is wrapped: PEFT ``LoraModel`` → ``base_model.model``
    → ``Gemma4ForConditionalGeneration`` → ``self.model`` →
    ``Gemma4Model``. The innermost one is the only module that has
    ``vision_tower`` AND ``embed_vision`` AND a callable
    ``get_image_features``. Match on that triple to be wrapper-agnostic.
    """
    for submodule in top_module.modules():
        if (
            hasattr(submodule, "vision_tower")
            and hasattr(submodule, "embed_vision")
            and callable(getattr(submodule, "get_image_features", None))
        ):
            return submodule
    raise RuntimeError(
        "atlas splice: could not locate the inner Gemma 4 model — no submodule "
        "has both `vision_tower`, `embed_vision`, and `get_image_features`. "
        "Verify the model loader produced a Gemma 4 architecture."
    )


def _count_real_patches_per_image(image_position_ids: torch.Tensor) -> torch.Tensor:
    """Per-image real-patch count from Gemma 4's ``(N, max_patches, 2)`` position-ids tensor.

    Padding patches use ``(-1, -1)``. A real patch has both coordinates >= 0.
    """
    real_mask = (image_position_ids >= 0).all(dim=-1)  # (N, max_patches)
    return real_mask.sum(dim=-1).to(torch.long)


def install_atlas_splice(
    top_model: torch.nn.Module,
    *,
    frozen_vision: bool = False,
) -> None:
    """Wire the cached-embedding splice into ``top_model``.

    Idempotent — calling twice on the same outer model is a no-op.

    ``frozen_vision``: when True, wrap the vision-tower + projector forward
    calls in ``torch.no_grad()``. Use this whenever the LoRA config has
    ``finetune_vision_layers=False`` — autograd tape over a frozen vision
    pipeline is pure activation-memory waste. Must flip back to False if
    vision layers are ever fine-tuned, otherwise gradients won't reach
    SigLIP / the projector.
    """
    if getattr(top_model, _INSTALLED_ATTR, False):
        return

    inner = _find_gemma_inner(top_model)
    original_get_image_features = inner.get_image_features

    # contextlib.nullcontext keeps both branches in one code path so the rest
    # of the splice doesn't need to fork on ``frozen_vision``.
    import contextlib

    def _vision_ctx() -> Any:
        return torch.no_grad() if frozen_vision else contextlib.nullcontext()

    def spliced_get_image_features(
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        mask: torch.Tensor | None = getattr(inner, _MASK_ATTR, None)
        cached_flat: torch.Tensor | None = getattr(inner, _CACHED_FLAT_ATTR, None)
        cached_pc: torch.Tensor | None = getattr(inner, _CACHED_PATCH_COUNTS_ATTR, None)

        if mask is None or cached_flat is None or cached_pc is None or not bool(mask.any().item()):
            # One-shot diagnostic: confirm whether images reach the generate /
            # eval path at all (a text-only rollout never calls this with real
            # pixels). Logs pixel_values shape the first time the fall-through
            # fires; harmless and fires at most once.
            if not getattr(inner, "_atlas_splice_logged_fallthrough", False):
                logger.warning(
                    "atlas splice FALL-THROUGH (generate/eval): pixel_values=%s "
                    "image_position_ids=%s",
                    None if pixel_values is None else tuple(pixel_values.shape),
                    None if image_position_ids is None else tuple(image_position_ids.shape),
                )
                inner._atlas_splice_logged_fallthrough = True
            with _vision_ctx():
                return original_get_image_features(
                    pixel_values, image_position_ids=image_position_ids, **kwargs
                )

        if mask.shape[0] != pixel_values.shape[0]:
            raise RuntimeError(
                f"atlas splice: mask length {mask.shape[0]} != pixel_values "
                f"batch {pixel_values.shape[0]} — collator and forward path "
                "are out of sync."
            )
        if image_position_ids is None:
            raise RuntimeError(
                "atlas splice: variable-patch-count splice requires "
                "image_position_ids to compute per-image patch counts. "
                "Pass it through the model forward."
            )
        if not getattr(inner, "_atlas_splice_logged_shapes", False):
            logger.warning(
                "atlas splice runtime shapes: pixel_values=%s "
                "image_position_ids=%s cached_flat=%s patch_counts=%s mask=%s",
                tuple(pixel_values.shape),
                tuple(image_position_ids.shape),
                tuple(cached_flat.shape),
                tuple(cached_pc.shape),
                tuple(mask.shape),
            )
            inner._atlas_splice_logged_shapes = True

        device = pixel_values.device
        mask_dev = mask.to(device=device)
        not_mask_dev = ~mask_dev
        n_total = pixel_values.shape[0]

        # Per-image OUTPUT patch counts. ``image_position_ids`` records INPUT
        # patches (pre-SigLIP pan-and-scan fan-out, typically 2394 per image),
        # but ``vision_tower`` internally pools each image down to a smaller
        # output (e.g. 266 patches/image). The cache stores the OUTPUT count;
        # we never cross-check against pid because they're different metrics.
        cached_pc_dev = cached_pc.to(device=device, dtype=torch.long)
        patch_counts_all = torch.zeros(n_total, dtype=torch.long, device=device)
        patch_counts_all[mask_dev] = cached_pc_dev

        if bool(not_mask_dev.any().item()):
            uncached_pv = pixel_values[not_mask_dev]
            uncached_pid = image_position_ids[not_mask_dev]
            n_uncached = int(uncached_pv.shape[0])

            # Fast path requires uniform INPUT real-patch counts across the
            # uncached subset so SigLIP's OUTPUT patch count is also uniform.
            # n_uncached==1 is trivially uniform. n_uncached>1 is uniform for
            # CLAHE'd queries from the same (atlas, plane) pair but breaks
            # when CLAHE'd queries from different atlases (with different
            # in_plane_long_edge downsample targets) land in the same batch.
            if n_uncached == 1:
                homogeneous = True
            else:
                real_pc = _count_real_patches_per_image(uncached_pid)
                homogeneous = bool(torch.all(real_pc == real_pc[0]).item())

            if homogeneous:
                with _vision_ctx():
                    partial = original_get_image_features(
                        uncached_pv, image_position_ids=uncached_pid, **kwargs
                    )
                partial_flat = partial.last_hidden_state
                if partial_flat.shape[0] % n_uncached != 0:
                    raise RuntimeError(
                        f"atlas splice: live partial output "
                        f"{partial_flat.shape[0]} is not divisible by "
                        f"N_uncached={n_uncached} despite homogeneous input "
                        "pids — model violated the assumption that uniform "
                        "input patch count implies uniform output."
                    )
                uncached_pc_per_image = partial_flat.shape[0] // n_uncached
                uncached_pc = torch.full(
                    (n_uncached,),
                    uncached_pc_per_image,
                    dtype=torch.long,
                    device=device,
                )
            else:
                # Heterogeneous: probe each uncached image individually so
                # we get exact per-image output counts. Same total compute,
                # split across n_uncached SigLIP calls instead of one.
                per_image_flats: list[torch.Tensor] = []
                per_image_pc_list: list[int] = []
                partial = None
                for j in range(n_uncached):
                    single_pv = uncached_pv[j : j + 1]
                    single_pid = uncached_pid[j : j + 1]
                    with _vision_ctx():
                        single_out = original_get_image_features(
                            single_pv, image_position_ids=single_pid, **kwargs
                        )
                    single_flat = single_out.last_hidden_state
                    per_image_flats.append(single_flat)
                    per_image_pc_list.append(int(single_flat.shape[0]))
                    if partial is None:
                        partial = single_out  # template for return type
                partial_flat = torch.cat(per_image_flats, dim=0)
                uncached_pc = torch.tensor(
                    per_image_pc_list,
                    dtype=torch.long,
                    device=device,
                )

            patch_counts_all[not_mask_dev] = uncached_pc
        else:
            partial = None
            partial_flat = None

        # Total patches in the merged output and per-image offsets.
        total_patches = int(patch_counts_all.sum().item())
        offsets = torch.zeros(n_total + 1, dtype=torch.long, device=device)
        offsets[1:] = patch_counts_all.cumsum(dim=0)

        # Cached source offsets within ``cached_flat``.
        cached_offsets = torch.zeros(int(cached_pc.shape[0]) + 1, dtype=torch.long, device=device)
        cached_offsets[1:] = cached_pc_dev.cumsum(dim=0)

        # Allocate the full output and scatter per-image chunks into place.
        ref = partial_flat if partial_flat is not None else cached_flat
        full_flat = torch.empty(
            (total_patches, int(ref.shape[1])),
            dtype=ref.dtype,
            device=ref.device,
        )
        cached_flat_dev = cached_flat.to(device=ref.device, dtype=ref.dtype)
        cached_idx = 0
        uncached_cursor = 0
        for i in range(n_total):
            out_lo = int(offsets[i].item())
            out_hi = int(offsets[i + 1].item())
            if bool(mask_dev[i].item()):
                src_lo = int(cached_offsets[cached_idx].item())
                src_hi = int(cached_offsets[cached_idx + 1].item())
                if (src_hi - src_lo) != (out_hi - out_lo):
                    raise RuntimeError(
                        f"atlas splice: cached patch-count {src_hi - src_lo} "
                        f"for image {i} does not match output slot "
                        f"{out_hi - out_lo}. Cache may be stale or wrong."
                    )
                full_flat[out_lo:out_hi] = cached_flat_dev[src_lo:src_hi]
                cached_idx += 1
            else:
                assert partial_flat is not None
                slot = out_hi - out_lo
                full_flat[out_lo:out_hi] = partial_flat[uncached_cursor : uncached_cursor + slot]
                uncached_cursor += slot
        if partial_flat is not None and uncached_cursor != partial_flat.shape[0]:
            raise RuntimeError(
                f"atlas splice: consumed {uncached_cursor} uncached patches "
                f"but partial output had {partial_flat.shape[0]} — internal "
                "scatter accounting is wrong."
            )

        # Run embed_vision live on the merged tensor — cheap projection
        # that may be trainable in future LoRA configs, so we never cache
        # past this point. Wrapped in the vision context: when the projector
        # is frozen (``frozen_vision=True``), autograd tape over it is just
        # activation-memory waste.
        with _vision_ctx():
            pooler = inner.embed_vision(inputs_embeds=full_flat)

        # Match the return type the original would have produced.
        if partial is not None:
            partial.last_hidden_state = full_flat
            partial.pooler_output = pooler
            return partial

        from transformers.modeling_outputs import BaseModelOutputWithPooling

        return BaseModelOutputWithPooling(
            last_hidden_state=full_flat,
            pooler_output=pooler,
        )

    inner.get_image_features = spliced_get_image_features  # type: ignore[method-assign]

    def pre_forward_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        sidecar_mask = kwargs.pop("precomputed_image_mask", None)
        sidecar_flat = kwargs.pop("precomputed_cached_flat", None)
        sidecar_pc = kwargs.pop("precomputed_cached_patch_counts", None)
        if sidecar_mask is not None and sidecar_flat is not None and sidecar_pc is not None:
            setattr(inner, _MASK_ATTR, sidecar_mask)
            setattr(inner, _CACHED_FLAT_ATTR, sidecar_flat)
            setattr(inner, _CACHED_PATCH_COUNTS_ATTR, sidecar_pc)
        else:
            # Always clear so a sidecar-free batch (eval, generate) doesn't
            # leak the previous training step's mask.
            setattr(inner, _MASK_ATTR, None)
            setattr(inner, _CACHED_FLAT_ATTR, None)
            setattr(inner, _CACHED_PATCH_COUNTS_ATTR, None)
        return args, kwargs

    top_model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    # HuggingFace ``generate()`` calls ``self._validate_model_kwargs(model_kwargs.copy())``
    # BEFORE delegating to forward (transformers/generation/utils.py around line 2360).
    # The forward pre-hook above strips the three sidecars from forward kwargs — but
    # by then generate() has already raised ValueError because the sidecar names
    # aren't in the model's forward signature.
    #
    # The PEFT-wrapped trainable model is a chain: PeftModel → LoraModel
    # (base_model) → Gemma4ForConditionalGeneration. ``generate`` may be invoked
    # on any wrapper; ``self._validate_model_kwargs`` runs on whichever class in
    # the chain inherits ``GenerationMixin`` (in practice the Gemma4
    # ForConditionalGeneration deeper in the tree, NOT ``top_model``).
    #
    # Walk the submodule tree and patch every instance that exposes
    # ``_validate_model_kwargs`` so the sidecar names are dropped from the
    # validator's copy before it inspects them. The actual ``model_kwargs``
    # dict that flows through to forward is untouched, so the pre-hook still
    # sees and strips the sidecars at forward time. Per-instance bind so the
    # patch doesn't leak to other models in the process.
    sidecar_names = (
        "precomputed_image_mask",
        "precomputed_cached_flat",
        "precomputed_cached_patch_counts",
    )

    def _make_patched_validator(original):  # type: ignore[no-untyped-def]
        def patched(model_kwargs: dict[str, Any]) -> Any:
            for name in sidecar_names:
                model_kwargs.pop(name, None)
            return original(model_kwargs)

        return patched

    patched_count = 0
    seen: set[int] = set()
    for submodule in top_model.modules():
        if id(submodule) in seen:
            continue
        seen.add(id(submodule))
        if hasattr(submodule, "_validate_model_kwargs") and callable(
            getattr(submodule, "_validate_model_kwargs", None)
        ):
            original_validate = submodule._validate_model_kwargs
            submodule._validate_model_kwargs = _make_patched_validator(  # type: ignore[method-assign]
                original_validate
            )
            patched_count += 1
    logger.info(
        "atlas splice: patched _validate_model_kwargs on %d submodule(s)",
        patched_count,
    )

    setattr(top_model, _INSTALLED_ATTR, True)
    logger.info(
        "atlas splice installed: stripped sidecars + patched "
        "Gemma4Model.get_image_features + patched generate-time kwarg validator"
    )
