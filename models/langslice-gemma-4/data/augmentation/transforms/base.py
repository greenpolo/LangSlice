"""Transform protocol and composition helpers.

Image arrays use HWC float32 in [0, 1] as the standard interchange format
throughout this module and all sibling transform modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TransformContext:
    """Shared per-image state threaded through the transform pipeline."""

    modality: str
    annotation_slice: np.ndarray | None
    density_map: np.ndarray | None
    tissue_mask: np.ndarray | None
    pixel_size_um: float
    tissue_class_masks: dict[str, np.ndarray] | None = None
    plane: str = "coronal"
    position_mm: float | None = None
    # Per-pixel coverage map populated by counterstain renderers, indicating
    # where the counterstain has deposited pigment. Signal renderers (NBT/BCIP
    # wash, DAB chromogen, fluorescent probes) consult this mask to confine
    # diffuse washes to substrate-only pixels and avoid double-staining nuclei.
    counterstain_signal_mask: np.ndarray | None = None
    # Per-pixel theta giving local tract direction inside the WM mask. Computed
    # once per image (lazily) from `tissue_class_masks["white_matter"]` so both
    # WM counterstain renderers and signal renderers can reuse it.
    tract_orientation: np.ndarray | None = None


@runtime_checkable
class Transform(Protocol):
    """A single composable image transform.

    Implementations receive the current image, an RNG, and a shared context.
    They must return a new HWC float32 array in [0, 1] of the same spatial
    dimensions (H, W may differ only for geometry transforms that document the change).
    """

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray: ...


def compose(transforms: list[Transform]) -> Transform:
    """Return a Transform that applies each element of *transforms* left-to-right."""

    class _Composed:
        def __call__(
            self,
            image: np.ndarray,
            *,
            rng: np.random.Generator,
            ctx: TransformContext,
        ) -> np.ndarray:
            result = image
            for t in transforms:
                result = t(result, rng=rng, ctx=ctx)
            return result

    return _Composed()
