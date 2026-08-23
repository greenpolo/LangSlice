"""Provider-agnostic helpers shared by AP estimation backends."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from langslice.image_prep import normalize_image


@dataclass
class _APLoopState:
    max_iterations: int
    estimate_result: dict[str, object] | None = None
    reasoning_log: list[dict[str, object]] = field(default_factory=list)
    turn_metrics: list[dict[str, object]] = field(default_factory=list)
    images_fetched: int = 0
    fetched_positions: list[float] = field(default_factory=list)
    saw_broad_sweep: bool = False
    saw_narrow_sweep: bool = False


@dataclass
class _GroupLoopState(_APLoopState):
    """Mutable state for the multi-slice estimation loop."""

    n_slices: int = 0
    interval_mm: float = 0.0


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to raw bytes."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=85)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _emit_trace(
    on_trace: Callable[[dict[str, object]], None] | None,
    event: dict[str, object],
) -> None:
    if on_trace:
        on_trace(event)


def _load_atlas_lazy(atlas_name: str) -> Any:
    from langslice.atlas.core import load_atlas

    return load_atlas(atlas_name)


def _get_position_range_lazy(atlas: Any) -> tuple[float, float]:
    from langslice.atlas.core import get_position_range_mm

    return get_position_range_mm(atlas)


def _fetch_atlas_slice_bytes(
    atlas: Any,
    position_mm: float,
    *,
    show_borders: bool = False,
) -> bytes:
    """Fetch a single atlas slice, normalize, and return JPEG bytes.

    Atlas slices are sent at their native resolution — no resize.
    """
    if show_borders:
        from langslice.atlas.core import get_composite_slice

        img = get_composite_slice(atlas, position_mm)
    else:
        from langslice.atlas.core import get_reference_slice

        img = get_reference_slice(atlas, position_mm)
    img = normalize_image(img)
    return _image_to_bytes(img)
