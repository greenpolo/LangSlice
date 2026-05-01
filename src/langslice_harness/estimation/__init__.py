"""Public AP-estimation compatibility exports."""
from collections.abc import Callable
from typing import cast

from langslice_harness.harness.estimation import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
    estimate_group,
    estimate_position,
)

__all__ = [
    "APResult",
    "MultiSliceResult",
    "PositionResult",
    "estimate_group",
    "estimate_position",
    "estimate_position_image_gen",
]


def estimate_position_image_gen(*args: object, **kwargs: object) -> object:
    """Lazy compatibility wrapper for the image-gen AP estimator."""

    from langslice_harness.harness.estimation.image_gen import (
        estimate_position_image_gen as _estimate_position_image_gen,
    )

    func = cast(Callable[..., object], _estimate_position_image_gen)
    return func(*args, **kwargs)
