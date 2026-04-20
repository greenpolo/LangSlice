"""Public API of the position-estimation harness."""

from langslice.harness.estimation._types import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
)

__all__ = ["APResult", "MultiSliceResult", "PositionResult", "estimate_position"]


def estimate_position(
    image,
    atlas_name: str,
    *,
    plane: str = "coronal",
    model: str | object = "gemini-3-flash-preview",
    max_iterations: int = 20,
    **_ignored,
) -> "PositionResult":
    """Synchronous wrapper over the async runner. Sync API for CLI / eval consumers."""
    import asyncio

    from langslice.harness.estimation.runner import run_single_slice_session

    return asyncio.run(
        run_single_slice_session(
            image=image, atlas_name=atlas_name, plane=plane,  # type: ignore[arg-type]
            model=model, max_iterations=max_iterations,
        )
    )
