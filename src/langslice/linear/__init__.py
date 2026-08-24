"""Public API of the linear (position-estimation) stage."""

from langslice.linear._types import (  # noqa: F401
    APResult,
    PositionResult,
)

__all__ = [
    "APResult",
    "PositionResult",
    "estimate_position",
]


def estimate_position(
    image,
    atlas_name: str,
    *,
    plane: str = "coronal",
    model: str | object | None = None,
    model_name: str | object | None = None,
    max_iterations: int = 20,
    media_resolution: str | None = None,
    thinking: str | None = None,
    temperature: float | None = None,
    apply_clahe: bool = True,
    debug_dir: str | None = None,
    **_ignored,
) -> "PositionResult":
    """Synchronous wrapper over the async runner. Sync API for CLI / eval consumers.

    Synchronous API; not safe to call from within a running asyncio event loop
    (e.g. Jupyter, async CLI) — :func:`asyncio.run` raises ``RuntimeError`` in
    that case.
    """
    import asyncio

    from langslice.linear.runner import run_single_slice_session
    from langslice.providers import vlm_config

    resolved_model = model_name if model_name is not None else model
    if resolved_model is None:
        resolved_model = vlm_config.MODEL_NAME

    result = asyncio.run(
        run_single_slice_session(
            image=image, atlas_name=atlas_name, plane=plane,  # type: ignore[arg-type]
            model=resolved_model,
            max_iterations=max_iterations,
            media_resolution=media_resolution,
            thinking_level=thinking or vlm_config.THINKING_LEVEL,
            temperature=(
                float(temperature)
                if temperature is not None
                else float(vlm_config.TEMPERATURE)
            ),
            apply_clahe=apply_clahe,
        )
    )
    if debug_dir is not None:
        result.debug_dir = debug_dir
    return result
