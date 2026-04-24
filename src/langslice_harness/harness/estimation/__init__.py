"""Public API of the position-estimation harness."""

from langslice_harness.harness.estimation._types import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
)

__all__ = [
    "APResult",
    "MultiSliceResult",
    "PositionResult",
    "estimate_group",
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

    from langslice_harness import vlm_config
    from langslice_harness.harness.estimation.runner import run_single_slice_session

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


def estimate_group(
    images,
    atlas_name: str,
    interval_um: int,
    thickness_um: int = 50,
    *,
    model_name: str | object | None = None,
    max_iterations: int = 25,
    plane: str = "coronal",
    # Legacy kwargs — accepted for call-site compat. Supported ones are forwarded.
    send_individually: bool = True,  # ADK always uses individual atlas images
    on_progress=None,  # ADK has its own event stream; legacy progress hook not plumbed
    media_resolution: str | None = None,
    show_borders: bool = False,  # ADK agent does not expose border toggling
    debug_dir: str | None = None,  # ADK runner writes traces via its own channels
    **_ignored,
) -> "MultiSliceResult":
    """Synchronous wrapper over :func:`run_group_session`.

    Mirrors the legacy Google SDK group-estimation signature so existing callers keep
    working through the ADK migration.

    The ``interval_um`` → ``interval_mm`` conversion happens here; the runner is
    millimetre-native.  ``model_name=None`` falls through to the runner default
    rather than being pinned here — the runner owns that default.

    Accepts legacy kwargs for compat with the pre-ADK call sites; unsupported
    kwargs are ignored, while ``media_resolution`` is forwarded to the runner.

    Synchronous API; not safe to call from within a running asyncio event loop
    (e.g. Jupyter, async CLI) — :func:`asyncio.run` raises ``RuntimeError`` in
    that case.
    """
    import asyncio

    from langslice_harness.harness.estimation.runner import run_group_session

    interval_mm = interval_um / 1000.0

    # Route through run_group_session, letting it apply its own default model
    # when ``model_name`` is None.
    kwargs: dict[str, object] = {
        "images": images,
        "atlas_name": atlas_name,
        "interval_mm": interval_mm,
        "thickness_um": thickness_um,
        "plane": plane,
        "max_iterations": max_iterations,
    }
    if media_resolution is not None:
        kwargs["media_resolution"] = media_resolution
    if model_name is not None:
        kwargs["model"] = model_name

    return asyncio.run(run_group_session(**kwargs))  # type: ignore[arg-type]
