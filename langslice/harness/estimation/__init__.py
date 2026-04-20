"""Public API of the position-estimation harness."""

from langslice.harness.estimation._types import (  # noqa: F401
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
    model: str | object = "gemini-3-flash-preview",
    max_iterations: int = 20,
    **_ignored,
) -> "PositionResult":
    """Synchronous wrapper over the async runner. Sync API for CLI / eval consumers.

    Synchronous API; not safe to call from within a running asyncio event loop
    (e.g. Jupyter, async CLI) — :func:`asyncio.run` raises ``RuntimeError`` in
    that case.
    """
    import asyncio

    from langslice.harness.estimation.runner import run_single_slice_session

    return asyncio.run(
        run_single_slice_session(
            image=image, atlas_name=atlas_name, plane=plane,  # type: ignore[arg-type]
            model=model, max_iterations=max_iterations,
        )
    )


def estimate_group(
    images,
    atlas_name: str,
    interval_um: int,
    thickness_um: int = 50,
    *,
    model_name: str | object | None = None,
    max_iterations: int = 25,
    plane: str = "coronal",
    # Legacy kwargs — accepted and ignored for call-site compat. See notes below.
    send_individually: bool = True,  # ADK always uses individual atlas images
    on_progress=None,  # ADK has its own event stream; legacy progress hook not plumbed
    media_resolution: str | None = None,  # set by the agent builder (MEDIUM for 25um)
    show_borders: bool = False,  # ADK agent does not expose border toggling
    debug_dir: str | None = None,  # ADK runner writes traces via its own channels
    **_ignored,
) -> "MultiSliceResult":
    """Synchronous wrapper over :func:`run_group_session`.

    Mirrors the legacy :func:`langslice.estimation.google.ap_multi_slice.estimate_group`
    signature so existing callers (``eval/eval_group.py``, ``langslice/cli.py``) keep
    working through the ADK migration.

    The ``interval_um`` → ``interval_mm`` conversion happens here; the runner is
    millimetre-native.  ``model_name=None`` falls through to the runner default
    rather than being pinned here — the runner owns that default.

    Accepts and ignores legacy kwargs (``send_individually``, ``on_progress``, etc.)
    for compat with the pre-ADK call sites; they have no new-runner equivalent.

    Synchronous API; not safe to call from within a running asyncio event loop
    (e.g. Jupyter, async CLI) — :func:`asyncio.run` raises ``RuntimeError`` in
    that case.
    """
    import asyncio

    from langslice.harness.estimation.runner import run_group_session

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
    if model_name is not None:
        kwargs["model"] = model_name

    return asyncio.run(run_group_session(**kwargs))  # type: ignore[arg-type]
