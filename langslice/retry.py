"""Shared retry and progress-heartbeat utilities for Gemini API calls."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
HEARTBEAT_INTERVAL_S = 10.0


def format_elapsed_seconds(elapsed_s: float) -> str:
    """Format seconds as a human-readable duration string."""
    if elapsed_s < 60.0:
        return f"{elapsed_s:.1f}s"
    minutes = int(elapsed_s // 60.0)
    seconds = int(round(elapsed_s - (minutes * 60)))
    return f"{minutes}m {seconds:02d}s"


def run_with_progress_heartbeat(
    fn: Callable[[], Any],
    *,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
) -> Any:
    """Run *fn* while emitting periodic heartbeat messages via *on_progress*."""
    started_at = time.perf_counter()
    stop_event = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    if on_progress:
        on_progress(f"{request_label}: request started")
        if heartbeat_interval_s > 0:

            def _heartbeat_loop() -> None:
                while not stop_event.wait(heartbeat_interval_s):
                    elapsed_s = time.perf_counter() - started_at
                    on_progress(
                        f"{request_label}: still waiting for Gemini "
                        f"after {format_elapsed_seconds(elapsed_s)}"
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            heartbeat_thread.start()

    try:
        result = fn()
    except Exception as exc:
        elapsed_s = time.perf_counter() - started_at
        if on_progress:
            on_progress(
                f"{request_label}: request failed after {format_elapsed_seconds(elapsed_s)} "
                f"({type(exc).__name__}: {exc})"
            )
        raise
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.05)

    elapsed_s = time.perf_counter() - started_at
    if on_progress:
        on_progress(f"{request_label}: response received in {format_elapsed_seconds(elapsed_s)}")
    return result


def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
    max_retries: int = MAX_RETRIES,
    initial_backoff_s: float = INITIAL_BACKOFF_S,
) -> Any:
    """Call *fn* with exponential backoff on retryable Gemini API errors.

    Wraps each attempt in :func:`run_with_progress_heartbeat`.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        attempt_label = f"{request_label} (attempt {attempt + 1}/{max_retries + 1})"
        try:
            return run_with_progress_heartbeat(
                fn,
                request_label=attempt_label,
                on_progress=on_progress,
            )
        except Exception as exc:
            last_exc = exc
            # Check for retryable HTTP status code
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
                if attempt < max_retries:
                    delay = initial_backoff_s * (2**attempt)
                    msg = (
                        f"Gemini API error (status {status}), "
                        f"retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    logger.warning(msg)
                    if on_progress:
                        on_progress(msg)
                    time.sleep(delay)
                    continue
            # Also retry on generic connection / timeout errors
            exc_name = type(exc).__name__.lower()
            if any(kw in exc_name for kw in ("timeout", "connection", "transport")):
                if attempt < max_retries:
                    delay = initial_backoff_s * (2**attempt)
                    msg = (
                        f"Transient error ({type(exc).__name__}), "
                        f"retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    logger.warning(msg)
                    if on_progress:
                        on_progress(msg)
                    time.sleep(delay)
                    continue
            # Non-retryable error
            raise
    assert last_exc is not None
    raise last_exc
