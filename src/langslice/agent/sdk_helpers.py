"""Gemini File API helpers (upload polling; the SDK does not wait for ACTIVE)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def _file_state_name(file_obj: object) -> str | None:
    state = getattr(file_obj, "state", None)
    if isinstance(state, str):
        return state.upper()
    state_name = getattr(state, "name", None)
    if isinstance(state_name, str):
        return state_name.upper()
    return None


def _wait_for_uploaded_file(
    client: Any,
    *,
    file_name: str,
    timeout_s: float,
    on_progress: Callable[[str], None] | None = None,
) -> object:
    deadline = time.perf_counter() + timeout_s
    while True:
        uploaded = client.files.get(name=file_name)
        state_name = _file_state_name(uploaded)
        if state_name in {None, "ACTIVE"}:
            return uploaded
        if state_name in {"FAILED", "ERROR"}:
            raise RuntimeError(
                f"Gemini File API processing failed for {file_name}: state={state_name}"
            )
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"Gemini File API processing timed out for {file_name}: last_state={state_name}"
            )
        if on_progress:
            on_progress(
                f"Waiting for Gemini file '{file_name}' to become ACTIVE (state={state_name})"
            )
        time.sleep(0.5)
