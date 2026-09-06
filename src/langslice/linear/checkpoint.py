"""JSON checkpoint for a linear run — the record of truth.

Every write tool saves the full :class:`StackState` here, atomically (temp
file + rename) so a crash mid-write cannot leave a truncated file. Resume =
load the checkpoint and re-seed the agent with the state it had.
"""

from __future__ import annotations

import json
import os
import tempfile

from langslice.linear.state import StackState

CHECKPOINT_FILENAME = "linear_state.json"


def default_checkpoint_path(image_folder: str) -> str:
    """``<image_folder>/linear_state.json``."""
    return os.path.join(image_folder, CHECKPOINT_FILENAME)


def save_checkpoint(state: StackState, path: str) -> None:
    """Write *state* to *path* atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)
        # mkstemp creates 0600; a checkpoint is an ordinary output file.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_checkpoint(path: str) -> StackState | None:
    """Load a checkpoint, or ``None`` if there isn't one at *path*."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return StackState.from_dict(json.load(handle))
