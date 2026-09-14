"""JSON checkpoint for a linear run — the record of truth.

Every write tool saves the full :class:`StackState` here, atomically (temp
file + rename) so a crash mid-write cannot leave a truncated file. Resume =
load the checkpoint and re-seed the agent with the state it had.

Every tool write funnels through :func:`save_checkpoint`, which makes it the
one place a host can watch a run live: :func:`observe_checkpoints` registers a
callback fired with the state after every write, which is what the ABBA
mirror (:mod:`langslice.integrations.abba_linear`) attaches to.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator

from langslice.linear.state import StackState

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "linear_state.json"

#: Called with the state after every atomic write. A list, not a single slot,
#: so nested runs (tests, a resumed session) can each hold their own observer
#: without clobbering another's.
_observers: list[Callable[[StackState], None]] = []


def default_checkpoint_path(image_folder: str) -> str:
    """``<image_folder>/linear_state.json``."""
    return os.path.join(image_folder, CHECKPOINT_FILENAME)


def save_checkpoint(state: StackState, path: str) -> None:
    """Write *state* to *path* atomically, then notify any observers.

    An observer that raises is logged and skipped — a display glitch (the
    ABBA mirror hiccuping on a JPype call) must never break the agent's
    write, which has already happened by the time observers run.
    """
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
    for observer in list(_observers):
        try:
            observer(state)
        except Exception:
            logger.exception("Checkpoint observer raised; ignoring")


@contextlib.contextmanager
def observe_checkpoints(fn: Callable[[StackState], None]) -> Iterator[None]:
    """Call *fn* with the state after every :func:`save_checkpoint` write.

    A host wanting a live view of a run (the ABBA mirror) wraps its run in
    this; *fn* fires inline, on the writer's thread, right after the atomic
    replace — see :func:`save_checkpoint` for what happens if it raises.
    """
    _observers.append(fn)
    try:
        yield
    finally:
        _observers.remove(fn)


def load_checkpoint(path: str) -> StackState | None:
    """Load a checkpoint, or ``None`` if there isn't one at *path*."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return StackState.from_dict(json.load(handle))
