"""The whole-brain node engine: an ordered graph of steps over one StackState.

The graph is Nash's decision tree, normalized::

    ingest -> survey -> fix -> survey ...        (fix loops back until clean)
                     -> seed -> position         (position may self-loop)
                     -> transforms -> review     (review may route back)
                     -> emit

Plain-Python runner on purpose: ADK's workflow package is underscore-private
and its resume machinery is young, while our JSON checkpoint is the record of
truth. The node contract (``async (state, ctx) -> next_node_name``) is kept
narrow so swapping the runner later stays possible.

Routing: a node returns "" for its default successor, or the name of another
node. Backward edges (a loop) are bounded per node by ``CYCLE_LIMITS``; once a
node has been entered its limit number of times, further loops into it are
refused and the engine falls through to the default successor.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langslice.atlas.core import load_atlas
from langslice.linear.whole_brain.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "brain_results.json"

NodeFn = Callable[["StackState", "EngineContext"], Awaitable[str]]
Node = tuple[str, NodeFn]

#: How many times a node may be *entered* in one run. Anything not listed
#: here runs once; loops back into it are refused.
CYCLE_LIMITS: dict[str, int] = {
    "survey": 3,
    "position": 3,
    "review": 2,
}
_DEFAULT_CYCLE_LIMIT = 1


def _log_progress(message: str) -> None:
    logger.info(message)


@dataclass
class EngineContext:
    """Everything a node needs that is not slice state."""

    config: BrainConfig
    image_folder: str
    checkpoint_path: str
    results_path: str
    model: str | None = None
    emit: Callable[[str], None] = _log_progress
    #: Atlas accessor, injectable so tests (and offline hosts) can supply one.
    atlas_loader: Callable[[str], Any] = load_atlas

    def progress(self, message: str) -> None:
        self.emit(message)
        logger.info(message)

    def image_path(self, slice_id: str) -> str:
        """Absolute path of a slice image. Slice ids are folder-relative names."""
        return os.path.join(self.image_folder, slice_id)


def build_context(
    config: BrainConfig,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
) -> EngineContext:
    """Assemble an :class:`EngineContext` from a request."""
    folder = os.path.abspath(config.image_folder)
    return EngineContext(
        config=config,
        image_folder=folder,
        checkpoint_path=default_checkpoint_path(folder),
        results_path=config.out or os.path.join(folder, RESULTS_FILENAME),
        model=config.model,
        emit=emit or _log_progress,
        atlas_loader=atlas_loader or load_atlas,
    )


async def run_nodes(
    nodes: Sequence[Node],
    state: StackState,
    ctx: EngineContext,
    *,
    resume: bool = True,
    stop_after: str | None = None,
) -> StackState:
    """Drive *nodes* over *state*, checkpointing after each one.

    Nodes already listed in ``state.completed_nodes`` from an earlier run are
    skipped when *resume* is set — but only until something in this run routes
    to them, at which point they run again.

    ``state.node_cycles`` is cumulative across resumes on purpose: a run that
    already spent its loop budget on a node does not get a fresh one by being
    restarted.

    *stop_after* halts the run once that node completes (its checkpoint is
    already written) — the per-step measurement seam: run one node on a saved
    checkpoint, then look at what it did.
    """
    order = [name for name, _ in nodes]
    fns = dict(nodes)
    if stop_after is not None and stop_after not in fns:
        raise ValueError(f"stop_after names an unknown node {stop_after!r}")
    # Nodes this run has run or been explicitly routed to: never resume-skipped.
    claimed: set[str] = set()
    index = 0

    while index < len(order):
        name = order[index]

        if resume and name not in claimed and name in state.completed_nodes:
            ctx.progress(f"[{name}] already complete, skipping")
            index += 1
            continue

        entered = state.node_cycles.get(name, 0) + 1
        state.node_cycles[name] = entered
        claimed.add(name)
        ctx.progress(f"[{name}] running" + (f" (cycle {entered})" if entered > 1 else ""))

        try:
            target = await fns[name](state, ctx)
        except Exception as exc:
            save_checkpoint(state, ctx.checkpoint_path)
            raise RuntimeError(f"whole-brain node {name!r} failed: {exc}") from exc

        state.mark_complete(name)
        save_checkpoint(state, ctx.checkpoint_path)

        if name == stop_after:
            ctx.progress(f"[{name}] --stop-after: halting; checkpoint written")
            return state

        if not target:
            index += 1
            continue
        if target not in fns:
            raise ValueError(f"node {name!r} routed to unknown node {target!r}")

        target_index = order.index(target)
        if target_index <= index:
            limit = CYCLE_LIMITS.get(target, _DEFAULT_CYCLE_LIMIT)
            if state.node_cycles.get(target, 0) >= limit:
                ctx.progress(
                    f"[{name}] loop back to {target!r} refused "
                    f"(cycle limit {limit} reached); continuing"
                )
                state.notes.append(
                    f"{name}: loop back to {target} refused at cycle limit {limit}"
                )
                index += 1
                continue
        else:
            # A forward jump rules the nodes in between out for this run
            # (survey skipping fix when the stack is clean). Book them as done
            # or a resumed run would walk straight back into them.
            for skipped in order[index + 1:target_index]:
                state.mark_complete(skipped)
                ctx.progress(f"[{name}] skipping {skipped!r}: nothing to do")
        claimed.add(target)
        index = target_index

    return state


async def run_brain(
    config: BrainConfig,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
    stop_after: str | None = None,
) -> StackState:
    """Run the whole-brain engine and return the final stack state.

    *stop_after* halts the run after that node completes; combined with the
    default resume behaviour it runs the graph one step at a time.
    """
    # Imported here: nodes.py imports EngineContext from this module.
    from langslice.linear.whole_brain.nodes import NODES

    ctx = build_context(config, emit=emit, atlas_loader=atlas_loader)

    state: StackState | None = None
    if config.resume:
        state = load_checkpoint(ctx.checkpoint_path)
        if state is not None:
            ctx.progress(
                f"Resuming from {ctx.checkpoint_path} "
                f"(completed: {', '.join(state.completed_nodes) or 'nothing'})"
            )
    if state is None:
        state = StackState()

    return await run_nodes(
        NODES, state, ctx, resume=config.resume, stop_after=stop_after
    )
