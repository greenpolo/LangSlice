"""Node-runner behaviour: ordering, routing, loop bounds, resume, failures."""

import asyncio
from pathlib import Path

import pytest

from langslice.linear.whole_brain.checkpoint import load_checkpoint
from langslice.linear.whole_brain.engine import (
    CYCLE_LIMITS,
    EngineContext,
    Node,
    build_context,
    run_nodes,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState


def _ctx(tmp_path: Path) -> EngineContext:
    return build_context(BrainConfig(image_folder=str(tmp_path)), emit=lambda _msg: None)


def _graph(routes: dict[str, list[str]], log: list[str]) -> list[Node]:
    """Build fake nodes named like the real graph.

    ``routes[name]`` is the sequence of targets that node returns on
    successive calls; it falls back to "" (default successor) once exhausted.
    """

    def make(name: str):
        async def node(state: StackState, ctx: EngineContext) -> str:
            log.append(name)
            pending = routes.get(name, [])
            return pending.pop(0) if pending else ""

        return node

    names = ["ingest", "survey", "fix", "seed", "position", "transforms", "review", "emit"]
    return [(name, make(name)) for name in names]


def _run(nodes: list[Node], ctx: EngineContext, state: StackState, **kwargs) -> StackState:
    return asyncio.run(run_nodes(nodes, state, ctx, **kwargs))


def test_runs_every_node_once_in_order(tmp_path: Path):
    log: list[str] = []
    state = _run(_graph({}, log), _ctx(tmp_path), StackState())
    assert log == [
        "ingest", "survey", "fix", "seed", "position", "transforms", "review", "emit",
    ]
    assert state.completed_nodes == log
    assert all(count == 1 for count in state.node_cycles.values())


def test_checkpoint_written_after_every_node(tmp_path: Path):
    ctx = _ctx(tmp_path)
    seen: list[list[str]] = []

    async def watcher(state: StackState, _ctx: EngineContext) -> str:
        saved = load_checkpoint(ctx.checkpoint_path)
        seen.append(list(saved.completed_nodes) if saved else [])
        return ""

    nodes: list[Node] = [("ingest", watcher), ("survey", watcher), ("emit", watcher)]
    _run(nodes, ctx, StackState())
    assert seen == [[], ["ingest"], ["ingest", "survey"]]


def test_fix_can_route_back_to_survey(tmp_path: Path):
    log: list[str] = []
    state = _run(_graph({"fix": ["survey"]}, log), _ctx(tmp_path), StackState())
    assert log[:5] == ["ingest", "survey", "fix", "survey", "fix"]
    assert state.node_cycles["survey"] == 2


def test_review_can_route_back_to_position(tmp_path: Path):
    log: list[str] = []
    _run(_graph({"review": ["position"]}, log), _ctx(tmp_path), StackState())
    assert log.count("position") == 2
    assert log[-1] == "emit"


def test_survey_loop_is_bounded(tmp_path: Path):
    log: list[str] = []
    state = _run(_graph({"fix": ["survey"] * 10}, log), _ctx(tmp_path), StackState())
    assert log.count("survey") == CYCLE_LIMITS["survey"]
    assert state.node_cycles["survey"] == CYCLE_LIMITS["survey"]
    assert log[-1] == "emit"
    assert any("cycle limit" in note for note in state.notes)


def test_position_self_loop_is_bounded(tmp_path: Path):
    log: list[str] = []
    _run(_graph({"position": ["position"] * 10}, log), _ctx(tmp_path), StackState())
    assert log.count("position") == CYCLE_LIMITS["position"]
    assert log[-1] == "emit"


def test_resume_skips_completed_nodes(tmp_path: Path):
    log: list[str] = []
    state = StackState(completed_nodes=["ingest", "survey", "fix"])
    _run(_graph({}, log), _ctx(tmp_path), state)
    assert log == ["seed", "position", "transforms", "review", "emit"]


def test_resume_false_reruns_everything(tmp_path: Path):
    log: list[str] = []
    state = StackState(completed_nodes=["ingest", "survey"])
    _run(_graph({}, log), _ctx(tmp_path), state, resume=False)
    assert log[0] == "ingest"
    assert len(log) == 8


def test_completed_node_reruns_when_routed_back(tmp_path: Path):
    """A skipped-on-resume node still runs if this run loops back to it."""
    log: list[str] = []
    state = StackState(completed_nodes=["ingest", "survey"])
    _run(_graph({"fix": ["survey"]}, log), _ctx(tmp_path), state)
    assert log == [
        "fix", "survey", "fix", "seed", "position", "transforms", "review", "emit",
    ]


def test_unknown_route_target_raises(tmp_path: Path):
    log: list[str] = []
    with pytest.raises(ValueError, match="unknown node"):
        _run(_graph({"survey": ["nowhere"]}, log), _ctx(tmp_path), StackState())


def test_node_failure_names_the_node_and_checkpoints(tmp_path: Path):
    ctx = _ctx(tmp_path)

    async def ok(state: StackState, _ctx: EngineContext) -> str:
        state.notes.append("ingested")
        return ""

    async def boom(state: StackState, _ctx: EngineContext) -> str:
        raise RuntimeError("model exploded")

    nodes: list[Node] = [("ingest", ok), ("survey", boom)]
    with pytest.raises(RuntimeError, match="node 'survey' failed"):
        _run(nodes, ctx, StackState())

    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None
    assert saved.notes == ["ingested"]
    assert saved.completed_nodes == ["ingest"]
