"""Node-runner behaviour: ordering, routing, loop bounds, resume, failures."""

import asyncio
from pathlib import Path

import pytest

from langslice.linear.whole_brain.checkpoint import load_checkpoint
from langslice.linear.whole_brain.engine import (
    CYCLE_LIMITS,
    REWINDABLE_NODES,
    EngineContext,
    Node,
    build_context,
    rewind_state,
    run_nodes,
)
from langslice.linear.whole_brain.state import BrainConfig, SliceState, StackState


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


def test_forward_jump_books_the_skipped_nodes_as_complete(tmp_path: Path):
    """A clean survey routes to seed; fix must not re-run on the next resume."""
    log: list[str] = []
    state = _run(_graph({"survey": ["seed"]}, log), _ctx(tmp_path), StackState())
    assert "fix" not in log
    assert state.completed_nodes[:4] == ["ingest", "survey", "fix", "seed"]
    assert state.node_cycles.get("fix") is None


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


# --- rewind_state ----------------------------------------------------------


_ALL_NODES = ["ingest", "survey", "fix", "seed", "position", "transforms", "review", "emit"]


def _completed_state() -> StackState:
    """A checkpoint as if a full run just finished, with something on every field."""
    return StackState(
        image_folder="/tmp/brain",
        atlas="allen_mouse_25um",
        plane="coronal",
        axis_directions={"ap": "anterior_to_posterior"},
        interval_mm=0.2,
        thickness_mm=0.05,
        keep_order=False,
        interval_breaks=[3, 7],
        notes=[
            "ingest: 2 slices",
            "survey: clean",
            "seed: no automatic seeding available",
            "position: ladder is 1.05, 1.25, 1.45 mm",
            "transforms: 2 affine fit(s)",
            "review: approved",
        ],
        contact_sheet="/tmp/brain/contact_sheet.png",
        slices=[
            SliceState(
                id="s1.png",
                index_original=0,
                index_corrected=1,
                flip=True,
                damaged=True,
                damage_note="torn cortex",
                position_mm=1.25,
                position_source="refined",
                interactive_transform={"rotation_deg": 2.0},
                confidence="medium",
                caveats=[
                    "affine failed for neighbour fit",
                    "interactive transform: manual nudge",
                    "damaged: transform is approximate",
                ],
            ),
            SliceState(
                id="s2.png",
                index_original=1,
                index_corrected=0,
                position_mm=0.9,
                position_source="refined",
                confidence="high",
                affine=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                caveats=["weak affine fit (silhouette overlap 0.55)"],
            ),
        ],
        completed_nodes=list(_ALL_NODES),
        node_cycles={name: (2 if name in ("survey", "position") else 1) for name in _ALL_NODES},
    )


def test_rewind_unsupported_node_raises():
    with pytest.raises(ValueError, match="survey"):
        rewind_state(_completed_state(), "survey")


def test_rewind_position_clears_position_and_downstream_fields():
    state = _completed_state()
    rewind_state(state, "position")

    for record in state.slices:
        assert record.position_mm is None
        assert record.position_source == ""
        assert record.confidence == ""
        assert record.affine is None
        assert record.interactive_transform is None
        assert not any(
            c.startswith(("affine failed", "weak affine fit", "interactive transform"))
            for c in record.caveats
        )
    assert state.interval_breaks == []
    assert state.notes[-1] == "rewound from 'position' for a fresh pass"

    # Untracked caveat (a review flag_slice free-text note) survives.
    s1 = state.by_id("s1.png")
    assert s1 is not None
    assert "damaged: transform is approximate" in s1.caveats


def test_rewind_preserves_survey_outputs():
    state = _completed_state()
    rewind_state(state, "position")

    s1 = state.by_id("s1.png")
    assert s1 is not None
    assert s1.flip is True
    assert s1.damaged is True
    assert s1.damage_note == "torn cortex"
    assert s1.index_corrected == 1
    s2 = state.by_id("s2.png")
    assert s2 is not None
    assert s2.index_corrected == 0
    assert state.axis_directions == {"ap": "anterior_to_posterior"}
    assert state.notes[:3] == [
        "ingest: 2 slices",
        "survey: clean",
        "seed: no automatic seeding available",
    ]


def test_rewind_drops_the_rewound_steps_own_notes():
    """The rejected pass's ladder must not seed the fresh one."""
    state = _completed_state()
    rewind_state(state, "position")

    assert state.notes == [
        "ingest: 2 slices",
        "survey: clean",
        "seed: no automatic seeding available",
        "rewound from 'position' for a fresh pass",
    ]

    # Rewinding a later node leaves the earlier steps' notes in place.
    later = _completed_state()
    rewind_state(later, "review")
    assert any(note.startswith("position:") for note in later.notes)
    assert any(note.startswith("transforms:") for note in later.notes)
    assert not any(note.startswith("review:") for note in later.notes)


def test_rewind_transforms_leaves_position_alone():
    state = _completed_state()
    rewind_state(state, "transforms")

    s1 = state.by_id("s1.png")
    assert s1 is not None
    assert s1.position_mm == 1.25
    assert s1.position_source == "refined"
    assert s1.confidence == "medium"
    assert s1.interactive_transform is None
    assert state.interval_breaks == [3, 7]

    s2 = state.by_id("s2.png")
    assert s2 is not None
    assert s2.affine is None
    assert not any(c.startswith("weak affine fit") for c in s2.caveats)


def test_rewind_review_touches_no_slice_fields():
    state = _completed_state()
    rewind_state(state, "review")

    s1 = state.by_id("s1.png")
    assert s1 is not None
    assert s1.position_mm == 1.25
    assert s1.interactive_transform == {"rotation_deg": 2.0}
    s2 = state.by_id("s2.png")
    assert s2 is not None
    assert s2.affine == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("node", "expected_completed"),
    [
        ("position", ["ingest", "survey", "fix", "seed"]),
        ("transforms", ["ingest", "survey", "fix", "seed", "position"]),
        ("review", ["ingest", "survey", "fix", "seed", "position", "transforms"]),
    ],
)
def test_rewind_trims_completed_nodes_in_graph_order(node, expected_completed):
    state = _completed_state()
    rewind_state(state, node)
    assert state.completed_nodes == expected_completed


def test_rewind_zeroes_node_cycles_for_rewound_nodes_only():
    state = _completed_state()
    rewind_state(state, "position")

    for name in ("position", "transforms", "review", "emit"):
        assert state.node_cycles[name] == 0
    assert state.node_cycles["ingest"] == 1
    assert state.node_cycles["survey"] == 2
    assert state.node_cycles["fix"] == 1
    assert state.node_cycles["seed"] == 1


def test_rewind_trims_completed_nodes_regardless_of_list_order():
    """completed_nodes need not be in canonical order; the cut is by node identity."""
    state = _completed_state()
    state.completed_nodes = [
        "fix", "ingest", "survey", "position", "seed", "review", "transforms", "emit",
    ]
    rewind_state(state, "transforms")
    assert set(state.completed_nodes) == {"ingest", "survey", "fix", "seed", "position"}


def test_rewindable_nodes_are_position_transforms_review():
    assert REWINDABLE_NODES == ("position", "transforms", "review")
