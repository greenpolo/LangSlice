"""Executed-tool following drives the agent viewer, never the main ABBA view."""

from types import SimpleNamespace

import pytest

from langslice.hosts.integrations import abba_follow


class _Slice:
    def __init__(self, name, selected=False):
        self.name, self.selected = name, selected
        self.position = 1.0


class _Comparison:
    def __init__(self):
        self.batches = []

    def show_targets(self, targets):
        self.batches.append([(sl.name, sl.position) for sl in targets])


@pytest.fixture
def setup(monkeypatch):
    slices = [_Slice("a", True), _Slice("b"), _Slice("outside", True)]
    dispatches = []

    def dispatch(callback):
        dispatches.append(callback)
        callback()

    monkeypatch.setattr(abba_follow, "_on_edt", dispatch)
    comparison = _Comparison()
    abba = SimpleNamespace(mp=SimpleNamespace(getSlices=lambda: slices))
    return SimpleNamespace(
        abba=abba, slices=slices, comparison=comparison, dispatches=dispatches,
        mapping={sl.name: sl for sl in slices[:2]},
    )


def _follower(setup, factory=None):
    return abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=factory or (lambda: setup.comparison),
    )


def _event(kind="tool_start", name="look", targets=("b",), execution="one"):
    return dict(kind=kind, name=name, target_ids=list(targets), execution_id=execution)


def test_disparate_sections_in_large_stack_compare_without_touching_selection(setup):
    slices = [_Slice(str(index), selected=index == 7) for index in range(1, 41)]
    setup.mapping = {sl.name: sl for sl in slices}
    created = []

    def factory():
        created.append(True)
        return setup.comparison

    follower = _follower(setup, factory)
    assert not created
    follower.on_event(_event(targets=("1", "20", "40", "20", "unknown")))
    assert created == [True]
    assert setup.comparison.batches == [[("1", 1), ("20", 1), ("40", 1)]]
    assert [sl.name for sl in slices if sl.selected] == ["7"]
    follower.finish()
    assert [sl.name for sl in slices if sl.selected] == ["7"]


def test_viewer_follows_single_and_multiple_targets(setup):
    follower = _follower(setup)
    follower.on_event(_event(targets=("a", "b")))
    follower.on_event(_event(name="interactive_transform", targets=("b",), execution="two"))
    assert setup.comparison.batches[-1] == [("b", 1)]
    follower.on_event(_event(targets=("b", "a"), execution="three"))
    assert setup.comparison.batches[-1] == [("b", 1), ("a", 1)]


@pytest.mark.parametrize("event", [{"kind": "seed"}, _event(name="look", targets=("a", "b"))])
def test_seed_and_stack_overview_show_the_run(setup, event):
    _follower(setup).on_event(event)
    assert setup.comparison.batches == [[("a", 1), ("b", 1)]]


def test_write_refreshes_after_its_end_and_ignores_a_stale_end(setup):
    follower = _follower(setup)
    follower.on_event(_event(name="interactive_transform", targets=("a", "b")))
    setup.slices[0].position = 4.5
    follower.on_event(_event(kind="tool_end", name="interactive_transform", execution="stale"))
    assert len(setup.comparison.batches) == 1
    follower.on_event(_event(kind="tool_end", name="interactive_transform"))
    assert setup.comparison.batches[-1] == [("a", 4.5), ("b", 1)]
    assert len(setup.comparison.batches) == 2
    # Proposed calls and events for sections outside the run are ignored.
    follower.on_event(_event(kind="tool_call", name="position_sections"))
    follower.on_event(_event(targets=("not-in-run",)))
    follower.on_event(_event(name="status"))
    assert len(setup.comparison.batches) == 2


def test_failed_viewer_creation_never_fails_the_tool(setup):
    def fail():
        raise RuntimeError("Native comparison unavailable")

    follower = _follower(setup, fail)
    follower.on_event(_event(targets=("a", "b")))
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


