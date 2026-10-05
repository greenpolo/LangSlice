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


def _event(kind="tool_start", name="view_slices", targets=("b",), execution="one"):
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
    follower.on_event(_event(name="adjust_transforms", targets=("b",), execution="two"))
    assert setup.comparison.batches[-1] == [("b", 1)]
    follower.on_event(_event(targets=("b", "a"), execution="three"))
    assert setup.comparison.batches[-1] == [("b", 1), ("a", 1)]


@pytest.mark.parametrize("event", [{"kind": "seed"}, _event(name="view_stack", targets=("a", "b"))])
def test_seed_and_stack_overview_show_the_run(setup, event):
    _follower(setup).on_event(event)
    assert setup.comparison.batches == [[("a", 1), ("b", 1)]]


def test_write_refreshes_after_its_end_and_ignores_a_stale_end(setup):
    follower = _follower(setup)
    follower.on_event(_event(name="adjust_transforms", targets=("a", "b")))
    setup.slices[0].position = 4.5
    follower.on_event(_event(kind="tool_end", name="adjust_transforms", execution="stale"))
    assert len(setup.comparison.batches) == 1
    follower.on_event(_event(kind="tool_end", name="adjust_transforms"))
    assert setup.comparison.batches[-1] == [("a", 4.5), ("b", 1)]
    assert len(setup.comparison.batches) == 2


def test_looks_do_not_refresh_on_end(setup):
    follower = _follower(setup)
    follower.on_event(_event(targets=("a", "b")))
    follower.on_event(_event(kind="tool_end", targets=("a", "b")))
    assert len(setup.comparison.batches) == 1


def test_proposed_calls_and_unrelated_events_are_ignored(setup):
    follower = _follower(setup)
    follower.on_event(_event(kind="tool_call", name="set_positions"))
    follower.on_event(_event(kind="tool_end", name="set_positions"))
    follower.on_event(_event(targets=("not-in-run",)))
    follower.on_event(_event(name="note"))
    assert not setup.comparison.batches


def test_failed_viewer_creation_never_fails_the_tool(setup):
    def fail():
        raise RuntimeError("Native comparison unavailable")

    follower = _follower(setup, fail)
    follower.on_event(_event(targets=("a", "b")))
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


def test_finished_follower_ignores_later_events(setup):
    follower = _follower(setup)
    follower.finish()
    follower.on_event(_event())
    assert not setup.comparison.batches and not setup.dispatches


@pytest.mark.parametrize("name", ["view_landmarks", "edit_landmarks", "warp_landmarks"])
def test_removed_landmark_tools_do_not_drive_the_viewer(setup, name):
    _follower(setup).on_event(_event(name=name, targets=("b",)))
    assert not setup.comparison.batches
    assert name not in abba_follow._WRITES


@pytest.mark.parametrize("name, refreshed", [
    ("fit_deformable", True), ("trace_borders", True), ("mark_damaged", True),
    ("preprocess", False),
])
def test_nonlinear_damage_and_appearance_tools_are_followed(setup, name, refreshed):
    follower = _follower(setup)
    follower.on_event(_event(name=name, targets=("b",)))
    assert setup.comparison.batches == [[("b", 1.0)]]
    follower.on_event(_event(kind="tool_end", name=name))
    assert len(setup.comparison.batches) == (2 if refreshed else 1)
