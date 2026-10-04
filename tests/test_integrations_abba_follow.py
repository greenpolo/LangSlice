"""Executed-tool following changes the view, never the registration state."""

import sys
from types import SimpleNamespace

import pytest

from langslice.hosts.integrations import abba_follow


class _Slice:
    def __init__(self, name, selected=False):
        self.name, self.selected = name, selected
        self.position = 1.0

    def isSelected(self):
        return self.selected

    def select(self):
        self.selected = True

    def deSelect(self):
        self.selected = False

    def getRegisteredSources(self):
        return [object()]


class _View:
    def __init__(self):
        self.mode = 0
        self.overlap = 1
        self.display_mode = 0
        self.navigated = []
        self.centered = []
        self.repaints = 0
        self.shown = False
        self.channels = {}

    def setSelectedSlicesVisibility(self, value):
        self.shown = value

    def getChannelVisibility(self, sl, index):
        return self.channels.get((sl.name, index), False)

    def setSliceChannelVisibility(self, sl, index, value):
        self.channels[(sl.name, index)] = value

    def state(self):
        return SimpleNamespace(getViewerTransform=lambda: "original camera")

    def setDisplayMode(self, value):
        self.mode = value

    def setSliceDisplayMode(self, value):
        self.display_mode = value

    def getDisplayedCenter(self, sl):
        point = [0, 0 if self.overlap == 0 or self.mode == 1 else 10, sl.position]
        return SimpleNamespace(getDoublePosition=lambda axis: point[axis])

    def toggleOverlap(self):
        self.overlap = (self.overlap + 1) % 3

    def navigateSlice(self, sl):
        self.navigated.append(sl.name)

    def centerBdvViewOn(self, sl):
        self.centered.append(sl.name)

    def getBdvh(self):
        return SimpleNamespace(getViewerPanel=lambda: self)

    def requestRepaint(self):
        self.repaints += 1


@pytest.fixture
def setup(monkeypatch):
    slices = [_Slice("a", True), _Slice("b"), _Slice("outside", True)]
    view = _View()
    abba = SimpleNamespace(
        mp=SimpleNamespace(getSlices=lambda: slices), get_bdv_view=lambda: view,
    )
    dispatches, fits = [], []

    def dispatch(callback):
        dispatches.append(callback)
        callback()

    monkeypatch.setattr(abba_follow, "_on_edt", dispatch)
    monkeypatch.setattr(
        abba_follow, "_fit_camera",
        lambda abba, view, targets, **kwargs: fits.append(
            ([sl.name for sl in targets], kwargs["review"])
        ),
    )
    return SimpleNamespace(
        abba=abba, slices=slices, view=view, fits=fits, dispatches=dispatches,
        mapping={sl.name: sl for sl in slices[:2]},
    )


def _event(kind="tool_start", name="view_slices", targets=("b",), execution="one"):
    return dict(kind=kind, name=name, target_ids=list(targets), execution_id=execution)


class _Comparison:
    def __init__(self):
        self.batches = []
        self.hidden = 0

    def show_targets(self, targets):
        self.batches.append([(sl.name, sl.position) for sl in targets])

    def hide(self):
        self.hidden += 1


def test_disparate_sections_in_large_stack_compare_without_moving_main_camera(setup):
    slices = [_Slice(str(index), selected=index == 7) for index in range(1, 41)]
    setup.slices[:] = slices
    comparison = _Comparison()
    created = []

    def factory():
        created.append(True)
        return comparison

    follower = abba_follow.AbbaFollower(
        setup.abba, {sl.name: sl for sl in slices}, comparison_factory=factory,
    )
    assert not created
    follower.on_event(_event(targets=("1", "20", "40", "20", "unknown")))
    assert created == [True]
    assert comparison.batches == [[("1", 1), ("20", 1), ("40", 1)]]
    assert [sl.name for sl in slices if sl.selected] == ["7"]
    assert not setup.view.navigated and not setup.fits
    assert [sl.position for sl in slices] == [1] * 40
    follower.finish()
    assert [sl.name for sl in slices if sl.selected] == ["7"]


def test_viewer_stays_open_for_single_and_multiple_targets(setup):
    comparison = _Comparison()
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=lambda: comparison,
    )
    follower.on_event(_event(targets=("a", "b")))
    follower.on_event(_event(name="adjust_transforms", targets=("b",), execution="two"))
    assert comparison.hidden == 0
    assert setup.view.mode == 0
    assert comparison.batches[-1] == [("b", 1)]
    assert not setup.view.navigated
    follower.on_event(_event(targets=("b", "a"), execution="three"))
    assert comparison.batches[-1] == [("b", 1), ("a", 1)]
    assert not setup.view.navigated


@pytest.mark.parametrize("event", [{"kind": "seed"}, _event(name="view_stack", targets=("a", "b"))])
def test_seed_and_stack_overview_use_persistent_viewer(setup, event):
    comparison = _Comparison()
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=lambda: comparison,
    )
    follower.on_event(event)
    assert comparison.batches == [[("a", 1), ("b", 1)]]
    assert not setup.view.navigated and not setup.fits


def test_batch_transform_comparison_refreshes_after_write_and_ignores_stale_end(setup):
    comparison = _Comparison()
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=lambda: comparison,
    )
    follower.on_event(_event(name="adjust_transforms", targets=("a", "b")))
    setup.slices[0].position = 4.5
    follower.on_event(_event(kind="tool_end", name="adjust_transforms", execution="stale"))
    assert len(comparison.batches) == 1
    follower.on_event(_event(kind="tool_end", name="adjust_transforms"))
    assert comparison.batches[-1] == [("a", 4.5), ("b", 1)]
    assert len(comparison.batches) == 2
    assert not setup.view.navigated


def test_disabling_follow_hides_comparison_without_navigating(setup):
    enabled = [True]
    comparison = _Comparison()
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, enabled=lambda: enabled[0],
        comparison_factory=lambda: comparison,
    )
    follower.on_event(_event(targets=("a", "b")))
    enabled[0] = False
    follower.on_event(_event(name="adjust_transforms"))
    assert comparison.hidden == 1
    assert not setup.view.navigated
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


def test_failed_viewer_creation_leaves_main_view_untouched(setup):
    def fail():
        raise RuntimeError("Native comparison unavailable")

    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=fail,
    )
    follower.on_event(_event(targets=("a", "b")))
    assert not setup.view.navigated
    assert not setup.fits
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


def test_read_only_tool_highlights_targets_and_fits_native_positioning_overlay(setup):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)
    follower.on_event(_event(targets=("a", "b", "b", "unknown")))
    assert [sl.selected for sl in setup.slices] == [True, True, False]
    assert setup.view.mode == 0
    assert setup.view.display_mode == 0
    assert setup.view.overlap == 0
    assert setup.view.shown
    assert setup.view.channels == {("a", 0): True, ("b", 0): True}
    assert setup.view.navigated == ["a"]
    assert setup.fits == [(["a", "b"], False)]
    follower.on_event(_event(kind="tool_end", targets=("a", "b")))
    assert len(setup.fits) == 1
    assert [sl.position for sl in setup.slices] == [1, 1, 1]


def test_seed_fits_and_highlights_the_run_stack_before_the_first_tool(setup):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)
    follower.on_event({"kind": "seed", "text": "Initial section images"})
    assert [sl.selected for sl in setup.slices] == [True, True, False]
    assert setup.view.mode == 0
    assert setup.view.overlap == 0
    assert setup.fits == [(["a", "b"], False)]
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


def test_transforms_review_and_refit_after_mirror_changes(setup):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)
    follower.on_event(_event(name="adjust_transforms"))
    assert setup.view.mode == 1
    assert setup.view.display_mode == 1
    assert setup.view.centered == ["b"]
    # Simulate the independent checkpoint mirror finishing before tool_end.
    setup.slices[1].position = 4.5
    follower.on_event(_event(kind="tool_end", name="adjust_transforms"))
    assert setup.fits == [(["b"], True), (["b"], True)]
    assert setup.slices[1].position == 4.5


def test_stale_or_proposed_events_do_not_navigate(setup):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)
    follower.on_event(_event(kind="tool_call", name="set_positions"))
    follower.on_event(_event(kind="tool_end", name="set_positions"))
    assert not setup.view.navigated
    follower.on_event(_event(name="set_positions"))
    follower.on_event(_event(kind="tool_end", name="set_positions", execution="old"))
    assert len(setup.fits) == 1
    follower.on_event(_event(kind="tool_end", name="set_positions"))
    assert len(setup.fits) == 2


def test_dynamic_disable_and_finish_restore_selection_after_reorder(setup):
    enabled = [False]
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, enabled=lambda: enabled[0],
    )
    follower.on_event(_event())
    assert not setup.view.navigated
    enabled[0] = True
    follower.on_event(_event())
    setup.slices.reverse()
    enabled[0] = False
    follower.finish()
    assert {sl.name: sl.selected for sl in setup.slices} == {
        "a": True, "b": False, "outside": True,
    }
    repaint_count = setup.view.repaints
    follower.finish()
    enabled[0] = True
    follower.on_event(_event())
    assert setup.view.repaints == repaint_count


def test_unmapped_targets_and_unrelated_events_preserve_user_selection(setup):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)
    follower.on_event(_event(targets=("not-in-run",)))
    follower.on_event(_event(name="note"))
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]
    assert not setup.view.navigated


def test_display_failure_does_not_fail_tool_and_selection_can_be_restored(setup, monkeypatch):
    follower = abba_follow.AbbaFollower(setup.abba, setup.mapping)

    def fail(*args):
        raise RuntimeError("Viewer closed")

    monkeypatch.setattr(setup.view, "navigateSlice", fail)
    follower.on_event(_event())
    follower.finish()
    assert [sl.selected for sl in setup.slices] == [True, False, True]


@pytest.mark.parametrize("review", [False, True])
def test_camera_animation_fits_tiles_with_correct_review_sampling_plane(monkeypatch, review):
    class Affine:
        def scale(self, value):
            self.scale_value = value

        def translate(self, *values):
            self.translation = values

    animations = []
    panel = SimpleNamespace(
        getWidth=lambda: 1200, getHeight=lambda: 800,
        setTransformAnimator=animations.append,
    )
    view = SimpleNamespace(
        getBdvh=lambda: SimpleNamespace(getViewerPanel=lambda: panel),
        getDisplayedCenter=lambda sl: SimpleNamespace(
            getDoublePosition=lambda axis: sl[axis],
        ),
    )
    monkeypatch.setitem(sys.modules, "scyjava", SimpleNamespace(jimport=lambda name: {
        "net.imglib2.realtransform.AffineTransform3D": Affine,
        "bdv.viewer.animate.SimilarityTransformAnimator": lambda *args: args,
    }[name]))
    abba = SimpleNamespace(mp=SimpleNamespace(sX=10, sY=8))
    targets = [(0, 0, 4), (20, 0, 5)]
    abba_follow._fit_camera(abba, view, targets, review=review, start_transform="start")
    start, final, center_x, center_y, duration = animations[0]
    scale = min(1200 / (10 if review else 30), 800 / 8) / 1.12
    assert start == "start"
    assert (center_x, center_y, duration) == (0, 0, 250)
    assert final.scale_value == pytest.approx(scale)
    assert final.translation == pytest.approx((600 - scale * (0 if review else 10),
                                              400, -scale * 4))


@pytest.mark.parametrize("name", ["view_landmarks", "edit_landmarks", "warp_landmarks"])
def test_removed_landmark_tools_do_not_drive_native_comparison(setup, name):
    comparison = _Comparison()
    follower = abba_follow.AbbaFollower(
        setup.abba, setup.mapping, comparison_factory=lambda: comparison,
    )
    follower.on_event(_event(name=name, targets=("b",)))
    assert not comparison.batches
    assert not setup.view.navigated and not setup.fits
    assert name not in abba_follow._WRITES
