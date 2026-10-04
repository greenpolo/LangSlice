"""ABBA menu choices and background-run lifecycle without starting a JVM."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from langslice.hosts.integrations.abba_gui import MenuController, MenuSettings, protocol_defaults


def _abba(positions=(), thicknesses=()):
    slices = [
        SimpleNamespace(
            getSlicingAxisPosition=lambda p=p: p,
            getThicknessInMm=lambda t=t: t,
        )
        for p, t in zip(positions, thicknesses, strict=True)
    ]
    return SimpleNamespace(mp=SimpleNamespace(getSlices=lambda: slices))


def test_protocol_defaults_read_unsorted_spacing_and_calibrated_thickness():
    abba = _abba([0.8, 0.0, 0.2, 0.4, 0.4], [0.04] * 5)
    assert protocol_defaults(abba) == (200, 40)


def test_protocol_defaults_ignore_invalid_thickness_and_use_empty_fallback():
    assert protocol_defaults(_abba()) == (200, 50)
    abba = _abba([0, 0.12, 0.24, 0.36], [float("nan"), -1, 0, 0.03])
    assert protocol_defaults(abba) == (120, 30)


def test_protocol_choices_are_read_when_run_is_started_and_can_be_overridden():
    abba = _abba([0, 0.3], [0.06, 0.06])
    settings = MenuSettings(reorder=False, transform=False)
    spec = settings.to_spec(abba)
    assert spec.tasks == ["position"]
    assert (spec.position.interval_um, spec.position.thickness_um) == (300, 60)
    settings.interval_um = 150
    settings.thickness_um = 40
    settings.strict_interval = True
    spec = settings.to_spec(abba)
    assert (spec.position.interval_um, spec.position.thickness_um) == (150, 40)
    assert spec.position.strict_interval


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (MenuSettings(reorder=False, position=False, transform=False), "at least one task"),
        (MenuSettings(interactive=False, automatic=False), "interactive adjustment"),
        (MenuSettings(model="  "), "Choose a model"),
    ],
)
def test_invalid_choices_are_rejected_before_start(settings, message):
    controller = MenuController(_abba(), settings)
    with pytest.raises(ValueError, match=message):
        controller.start()
    assert not controller.running


def test_disabled_transform_does_not_require_fitting_choices():
    settings = MenuSettings(
        transform=False, interactive=False, automatic=False,
        flip=False, hemisphere_cue="left notch", facts="First fact\n\nSecond fact",
    )
    spec = settings.to_spec(_abba())
    assert spec.tasks == ["reorder", "position"]
    assert not spec.reorder.flip
    assert spec.reorder.hemisphere_cue == "left notch"
    assert spec.facts == ["First fact", "Second fact"]


def test_controller_prevents_overlapping_runs_and_snapshots_settings(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    observed = []
    state = SimpleNamespace(submitted=True)

    def run(abba, spec, **kwargs):
        observed.append(spec)
        entered.set()
        assert release.wait(5)
        return state

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    controller = MenuController(_abba())
    statuses = []
    thread = controller.start(on_status=statuses.append)
    try:
        assert entered.wait(5)
        assert controller.running
        controller.settings.model = "changed-after-start"
        with pytest.raises(ValueError, match="already running"):
            controller.start()
        assert observed[0].model != "changed-after-start"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not controller.running
    assert controller.last_state is state
    assert statuses == ["Agent working…", "Finished"]


def test_failed_run_reports_error_and_allows_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Atlas unavailable")

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", fail)
    controller = MenuController(_abba())
    statuses, errors = [], []
    thread = controller.start(on_status=statuses.append, on_error=errors.append)
    thread.join(5)
    assert not thread.is_alive()
    assert not controller.running
    assert controller.last_error == "Atlas unavailable"
    assert errors == ["Atlas unavailable"]
    assert statuses == ["Agent working…", "Run failed"]

    monkeypatch.setattr(
        "langslice.hosts.integrations.abba_linear.run_existing_in_abba",
        lambda *args, **kwargs: SimpleNamespace(submitted=False),
    )
    retry = controller.start(on_status=statuses.append)
    retry.join(5)
    assert not retry.is_alive()
    assert not controller.running
    assert controller.last_error is None
    assert statuses[-1] == "Stopped before submission"


class _Activity:
    def __init__(self):
        self.events = []
        self.statuses = []
        self.disposed = False

    def on_event(self, event):
        assert not self.disposed, "A replaced activity window must not receive new events"
        self.events.append(event)

    def set_status(self, status):
        self.statuses.append(status)

    def dispose(self):
        self.disposed = True


def test_popup_factory_failure_releases_run_guard_and_allows_retry(monkeypatch):
    runs = []

    def run(*args, **kwargs):
        runs.append(True)
        return SimpleNamespace(submitted=True)

    def fail_factory(model):
        raise RuntimeError("Unable to create activity window")

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    controller = MenuController(_abba(), activity_factory=fail_factory)
    with pytest.raises(RuntimeError, match="Unable to create activity window"):
        controller.start()
    assert not controller.running
    assert not runs

    activity = _Activity()
    controller.activity_factory = lambda model: activity
    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive()
    assert not controller.running
    assert runs == [True]
    assert activity.statuses == ["Submitted"]


@pytest.mark.parametrize("submitted", [False, True])
def test_popup_receives_stream_progress_and_completion_while_abba_writes_continue(
    monkeypatch, submitted,
):
    activity = _Activity()
    models = []
    state = SimpleNamespace(submitted=submitted)
    image = {"data": b"agent-visible-png", "label": "Atlas comparison"}
    stream = [
        {"kind": "reasoning", "text": "Comparing the boundaries"},
        {"kind": "tool_call", "name": "inspect", "args": {"section": 2}},
        {"kind": "tool_result", "name": "inspect", "response": {}, "images": [image]},
        {"kind": "complete", "submitted": submitted},
    ]

    def factory(model):
        models.append(model)
        return activity

    def run(abba, spec, *, emit, on_event, on_write, follow_agent):
        assert follow_agent()
        assert controller.activity_window is activity
        emit("Preparing the selected sections")
        for event in stream:
            on_event(event)
        on_write(state)
        return state

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    controller = MenuController(_abba(), activity_factory=factory)
    writes, statuses = [], []
    thread = controller.start(on_write=writes.append, on_status=statuses.append)
    thread.join(5)
    assert not thread.is_alive()
    assert not controller.running
    assert controller.last_error is None
    assert controller.last_state is state
    assert models == [controller.settings.model]
    assert activity.events == [
        {"kind": "progress", "text": "Preparing the selected sections"}, *stream,
    ]
    assert activity.statuses == ["Submitted" if submitted else "Stopped"]
    assert statuses[-1] == ("Finished" if submitted else "Stopped before submission")
    assert writes == [state]


def test_new_run_disposes_previous_popup_before_creating_replacement(monkeypatch):
    old, replacement = _Activity(), _Activity()
    controller = MenuController(_abba(), activity_window=old)

    def factory(model):
        assert old.disposed
        return replacement

    def run(*args, on_event, **kwargs):
        on_event({"kind": "text", "text": "Starting a new run"})
        return SimpleNamespace(submitted=True)

    controller.activity_factory = factory
    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive()
    assert controller.last_error is None
    assert controller.activity_window is replacement
    assert old.disposed and not old.events
    assert replacement.events == [{"kind": "text", "text": "Starting a new run"}]
    assert not replacement.disposed


def test_agent_failure_is_visible_in_popup_and_releases_run_guard(monkeypatch):
    activity = _Activity()

    def fail(*args, **kwargs):
        raise RuntimeError("Atlas unavailable")

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", fail)
    controller = MenuController(_abba(), activity_factory=lambda model: activity)
    errors = []
    thread = controller.start(on_error=errors.append)
    thread.join(5)
    assert not thread.is_alive()
    assert not controller.running
    assert controller.last_error == "Atlas unavailable"
    assert errors == ["Atlas unavailable"]
    assert activity.events == [{"kind": "error", "text": "Atlas unavailable"}]
    assert not activity.disposed


def test_comparison_is_lazy_retained_after_run_and_replaced_on_next_run(monkeypatch):
    created = []
    requested = [False]

    def factory():
        window = _Activity()
        created.append(window)
        return window

    def run(*args, comparison_factory, **kwargs):
        if requested[0]:
            first = comparison_factory()
            assert comparison_factory() is first
        return SimpleNamespace(submitted=True)

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    controller = MenuController(_abba(), comparison_factory=factory)
    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive() and controller.last_error is None
    assert created == [] and controller.comparison_window is None

    requested[0] = True
    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive() and controller.last_error is None
    assert len(created) == 1
    assert controller.comparison_window is created[0]
    assert not created[0].disposed
    assert controller.comparison() is created[0]

    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive() and controller.last_error is None
    assert len(created) == 2
    assert created[0].disposed and not created[1].disposed
    assert controller.comparison_window is created[1]


def test_comparison_factory_failure_does_not_cache_broken_window():
    def fail():
        raise RuntimeError("Could not create native viewer")

    controller = MenuController(_abba(), comparison_factory=fail)
    with pytest.raises(RuntimeError, match="Could not create native viewer"):
        controller.comparison()
    assert controller.comparison_window is None
    replacement = _Activity()
    controller.comparison_factory = lambda: replacement
    assert controller.comparison() is replacement


@pytest.mark.parametrize("open_viewer", [False, True])
@pytest.mark.parametrize("open_log", [False, True])
def test_independent_viewer_and_log_options_preserve_registration(
    monkeypatch, open_viewer, open_log,
):
    old_log, old_viewer = _Activity(), _Activity()
    log, viewer = _Activity(), _Activity()
    created_logs, created_viewers = [], []
    writes = []
    state = SimpleNamespace(submitted=True)

    def log_factory(model):
        assert old_log.disposed and old_viewer.disposed
        created_logs.append(model)
        return log

    def viewer_factory():
        assert old_log.disposed and old_viewer.disposed
        created_viewers.append(True)
        return viewer

    def run(*args, on_event, follow_agent, comparison_factory, on_write, **kwargs):
        # The adapter always receives the factory, but opens its native window
        # only when the independent viewer switch permits the seed/tool event.
        assert callable(comparison_factory)
        assert follow_agent() is open_viewer
        if follow_agent():
            assert comparison_factory() is viewer
        if on_event is not None:
            on_event({"kind": "reasoning", "text": "Inspecting slices 1–4."})
        assert (on_event is not None) is open_log
        on_write(state)
        return state

    monkeypatch.setattr("langslice.hosts.integrations.abba_linear.run_existing_in_abba", run)
    controller = MenuController(
        _abba(),
        settings=MenuSettings(open_agent_viewer=open_viewer, open_agent_log=open_log),
        activity_factory=log_factory, activity_window=old_log,
        comparison_factory=viewer_factory, comparison_window=old_viewer,
    )
    thread = controller.start(on_write=writes.append)
    thread.join(5)
    assert not thread.is_alive() and controller.last_error is None
    assert writes == [state] and controller.last_state is state
    assert len(created_logs) == int(open_log)
    assert len(created_viewers) == int(open_viewer)
    assert old_log.disposed and old_viewer.disposed
    assert controller.activity_window is (log if open_log else None)
    assert controller.comparison_window is (viewer if open_viewer else None)


def test_agent_windows_default_on_without_changing_model_choice():
    settings = MenuSettings()
    assert settings.open_agent_viewer and settings.open_agent_log
    assert settings.model == "openai-oauth/gpt-5.6-luna"


def test_disabled_log_disposes_previous_even_without_factory(monkeypatch):
    old = _Activity()
    controller = MenuController(
        _abba(), MenuSettings(open_agent_log=False), activity_window=old,
    )
    monkeypatch.setattr(
        "langslice.hosts.integrations.abba_linear.run_existing_in_abba",
        lambda *args, **kwargs: SimpleNamespace(submitted=True),
    )
    thread = controller.start()
    thread.join(5)
    assert not thread.is_alive() and controller.last_error is None
    assert old.disposed and controller.activity_window is None
