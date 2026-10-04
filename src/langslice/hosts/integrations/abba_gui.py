"""Session-local LangSlice controls in ABBA's BigDataViewer menu bar."""
from __future__ import annotations

import logging
import math
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any

from langslice.core.spec import JobSpec, PositionSpec, ReorderSpec, TransformSpec

logger = logging.getLogger(__name__)


@dataclass
class MenuSettings:
    """Editable user choices; unset protocol fields follow the current stack."""

    model: str = "openai-oauth/gpt-5.6-luna"
    reasoning: str = "medium"
    interval_um: int | None = None
    thickness_um: int | None = None
    reorder: bool = True
    position: bool = True
    transform: bool = True
    flip: bool = True
    hemisphere_cue: str = ""
    strict_interval: bool = False
    interactive: bool = True
    automatic: bool = True
    facts: str = ""
    open_agent_viewer: bool = True
    open_agent_log: bool = True
    # Kept for callers constructing older settings; window options now control display.
    follow_agent: bool = True

    def to_spec(self, abba: Any) -> JobSpec:
        interval, thickness = protocol_defaults(abba)
        tasks = [name for name in ("reorder", "position", "transform") if getattr(self, name)]
        if not tasks:
            raise ValueError("Select at least one task in the LangSlice menu.")
        if self.transform and not (self.interactive or self.automatic):
            raise ValueError("Enable interactive adjustment or automatic fitting under Transforms.")
        if not self.model.strip():
            raise ValueError("Choose a model before starting.")
        return JobSpec(
            image_folder="", model=self.model.strip(), reasoning=self.reasoning,
            tasks=tasks, resume=False, debrief=False,
            reorder=ReorderSpec(flip=self.flip, hemisphere_cue=self.hemisphere_cue),
            position=PositionSpec(
                interval_um=self.interval_um if self.interval_um is not None else interval,
                thickness_um=self.thickness_um if self.thickness_um is not None else thickness,
                strict_interval=self.strict_interval,
            ),
            transform=TransformSpec(
                interactive=self.interactive, automatic=self.automatic,
            ),
            facts=[line for line in self.facts.splitlines() if line.strip()],
        )


def protocol_defaults(abba: Any) -> tuple[int, int]:
    """Read current slice spacing and thickness in µm, falling back only if absent.

    ABBA retains imported positions rather than an independent interval field.
    Median positive spacing tolerates missing sections. Thickness is ABBA's
    current thickness (which may have been matched to neighbours by the user).
    """
    slices = list(abba.mp.getSlices())
    positions = sorted(float(s.getSlicingAxisPosition()) for s in slices)
    positions = [p for p in positions if math.isfinite(p)]
    gaps = [b - a for a, b in zip(positions, positions[1:], strict=False) if b - a > 1e-6]
    thicknesses = [float(s.getThicknessInMm()) for s in slices]
    thicknesses = [t for t in thicknesses if math.isfinite(t) and t > 0]
    interval = round(statistics.median(gaps) * 1000) if gaps else 200
    thickness = round(statistics.median(thicknesses) * 1000) if thicknesses else 50
    return max(1, interval), max(1, thickness)


@dataclass
class MenuController:
    abba: Any
    settings: MenuSettings = field(default_factory=MenuSettings)
    running: bool = False
    last_state: Any = None
    last_error: str | None = None
    activity_factory: Any = None
    activity_window: Any = None
    comparison_factory: Any = None
    comparison_window: Any = None
    _references: list[Any] = field(default_factory=list)
    _lock: Any = field(default_factory=threading.Lock)

    def comparison(self) -> Any:
        """Lazily create a spectator; the session owns it across run completion."""
        if self.comparison_window is None:
            if self.comparison_factory is None:
                raise RuntimeError("No ABBA comparison display is configured")
            self.comparison_window = self.comparison_factory()
        return self.comparison_window

    def start(
        self, *, on_status: Any = None, on_error: Any = None, on_write: Any = None
    ) -> threading.Thread:
        """Snapshot settings on the UI thread; do all registration work off it."""
        spec = self.settings.to_spec(self.abba)
        with self._lock:
            if self.running:
                raise ValueError("A LangSlice agent is already running in this ABBA session.")
            self.running = True
            self.last_error = None

        try:
            if self.comparison_window is not None:
                self.comparison_window.dispose()
                self.comparison_window = None
            if self.activity_window is not None:
                self.activity_window.dispose()
                self.activity_window = None
            if self.settings.open_agent_log and self.activity_factory is not None:
                self.activity_window = self.activity_factory(spec.model or "")
        except Exception:
            with self._lock:
                self.running = False
            raise

        activity = self.activity_window

        def emit(message: str) -> None:
            print(message)
            if activity is not None:
                activity.on_event({"kind": "progress", "text": message})

        def worker() -> None:
            try:
                from langslice.hosts.integrations.abba_linear import run_existing_in_abba

                if on_status:
                    on_status("Agent working…")
                display_options = {}
                if self.comparison_factory is not None:
                    display_options["comparison_factory"] = self.comparison
                self.last_state = run_existing_in_abba(
                    self.abba, spec, emit=emit, on_write=on_write,
                    on_event=activity.on_event if activity is not None else None,
                    follow_agent=lambda: self.settings.open_agent_viewer,
                    **display_options,
                )
                if activity is not None:
                    activity.set_status("Submitted" if self.last_state.submitted else "Stopped")
                if on_status:
                    message = (
                        "Finished" if self.last_state.submitted else "Stopped before submission"
                    )
                    on_status(message)
            except Exception as exc:
                logger.exception("ABBA LangSlice run failed")
                self.last_error = str(exc)
                if activity is not None:
                    activity.on_event({"kind": "error", "text": str(exc)})
                if on_status:
                    on_status("Run failed")
                if on_error:
                    on_error(str(exc))
            finally:
                with self._lock:
                    self.running = False

        thread = threading.Thread(target=worker, name="langslice-abba-agent", daemon=True)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.running = False
            raise
        return thread


def install_menu(abba: Any, *, settings: MenuSettings | None = None) -> MenuController:
    """Install once after ``show_bdv_ui``; all Swing access runs on the EDT."""
    existing = getattr(abba, "_langslice_menu", None)
    if existing is not None:
        return existing
    from jpype import JProxy  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    Swing = jimport("javax.swing.SwingUtilities")
    JMenu = jimport("javax.swing.JMenu")
    Item = jimport("javax.swing.JMenuItem")
    Check = jimport("javax.swing.JCheckBoxMenuItem")
    Dialog = jimport("javax.swing.JOptionPane")
    controller = MenuController(abba, settings or MenuSettings())

    def edt(fn: Any, *, wait: bool = False) -> None:
        if Swing.isEventDispatchThread():
            fn()
        else:
            proxy = JProxy("java.lang.Runnable", dict(run=fn))
            if wait:
                Swing.invokeAndWait(proxy)
            else:
                Swing.invokeLater(proxy)

    def build() -> None:
        # abba-python constructs the BDV view directly, bypassing the Java
        # launcher that initializes the theme. Null strokes otherwise crash
        # the overlay painter as soon as imported slices are selected.
        prefs = jimport("ch.epfl.biop.atlas.aligner.gui.bdv.ABBABdvViewPrefs")
        if prefs.line_between_selected_slices_stroke is None:
            theme = jimport("ch.epfl.biop.atlas.aligner.gui.bdv.ABBATheme")
            theme.setTheme(theme.createLightTheme())
        frame = Swing.getWindowAncestor(abba.get_bdv_view().getBdvh().getViewerPanel())
        from langslice.hosts.integrations.abba_chat import create_activity_window
        from langslice.hosts.integrations.abba_compare import ComparisonWindow

        controller.activity_factory = lambda model: create_activity_window(
            model=model, parent=frame,
        )
        controller.comparison_factory = lambda: ComparisonWindow(abba, parent=frame)
        menu = JMenu("LangSlice")
        status = Item("Ready")
        status.setEnabled(False)
        run_item = Item("Run agent on selected slices")
        configurable: list[Any] = []

        def error(message: str) -> None:
            edt(lambda: Dialog.showMessageDialog(frame, message, "LangSlice", Dialog.ERROR_MESSAGE))

        def action(item: Any, fn: Any) -> Any:
            def clicked(event: Any) -> None:
                try:
                    fn()
                except Exception as exc:
                    error(str(exc))
            listener = JProxy("java.awt.event.ActionListener", dict(actionPerformed=clicked))
            controller._references.append(listener)
            item.addActionListener(listener)
            return item

        def text_setting(label: str, attr: str, *, numeric: bool = False) -> None:
            item = Item(label)
            def edit() -> None:
                current = getattr(controller.settings, attr)
                if numeric and current is None:
                    current = protocol_defaults(abba)[0 if attr == "interval_um" else 1]
                value = Dialog.showInputDialog(frame, label, str(current))
                if value is None:
                    return
                parsed: Any = str(value).strip()
                if numeric:
                    number = float(parsed)
                    if not math.isfinite(number) or number <= 0 or number != int(number):
                        raise ValueError("Enter a positive whole number in micrometres.")
                    parsed = int(number)
                setattr(controller.settings, attr, parsed)
                refresh()
            action(item, edit)
            menu.add(item)
            configurable.append(item)
            setting_items[attr] = item

        def checkbox(parent: Any, label: str, attr: str) -> Any:
            item = Check(label, bool(getattr(controller.settings, attr)))
            action(item, lambda: setattr(controller.settings, attr, bool(item.isSelected())))
            parent.add(item)
            configurable.append(item)
            return item

        setting_items: dict[str, Any] = {}
        text_setting("Model", "model")
        text_setting("Reasoning effort", "reasoning")
        text_setting("Slice interval (µm)", "interval_um", numeric=True)
        text_setting("Slice thickness (µm)", "thickness_um", numeric=True)
        reset = Item("Read interval and thickness from ABBA")
        def reset_protocol() -> None:
            controller.settings.interval_um = controller.settings.thickness_um = None
            refresh()
        menu.add(action(reset, reset_protocol))
        configurable.append(reset)
        menu.addSeparator()

        order = JMenu("Ordering")
        checkbox(order, "Enable ordering", "reorder")
        checkbox(order, "Allow hemisphere flips", "flip")
        cue = Item("Describe hemisphere cue…")
        def edit_cue() -> None:
            value = Dialog.showInputDialog(frame, "Hemisphere cue (notch, injection, etc.)",
                                           controller.settings.hemisphere_cue)
            if value is not None:
                controller.settings.hemisphere_cue = str(value)
        order.add(action(cue, edit_cue))
        configurable.append(cue)
        menu.add(order)
        position = JMenu("Position")
        checkbox(position, "Enable positioning", "position")
        checkbox(position, "Enforce strict slice interval", "strict_interval")
        menu.add(position)
        transform = JMenu("Transforms")
        checkbox(transform, "Enable transforms", "transform")
        checkbox(transform, "Interactive adjustment by agent", "interactive")
        checkbox(transform, "Automatic affine fitting", "automatic")
        menu.add(transform)
        menu.addSeparator()
        checkbox(menu, "Open agent viewer in ABBA", "open_agent_viewer")
        checkbox(menu, "Open agent log", "open_agent_log")

        def update_status(message: str) -> None:
            def update() -> None:
                status.setText(message)
                menu.setText("LangSlice · " + message)
                busy = message == "Agent working…"
                run_item.setEnabled(not busy)
                for item in configurable:
                    item.setEnabled(not busy)
            edt(update)

        def start() -> None:
            controller.start(on_status=update_status, on_error=error)

        action(run_item, start)
        menu.add(run_item)
        activity_item = Item("Show agent log")
        def show_activity() -> None:
            if controller.activity_window is not None:
                controller.activity_window.show()
        menu.add(action(activity_item, show_activity))
        comparison_item = Item("Show agent viewer")
        def show_comparison() -> None:
            if controller.comparison_window is not None:
                controller.comparison_window.show()
        menu.add(action(comparison_item, show_comparison))
        menu.add(status)
        note = Item("No selection: runs on all slices")
        note.setEnabled(False)
        menu.add(note)

        def refresh() -> None:
            interval, thickness = protocol_defaults(abba)
            for attr, item in setting_items.items():
                value = getattr(controller.settings, attr)
                if attr in {"interval_um", "thickness_um"}:
                    auto = value is None
                    default = interval if attr == "interval_um" else thickness
                    value = value if value is not None else default
                    title = "Slice interval" if attr == "interval_um" else "Slice thickness"
                    item.setText(f"{title}: {value} µm" + (" (ABBA)" if auto else ""))
                else:
                    title = "Model" if attr == "model" else "Reasoning effort"
                    item.setText(f"{title}: {value}")
        refresh()
        listener = JProxy("javax.swing.event.MenuListener", dict(
            menuSelected=lambda event: refresh(), menuDeselected=lambda event: None,
            menuCanceled=lambda event: None,
        ))
        controller._references.extend([listener, menu, frame])
        menu.addMenuListener(listener)
        frame.getJMenuBar().add(menu)
        frame.getJMenuBar().revalidate()
        frame.getJMenuBar().repaint()

    edt(build, wait=True)
    abba._langslice_menu = controller
    return controller
