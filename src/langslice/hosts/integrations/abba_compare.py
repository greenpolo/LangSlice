"""Independent, read-only BigDataViewer panels for concurrent ABBA targets.

Panels borrow native image sources, but own their converters, cameras and BDV
handles. They never register/remove global sources or change ABBA display state.
All public methods must be called on the Swing EDT, like ABBA's own view API.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)
PAGE_SIZE = 4


def target_pages(targets: list[Any]) -> list[list[Any]]:
    """Keep execution order, with each native slice appearing only once."""
    unique: list[Any] = []
    for target in targets:
        if not any(target == previous for previous in unique):
            unique.append(target)
    return [unique[start : start + PAGE_SIZE] for start in range(0, len(unique), PAGE_SIZE)]


def camera_parameters(
    width: int, height: int, sx: float, sy: float, z: float
) -> tuple[float, float, float, float] | None:
    """Fit ABBA's calibrated atlas frame at a slice's native sampling depth."""
    if width <= 0 or height <= 0 or sx <= 0 or sy <= 0:
        return None
    if not all(math.isfinite(value) for value in (sx, sy, z)):
        return None
    scale = min(width / sx, height / sy) / 1.08
    return scale, width / 2, height / 2, -scale * z


def ensure_camera(panel: Any, desired: Any) -> None:
    """Repair late BDV auto-initialization without restarting settled rendering."""
    current = panel.state().getViewerTransform()
    if any(
        abs(float(current.get(r, c)) - float(desired.get(r, c))) > 1e-9
        for r in range(3)
        for c in range(4)
    ):
        panel.state().setViewerTransform(desired)


def atlas_channel_indices(keys: list[str]) -> list[int]:
    """Prefer native boundary rendering, excluding atlas coordinate channels."""
    borders = [
        i for i, key in enumerate(keys) if "border" in key.lower() or "outline" in key.lower()
    ]
    if borders:
        return borders
    excluded = {"x", "y", "z", "left right", "leftright", "left-right"}
    return [
        i
        for i, key in enumerate(keys)
        if key.lower() not in excluded and "coordinate" not in key.lower()
    ][:1]


def copy_native_source(j: Any, source: Any, display: Any = None) -> Any:
    # The outer transform and converter belong exclusively to this panel.
    # Even BDV keyboard transforms and brightness controls cannot edit
    # ABBA's registered transform or converter.
    helper = j("sc.fiji.bdvpg.sourceandconverter.SourceAndConverterHelper")
    local_source = j("bdv.tools.transformation.TransformedSource")(source.getSpimSource())
    copied = helper.createSourceAndConverter(local_source)
    settings = j("spimdata.util.Displaysettings")
    if display is None:
        display = settings(-1)
        settings.GetDisplaySettingsFromCurrentConverter(source, display)
    settings.applyDisplaysettings(copied, display)
    return copied


class _Panel:
    def __init__(self, owner: ComparisonWindow) -> None:
        self.owner = owner
        j = owner.jimport
        self.container = j("javax.swing.JPanel")(j("java.awt.BorderLayout")())
        self.label = j("javax.swing.JLabel")("")
        self.label.setBorder(j("javax.swing.BorderFactory").createEmptyBorder(9, 12, 9, 12))
        self.label.setForeground(j("java.awt.Color")(226, 233, 240))
        self.label.setFont(j("java.awt.Font")("SansSerif", 1, 13))
        self.container.setBackground(j("java.awt.Color")(28, 34, 42))
        self.container.add(self.label, "North")
        options = (
            j("bdv.util.BdvOptions").options().is2D().preferredSize(500, 400).numRenderingThreads(2)
        )
        self.handle = j("bdv.util.BdvHandlePanel")(owner.frame, options)
        self.stacks: list[Any] = []
        self.sources: list[Any] = []
        self.target: Any = None
        self.last_geometry: Any = None
        self.last_content: Any = None
        self.attached = False
        self.focus_index = 1

    def _copy_source(self, source: Any, display: Any = None) -> Any:
        return copy_native_source(self.owner.jimport, source, display)

    def update(self, target: Any) -> None:
        j = self.owner.jimport
        abba = self.owner.abba
        view = abba.get_bdv_view()
        native = list(target.getRegisteredSources())
        visible = [i for i in range(len(native)) if bool(view.getChannelVisibility(target, i))]
        if not visible and native:
            visible = [0]
        atlas = abba.mp.getReslicedAtlas()
        keys = [str(key) for key in abba.mp.getAtlas().getMap().getImagesKeys()]
        atlas_sources = list(atlas.nonExtendedSlicedSources)
        indices = atlas_channel_indices(keys)
        selected = [native[i] for i in visible] + [atlas_sources[i] for i in indices]
        changed = len(selected) != len(self.sources) or any(
            a != b for a, b in zip(selected, self.sources, strict=True)
        )
        if target != self.target or changed:
            for stack in self.stacks:
                stack.removeFromBdv()
            self.stacks.clear()
            self.sources = selected
            self.target = target
            for k, source in enumerate(selected):
                display = view.getDisplaySettings(target, visible[k]) if k < len(visible) else None
                copied = self._copy_source(source, display)
                stack = j("bdv.util.BdvFunctions").show(
                    copied, j("bdv.util.BdvOptions").options().addTo(self.handle)
                )
                stack.setActive(True)
                self.stacks.append(stack)
            if not self.attached:
                self.container.add(self.handle.getViewerPanel(), "Center")
                self.attached = True
            self.handle.getViewerPanel().state().setDisplayMode(j("bdv.viewer.DisplayMode").FUSED)
            self.last_geometry = None
        name = str(target.getName())
        z = float(target.getSlicingAxisPosition())
        self.label.setText(f"{self.focus_index} · {name}  ·  ABBA {z:.3f} mm")
        panel = self.handle.getViewerPanel()
        geometry = (
            int(panel.getWidth()),
            int(panel.getHeight()),
            float(abba.mp.sX),
            float(abba.mp.sY),
            z,
        )
        params = camera_parameters(*geometry)
        if params:
            affine = j("net.imglib2.realtransform.AffineTransform3D")()
            affine.scale(params[0])
            affine.translate(*params[1:])
            # The agent panes keep their calibrated frame even if BDV's late
            # source initialization replaces a camera set on an earlier tick.
            ensure_camera(panel, affine)
            self.last_geometry = geometry
        # BDV requestRepaint cancels its current progressive render. Polling
        # unchanged sources must not continually restart it at coarse resolution.
        # Affine fingerprints detect position/pre-transform edits made in place;
        # registration replacements are detected by source identity above.
        transform = j("net.imglib2.realtransform.AffineTransform3D")()
        matrices = []
        for source in selected:
            source.getSpimSource().getSourceTransform(0, 0, transform)
            matrices.append(tuple(float(transform.get(r, c)) for r in range(3) for c in range(4)))
        content = (tuple(matrices), float(atlas.getRotateX()), float(atlas.getRotateY()))
        self._repaint_changed_content(panel, content, sources_changed=changed)

    def _repaint_changed_content(self, panel: Any, content: Any, *, sources_changed: bool) -> None:
        if sources_changed or content != self.last_content:
            self.last_content = content
            panel.requestRepaint()

    def dispose(self) -> None:
        # Local BDV removal/close only; SourceServices.remove would destroy
        # sources still used by ABBA and must never be called here.
        for stack in self.stacks:
            stack.removeFromBdv()
        self.stacks.clear()
        self.sources.clear()
        self.handle.close()


class ComparisonWindow:
    """Show up to four independent native registered atlas overlays per page."""

    def __init__(self, abba: Any, parent: Any = None) -> None:
        from jpype import JProxy  # pyright: ignore[reportMissingImports]
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        self.jimport = jimport
        self.abba = abba
        self.parent = parent
        self.pages: list[list[Any]] = []
        self.page = 0
        self.panels: list[_Panel] = []
        self.closed = False
        self._listeners: list[Any] = []
        self.frame = jimport("javax.swing.JFrame")("LangSlice · Native ABBA comparison")
        self.frame.setDefaultCloseOperation(jimport("javax.swing.WindowConstants").HIDE_ON_CLOSE)
        self.frame.setLayout(jimport("java.awt.BorderLayout")())
        self.grid = jimport("javax.swing.JPanel")(jimport("java.awt.GridLayout")(1, 2, 5, 5))
        self.grid.setBackground(jimport("java.awt.Color")(12, 16, 21))
        from .abba_overview import NativeOverview

        self.overview = NativeOverview(self)
        self.split = jimport("javax.swing.JSplitPane")(
            jimport("javax.swing.JSplitPane").VERTICAL_SPLIT,
            self.overview.container,
            self.grid,
        )
        self.split.setResizeWeight(0.27)
        self.split.setDividerLocation(270)
        self.frame.add(self.split, "Center")
        header = jimport("javax.swing.JPanel")(jimport("java.awt.FlowLayout")(0, 12, 7))
        self.status = jimport("javax.swing.JLabel")(
            "Agent focus · current ABBA registrations · read-only"
        )
        header.add(self.status)
        for label, offset in [("Previous", -1), ("Next", 1)]:
            button = jimport("javax.swing.JButton")(label)
            listener = JProxy(
                "java.awt.event.ActionListener",
                dict(actionPerformed=lambda event, step=offset: self._page(step)),
            )
            self._listeners.append(listener)
            button.addActionListener(listener)
            header.add(button)
        self.frame.add(header, "North")
        self.frame.setMinimumSize(jimport("java.awt.Dimension")(600, 450))
        timer_listener = JProxy(
            "java.awt.event.ActionListener", dict(actionPerformed=lambda event: self._tick())
        )
        self._listeners.append(timer_listener)
        self.timer = jimport("javax.swing.Timer")(250, timer_listener)

    def _page(self, offset: int) -> None:
        if not self.pages:
            return
        self.page = (self.page + offset) % len(self.pages)
        self._rebuild()

    def _rebuild(self) -> None:
        targets = self.pages[self.page] if self.pages else []
        while len(self.panels) > len(targets):
            self.panels.pop().dispose()
        while len(self.panels) < len(targets):
            self.panels.append(_Panel(self))
        self.grid.removeAll()
        rows = 1 if len(targets) <= 2 else 2
        self.grid.setLayout(
            self.jimport("java.awt.GridLayout")(rows, 1 if len(targets) == 1 else 2, 5, 5)
        )
        from .abba_overview import FOCUS_COLORS

        for index, panel in enumerate(self.panels):
            panel.focus_index = index + 1
            panel.label.setForeground(self.jimport("java.awt.Color")(*FOCUS_COLORS[index]))
            self.grid.add(panel.container)
        total = sum(map(len, self.pages))
        self.status.setText(
            f"Agent focus · {total} of {len(self.abba.mp.getSlices())} slices · "
            f"page {self.page + 1}/{len(self.pages)} · read-only"
        )
        self.refresh()
        self.grid.revalidate()
        self.grid.repaint()

    def show_targets(self, targets: list[Any]) -> None:
        if self.closed:
            return
        pages = target_pages(targets)
        if not pages:
            self.hide()
            return
        changed = pages != self.pages
        self.pages = pages
        if changed:
            self.page = 0
            self._rebuild()
        if not self.frame.isVisible():
            if self.parent is not None:
                bounds = self.parent.getBounds()
                self.frame.setBounds(bounds)
            else:
                self.frame.setSize(1200, 900)
                self.frame.setLocationRelativeTo(None)
            self.frame.setVisible(True)
        if changed:
            self.frame.toFront()
        self.timer.start()
        self.refresh()

    def show(self) -> None:
        """Reopen the last comparison, retaining its selected page."""
        if not self.closed and self.pages:
            self.show_targets([target for page in self.pages for target in page])
            self.frame.toFront()

    def refresh(self) -> None:
        if self.closed or not self.pages:
            return
        self.overview.update(
            [target for page in self.pages for target in page], self.pages[self.page]
        )
        for panel, target in zip(self.panels, self.pages[self.page], strict=True):
            panel.update(target)

    def _tick(self) -> None:
        if not self.frame.isVisible():
            self.timer.stop()
            return
        try:
            self.refresh()
        except Exception:
            logger.warning("Native comparison could not refresh", exc_info=True)
            self.timer.stop()

    def hide(self) -> None:
        self.timer.stop()
        self.frame.setVisible(False)

    def dispose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        for panel in self.panels:
            panel.dispose()
        self.panels.clear()
        self.overview.dispose()
        self.pages.clear()
        self.frame.dispose()
