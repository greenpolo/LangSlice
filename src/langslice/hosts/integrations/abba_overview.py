"""ABBA's native positioning mosaic in an independent read-only BDV panel."""

from __future__ import annotations

import math
from typing import Any

FOCUS_COLORS = ((109, 222, 183), (118, 181, 255), (249, 199, 107), (207, 156, 244))


def native_position_x(z: float, voxel_size: float, tile_width: float, step: int) -> float:
    """ABBA 0.11 getDisplayedCenter, including its native voxel snapping."""
    if voxel_size <= 0 or step <= 0 or not math.isfinite(z):
        raise ValueError("Invalid native positioning calibration")
    return int(z / voxel_size) * tile_width / step + tile_width / 2


def native_handle_style(selected: bool, key: bool) -> tuple[int, tuple[int, int, int, int]]:
    """ABBA SliceGuiState's handle radius and RGBA, without changing selection."""
    if selected:
        return (16 if key else 12), (0, 255, 0, 255 if key else 180)
    return (16 if key else 12), (255, 255, 0, 128 if key else 64)


def native_key_style(selected: bool, key: bool) -> tuple[int, tuple[int, int, int, int]]:
    """ABBA's key-square size follows key status, not selection status."""
    radius, color = native_handle_style(selected, key)
    return radius if key else radius // 2, (255, 0, 255, 200) if key and selected else color


def native_centers(
    view: Any, slices: list[Any], voxel: float, sx: float, sy: float, step: int
) -> list[tuple[float, float]]:
    if int(view.getDisplayMode()) == 0:
        return [
            (float(p.getDoublePosition(0)), float(p.getDoublePosition(1)))
            for p in (view.getDisplayedCenter(sl) for sl in slices)
        ]
    return [
        (native_position_x(float(sl.getSlicingAxisPosition()), voxel, sx, step), sy)
        for sl in slices
    ]


def native_atlas_sources(view: Any, atlas: Any, structural_count: int) -> list[Any]:
    """Preserve the host's channel choices while borrowing its actual mosaic."""
    state = view.getBdvh().getViewerPanel().state()
    displayed = (
        atlas.extendedSlicedSources
        if int(view.getDisplayMode()) == 0
        else atlas.nonExtendedSlicedSources
    )
    return [
        atlas.extendedSlicedSources[i]
        for i in range(structural_count)
        if bool(state.isSourceActive(displayed[i]))
    ]


class NativeOverview:
    """Native spacing, registered images, layout and selection graphics.

    Borrowed atlas mosaic uses the host's existing getStep. Registered images
    retain native size and current displayed centers. Handles are ABBA's own
    CircleGraphicalHandle objects with local suppliers and no editing behaviors.
    No main-view mode, camera, source, atlas step or selection is changed.
    """

    def __init__(self, owner: Any) -> None:
        from jpype import JArray, JProxy  # pyright: ignore[reportMissingImports]

        from .abba_compare import copy_native_source

        self.owner, self.j = owner, owner.jimport
        j = self.j
        self.copy_source = lambda source, display: copy_native_source(j, source, display)
        self.JProxy = JProxy
        self.Integer = j("java.lang.Integer")
        self.IntegerArray = JArray(self.Integer)
        self.container = j("javax.swing.JPanel")(j("java.awt.BorderLayout")())
        self.container.setBackground(j("java.awt.Color")(19, 24, 31))
        self.title = j("javax.swing.JLabel")("ABBA positioning")
        self.title.setForeground(j("java.awt.Color")(192, 204, 217))
        self.title.setBorder(j("javax.swing.BorderFactory").createEmptyBorder(7, 12, 7, 12))
        self.container.add(self.title, "North")
        self.handle = j("bdv.util.BdvHandlePanel")(
            owner.frame,
            j("bdv.util.BdvOptions")
            .options()
            .is2D()
            .preferredSize(1000, 250)
            .numRenderingThreads(2),
        )
        self.stacks: list[Any] = []
        self.locals: list[tuple[int, Any]] = []
        self.sources: list[Any] = []
        self.slices: list[Any] = []
        self.targets: list[Any] = []
        self.centers: list[tuple[float, float]] = []
        self.circles: list[Any] = []
        self.proxies: list[Any] = []
        self.positions: list[float] = []
        self.last_content: Any = None
        self.last_step: Any = None
        self.bounds: tuple[float, float, float, float] | None = None
        self.attached = False
        self.overlay = JProxy(
            "bdv.viewer.OverlayRenderer",
            dict(
                drawOverlays=self._draw,
                setCanvasSize=lambda w, h: None,
            ),
        )
        self.prefs = j("ch.epfl.biop.atlas.aligner.gui.bdv.ABBABdvViewPrefs")
        self.circle_package = "ch.epfl.biop.bdv.gui.graphicalhandle"
        try:
            self.Circle = j(self.circle_package + ".CircleGraphicalHandle")
        except TypeError:
            self.circle_package = "ch.epfl.biop.viewer.bdv.graphicalhandle"
            self.Circle = j(self.circle_package + ".CircleGraphicalHandle")
        self.Square = j(self.circle_package + ".SquareGraphicalHandle")
        self.listener = JProxy(
            self.circle_package + ".GraphicalHandleListener",
            dict(
                disabled=lambda h: None,
                enabled=lambda h: None,
                hover_in=lambda h: None,
                hover_out=lambda h: None,
                created=lambda h: None,
                removed=lambda h: None,
            ),
        )

    def _show(self, source: Any, display: Any = None) -> Any:
        copied = self.copy_source(source, display)
        stack = self.j("bdv.util.BdvFunctions").show(
            copied, self.j("bdv.util.BdvOptions").options().addTo(self.handle)
        )
        stack.setActive(True)
        self.stacks.append(stack)
        return copied

    def _screen(self, index: int) -> tuple[int, int]:
        transform = self.handle.getViewerPanel().state().getViewerTransform()
        x, y = self.centers[index]
        return (
            int(float(transform.get(0, 0)) * x + float(transform.get(0, 3))),
            int(float(transform.get(1, 1)) * y + float(transform.get(1, 3))),
        )

    def _key_screen(self, index: int) -> tuple[int, int]:
        transform = self.handle.getViewerPanel().state().getViewerTransform()
        radius = self._style(index)[0]
        selected = any(self.slices[index] == target for target in self.targets)
        axis_y = float(transform.get(1, 1)) * float(self.owner.abba.mp.sY) / 2
        axis_y += float(transform.get(1, 3)) + (-0.6 if selected else 0.6) * radius
        return self._screen(index)[0], int(axis_y)

    def _style(self, index: int) -> tuple[int, tuple[int, int, int, int]]:
        sl = self.slices[index]
        return native_handle_style(
            any(sl == target for target in self.targets), bool(sl.isKeySlice())
        )

    def _make_circles(self) -> None:
        self.circles.clear()
        self.proxies.clear()
        for index in range(len(self.slices)):
            coords = self.JProxy(
                "java.util.function.Supplier",
                dict(
                    get=lambda i=index: self.IntegerArray(
                        [self.Integer(v) for v in self._screen(i)]
                    )
                ),
            )
            radius = self.JProxy(
                "java.util.function.Supplier",
                dict(get=lambda i=index: self.Integer(self._style(i)[0])),
            )
            color = self.JProxy(
                "java.util.function.Supplier",
                dict(
                    get=lambda i=index: self.IntegerArray(
                        [self.Integer(v) for v in self._style(i)[1]]
                    )
                ),
            )
            self.proxies.extend([coords, radius, color])
            behaviors = self.j("org.scijava.ui.behaviour.util.Behaviours")(
                self.j("org.scijava.ui.behaviour.io.InputTriggerConfig")()
            )
            self.circles.append(
                self.Circle(
                    self.listener,
                    behaviors,
                    self.handle.getTriggerbindings(),
                    f"langslice-read-only-{index}",
                    coords,
                    radius,
                    color,
                )
            )

            key_coords = self.JProxy(
                "java.util.function.Supplier",
                dict(
                    get=lambda i=index: self.IntegerArray(
                        [self.Integer(v) for v in self._key_screen(i)]
                    )
                ),
            )
            key_radius = self.JProxy(
                "java.util.function.Supplier",
                dict(
                    get=lambda i=index: self.Integer(
                        self._style(i)[0]
                        if bool(self.slices[i].isKeySlice())
                        else self._style(i)[0] // 2
                    )
                ),
            )
            key_color = self.JProxy(
                "java.util.function.Supplier",
                dict(
                    get=lambda i=index: self.IntegerArray(
                        [
                            self.Integer(v)
                            for v in native_key_style(
                                any(self.slices[i] == target for target in self.targets),
                                bool(self.slices[i].isKeySlice()),
                            )[1]
                        ]
                    )
                ),
            )
            self.proxies.extend([key_coords, key_radius, key_color])
            self.circles.append(
                self.Square(
                    self.listener,
                    behaviors,
                    self.handle.getTriggerbindings(),
                    f"langslice-read-only-key-{index}",
                    key_coords,
                    key_radius,
                    key_color,
                )
            )

    def update(self, targets: list[Any], focus: list[Any]) -> None:
        from .abba_compare import ensure_camera

        abba = self.owner.abba
        view = abba.get_bdv_view()
        slices = list(abba.mp.getSlices())
        atlas = abba.mp.getReslicedAtlas()
        step = int(atlas.getStep())
        voxel, sx, sy = float(abba.mp.sizePixX), float(abba.mp.sX), float(abba.mp.sY)
        centers = native_centers(view, slices, voxel, sx, sy, step)
        atlas_sources = native_atlas_sources(
            view, atlas, int(abba.mp.getAtlas().getMap().getStructuralImages().size())
        )
        channels = []
        for sl in slices:
            native = list(sl.getRegisteredSources())
            visible = [i for i in range(len(native)) if bool(view.getChannelVisibility(sl, i))]
            channels.append(
                [
                    (native[i], view.getDisplaySettings(sl, i))
                    for i in (visible or [0])
                    if i < len(native)
                ]
            )
        sources = atlas_sources + [source for items in channels for source, _ in items]
        changed = sources != self.sources or slices != self.slices
        self.targets, self.centers = list(targets), centers
        self.positions = [float(sl.getSlicingAxisPosition()) for sl in slices]
        if changed:
            for stack in self.stacks:
                stack.removeFromBdv()
            self.stacks.clear()
            self.locals.clear()
            self.sources, self.slices = sources, slices
            for source in atlas_sources:
                self._show(source)
            for index, items in enumerate(channels):
                for source, display in items:
                    self.locals.append((index, self._show(source, display)))
            if not sources:
                return
            panel = self.handle.getViewerPanel()
            if not self.attached:
                self.container.add(panel, "Center")
                panel.getDisplay().overlays().add(self.overlay)
                self.attached = True
            panel.state().setDisplayMode(self.j("bdv.viewer.DisplayMode").FUSED)
            self._make_circles()
        if not self.attached:
            return
        panel = self.handle.getViewerPanel()
        transform = self.j("net.imglib2.realtransform.AffineTransform3D")()
        matrices = []
        for source in sources:
            source.getSpimSource().getSourceTransform(0, 0, transform)
            matrices.append(tuple(float(transform.get(r, c)) for r in range(3) for c in range(4)))
        content = (
            tuple(self.positions),
            tuple(centers),
            tuple(matrices),
            step,
            float(atlas.getRotateX()),
            float(atlas.getRotateY()),
        )
        if changed or content != self.last_content:
            self.last_content = content
            for index, copied in self.locals:
                transform.identity()
                transform.translate(*centers[index], -self.positions[index])
                copied.getSpimSource().setFixedTransform(transform)
            panel.requestRepaint()
        # Hold a whole-stack camera stable through in-range movement. The host
        # is free to change its native display step; only then reframe the axis.
        if step != self.last_step:
            self.bounds = None
            self.last_step = step
        if centers:
            xs = [x for x, _ in centers]
            ys = [0.0] + [y for _, y in centers]
            current = (min(xs) - sx, max(xs) + sx, min(ys) - sy * 0.6, max(ys) + sy * 0.6)
            if self.bounds is not None:
                current = (
                    min(current[0], self.bounds[0]),
                    max(current[1], self.bounds[1]),
                    min(current[2], self.bounds[2]),
                    max(current[3], self.bounds[3]),
                )
            self.bounds = current
            width, height = int(panel.getWidth()), int(panel.getHeight())
            if width > 0 and height > 0:
                left, right, top, bottom = current
                scale = min(width / (right - left), height / (bottom - top)) / 1.04
                transform.identity()
                transform.scale(scale)
                transform.translate(
                    width / 2 - scale * (left + right) / 2,
                    height / 2 - scale * (top + bottom) / 2,
                    0.0,
                )
                ensure_camera(panel, transform)
        self.title.setText(
            f"ABBA positioning · {len(slices)} slices · native interval {voxel * step * 1000:g} µm"
        )
        panel.getDisplayComponent().repaint()

    def _draw(self, graphics: Any) -> None:
        if not self.attached:
            return
        transform = self.handle.getViewerPanel().state().getViewerTransform()
        sy = float(self.owner.abba.mp.sY)
        axis_y = int(float(transform.get(1, 1)) * sy / 2 + float(transform.get(1, 3)))
        selected = []
        graphics.setStroke(self.prefs.dashed_stroke_slice_handle_to_atlas)
        for index, sl in enumerate(self.slices):
            x, y = self._screen(index)
            active = any(sl == target for target in self.targets)
            graphics.setColor(
                self.prefs.color_slice_handle_selected
                if active
                else self.prefs.color_slice_handle_not_selected
            )
            graphics.drawLine(x, y, x, axis_y)
            if active:
                selected.append(x)
        if selected:
            graphics.setStroke(self.prefs.line_between_selected_slices_stroke)
            graphics.setColor(self.prefs.line_between_selected_slices_color)
            graphics.drawLine(min(selected), axis_y, max(selected), axis_y)
        for circle in self.circles:
            circle.enabledDraw(graphics)

    def dispose(self) -> None:
        for stack in self.stacks:
            stack.removeFromBdv()
        self.stacks.clear()
        if self.attached:
            self.handle.getViewerPanel().getDisplay().overlays().remove(self.overlay)
            self.handle.close()
        self.locals.clear()
        self.sources.clear()
        self.circles.clear()
        self.proxies.clear()
