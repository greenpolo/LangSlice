"""Follow executed agent tools using ABBA selection, display modes and camera."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Tools that work on sections in place: the viewer shows them registered
#: (review mode). The deformable fit and the image model's trace are
#: in-plane work on the targets, like the affine tools.
_TRANSFORMS = {"adjust_transforms", "fit_affine", "fit_deformable", "trace_borders"}
#: Tools that look at or move sections along the stack (positioning mode);
#: marking damage and changing a section's appearance are about which
#: sections, so the viewer shows them in the stack.
_POSITIONING = {
    "view_slices", "view_placement", "view_stack", "set_positions", "search_position",
    "run_deepslice", "orient_slices", "reorder_slices", "set_cutting_angles",
    "undo", "redo", "mark_damaged", "preprocess",
}
#: Tools after whose end the viewer refreshes its targets (ABBA has applied
#: the checkpoint's rows by then): everything but the pure looks.
_WRITES = _TRANSFORMS | (_POSITIONING - {
    "view_slices", "view_placement", "view_stack", "preprocess",
})


def _on_edt(callback: Callable[[], None]) -> None:
    from jpype import JProxy  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    swing = jimport("javax.swing.SwingUtilities")
    if swing.isEventDispatchThread():
        callback()
    else:
        swing.invokeAndWait(JProxy("java.lang.Runnable", dict(run=callback)))


def _fit_camera(
    abba: Any, view: Any, slices: list[Any], *, review: bool, start_transform: Any = None,
) -> None:
    """Fit native tile bounds, preserving the review slice's sampling plane."""
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    panel = view.getBdvh().getViewerPanel()
    width, height = int(panel.getWidth()), int(panel.getHeight())
    tile_x, tile_y = float(abba.mp.sX), float(abba.mp.sY)
    if width <= 0 or height <= 0 or not all(
        math.isfinite(value) and value > 0 for value in (tile_x, tile_y)
    ):
        return
    centers = [view.getDisplayedCenter(sl) for sl in (slices[:1] if review else slices)]
    xs = [float(point.getDoublePosition(0)) for point in centers]
    ys = [float(point.getDoublePosition(1)) for point in centers]
    zs = [float(point.getDoublePosition(2)) for point in centers]
    if not centers or not all(math.isfinite(value) for value in xs + ys + zs):
        return
    left, right = min(xs) - tile_x / 2, max(xs) + tile_x / 2
    top, bottom = min(ys) - tile_y / 2, max(ys) + tile_y / 2
    scale = min(width / (right - left), height / (bottom - top)) / 1.12
    affine = jimport("net.imglib2.realtransform.AffineTransform3D")()
    affine.scale(scale)
    affine.translate(
        width / 2 - scale * (left + right) / 2,
        height / 2 - scale * (top + bottom) / 2,
        -scale * zs[0],
    )
    if start_transform is None:
        start_transform = panel.state().getViewerTransform()
    animator = jimport("bdv.viewer.animate.SimilarityTransformAnimator")(
        start_transform, affine, 0.0, 0.0, 250,
    )
    # Zero animator centre is deliberate: these are already full viewer
    # transforms. A later real tool replaces this animation without waiting.
    panel.setTransformAnimator(animator)


class AbbaFollower:
    """An observer of serialized ``tool_start`` / ``tool_end`` live events.

    ``target_ids`` must be filenames resolved by the executing toolbox, not
    inferred from model proposals. End events arrive after checkpoint mirroring.
    This observer modifies only selection and visualization, never registration
    or slice positions. Call ``finish`` even after a failed run to restore the
    user's pre-run selection. Disabling follow takes effect at the next event.
    """

    def __init__(
        self, abba: Any, slice_by_id: Mapping[str, Any],
        *, enabled: Callable[[], bool] | None = None,
        comparison_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.abba = abba
        self.slice_by_id = dict(slice_by_id)
        self.enabled = enabled or (lambda: True)
        self._comparison_factory = comparison_factory
        self._comparison: Any = None
        self._selection: list[tuple[Any, bool]] = []
        self._active: tuple[str, str, list[Any]] | None = None
        self._finished = False
        self._selection_changed = False
        _on_edt(self._capture_selection)

    def _capture_selection(self) -> None:
        self._selection = [(sl, bool(sl.isSelected())) for sl in self.abba.mp.getSlices()]

    def on_event(self, event: Mapping[str, Any]) -> None:
        """Best-effort camera/selection updates; never fail an agent tool."""
        try:
            if self._finished:
                return
            if not self.enabled():
                if self._comparison is not None:
                    _on_edt(self._comparison.hide)
                return
            kind, name = str(event.get("kind", "")), str(event.get("name", ""))
            if kind == "seed":
                targets = list(self.slice_by_id.values())
                if targets:
                    _on_edt(lambda: self._show("view_stack", targets))
                return
            if name not in _TRANSFORMS | _POSITIONING:
                return
            execution = str(event.get("execution_id", ""))
            if kind == "tool_start":
                targets = []
                seen: set[str] = set()
                for target in event.get("target_ids", []):
                    if target in self.slice_by_id and target not in seen:
                        targets.append(self.slice_by_id[target])
                        seen.add(target)
                self._active = (execution, name, targets)
                if targets:
                    _on_edt(lambda: self._show(name, targets))
            elif kind == "tool_end" and self._active is not None:
                active_id, active_name, targets = self._active
                if execution == active_id and name == active_name:
                    self._active = None
                    if targets and name in _WRITES:
                        _on_edt(lambda: self._show(name, targets))
        except Exception:
            logger.warning("ABBA follow view could not update", exc_info=True)

    def _show(self, name: str, targets: list[Any]) -> None:
        if self._comparison_factory is not None:
            # A configured agent viewer owns all navigation, including single
            # targets and positioning. Its failure must not redirect the user's
            # main ABBA camera or selection.
            try:
                if self._comparison is None:
                    self._comparison = self._comparison_factory()
                self._comparison.show_targets(targets)
            except Exception:
                logger.warning("ABBA agent viewer could not update", exc_info=True)
            return
        view = self.abba.get_bdv_view()
        start_transform = view.getBdvh().getViewerPanel().state().getViewerTransform()
        review = name in _TRANSFORMS
        self._selection_changed = True
        for sl in self.abba.mp.getSlices():
            if any(sl == target for target in targets):
                sl.select()
            else:
                sl.deSelect()
        view.setSelectedSlicesVisibility(True)
        for sl in targets:
            channels = len(sl.getRegisteredSources())
            if channels and not any(bool(view.getChannelVisibility(sl, index))
                                    for index in range(channels)):
                view.setSliceChannelVisibility(sl, 0, True)
        view.setDisplayMode(1 if review else 0)
        view.setSliceDisplayMode(1 if review else 0)
        if not review:
            # Native ABBA has a cycle, not a public overlap setter. In
            # positioning mode only overlay-on-atlas gives every tile y=0.
            for _ in range(3):
                if all(abs(float(view.getDisplayedCenter(sl).getDoublePosition(1))) < 1e-8
                       for sl in targets):
                    break
                view.toggleOverlap()
        view.navigateSlice(targets[0])
        view.centerBdvViewOn(targets[0])
        _fit_camera(self.abba, view, targets, review=review, start_transform=start_transform)
        view.getBdvh().getViewerPanel().requestRepaint()

    def finish(self) -> None:
        """Restore pre-run selection once, even if follow was switched off."""
        if self._finished:
            return
        self._finished = True
        self._active = None
        if not self._selection_changed:
            return

        def restore() -> None:
            current = list(self.abba.mp.getSlices())
            for sl, selected in self._selection:
                if any(sl == present for present in current):
                    sl.select() if selected else sl.deSelect()
            self.abba.get_bdv_view().getBdvh().getViewerPanel().requestRepaint()

        try:
            _on_edt(restore)
        except Exception:
            logger.warning("ABBA follow view could not restore selection", exc_info=True)
