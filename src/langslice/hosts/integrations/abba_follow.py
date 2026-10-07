"""Route executed agent tools to the passive ABBA agent viewer."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Tools whose targets the viewer shows: the change and fit tools, look,
#: and marking damage and setting a section's preprocessed channel (both
#: about which sections).
_FOLLOWED = {
    "interactive_transform", "elastix_affine", "ants_syn", "trace_borders",
    "look", "position_sections", "undo", "redo", "mark_damage",
    "set_preprocessed_channel_properties",
}
#: Tools after whose end the viewer refreshes its targets (ABBA has applied
#: the checkpoint's rows by then): everything that changes a placement.
_WRITES = _FOLLOWED - {"look", "set_preprocessed_channel_properties"}


def _on_edt(callback: Callable[[], None]) -> None:
    from jpype import JProxy  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    swing = jimport("javax.swing.SwingUtilities")
    if swing.isEventDispatchThread():
        callback()
    else:
        swing.invokeAndWait(JProxy("java.lang.Runnable", dict(run=callback)))


class AbbaFollower:
    """An observer of serialized ``tool_start`` / ``tool_end`` live events.

    ``target_ids`` must be filenames resolved by the executing toolbox, not
    inferred from model proposals. End events arrive after the connector has
    applied the checkpoint's rows. Every followed event goes to the agent
    viewer built by *comparison_factory* (created on first use); the main
    ABBA view, its selection and its registrations are never changed.
    """

    def __init__(
        self, abba: Any, slice_by_id: Mapping[str, Any],
        *, comparison_factory: Callable[[], Any],
    ) -> None:
        self.abba = abba
        self.slice_by_id = dict(slice_by_id)
        self._comparison_factory = comparison_factory
        self._comparison: Any = None
        self._active: tuple[str, str, list[Any]] | None = None
        self._finished = False

    def on_event(self, event: Mapping[str, Any]) -> None:
        """Best-effort viewer updates; never fail an agent tool."""
        try:
            if self._finished:
                return
            kind, name = str(event.get("kind", "")), str(event.get("name", ""))
            if kind == "seed":
                targets = list(self.slice_by_id.values())
                if targets:
                    _on_edt(lambda: self._show(targets))
                return
            if name not in _FOLLOWED:
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
                    _on_edt(lambda: self._show(targets))
            elif kind == "tool_end" and self._active is not None:
                active_id, active_name, targets = self._active
                if execution == active_id and name == active_name:
                    self._active = None
                    if targets and name in _WRITES:
                        _on_edt(lambda: self._show(targets))
        except Exception:
            logger.warning("ABBA agent viewer could not follow", exc_info=True)

    def _show(self, targets: list[Any]) -> None:
        try:
            if self._comparison is None:
                self._comparison = self._comparison_factory()
            self._comparison.show_targets(targets)
        except Exception:
            logger.warning("ABBA agent viewer could not update", exc_info=True)

    def finish(self) -> None:
        """Stop following; the viewer window stays as it is."""
        self._finished = True
        self._active = None
