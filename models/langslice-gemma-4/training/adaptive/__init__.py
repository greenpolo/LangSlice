"""Shared adaptive primitives — schedule, buffer factory, and curriculum.

Phase 5 adds :class:`~adaptive.schedule.AdaptiveSchedule` and
:func:`~adaptive.schedule.make_error_buffer`.  Each pipeline owns its own
buffer; the schedule is stateless and reads the buffer on every call.

Phase 6 lifts the curriculum package into ``adaptive.curriculum``.  Callers
can do ``from adaptive import curriculum`` or import directly via
``from adaptive.curriculum.bins import compute_section_bins``.
"""

from adaptive.schedule import AdaptiveSchedule, ErrorObservation, make_error_buffer
from adaptive import curriculum  # noqa: F401 — re-export the subpackage

__all__ = [
    "AdaptiveSchedule",
    "ErrorObservation",
    "make_error_buffer",
    "curriculum",
]
