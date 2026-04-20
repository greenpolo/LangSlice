"""Debug-artifact helpers — re-exports during the harness migration.

The canonical implementation still lives in ``langslice.estimation.debug``.
Phase 7 physically moves it here.
"""

from langslice.estimation.debug import write_debug_artifacts  # noqa: F401
