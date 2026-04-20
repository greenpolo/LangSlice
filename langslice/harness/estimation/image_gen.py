"""Image-gen AP estimation — re-exports during the harness migration.

The canonical implementation still lives in
``langslice.estimation.google.ap_image_gen``. Phase 7 of the harness refactor
will physically move the module here and demote the old path to a shim.

Consumers (whole_brain pipeline, eval, slice-bench) should import from this
module going forward so the Phase 7 move is transparent.
"""

from langslice.estimation.google.ap_image_gen import (  # noqa: F401
    estimate_position_image_gen,
)
