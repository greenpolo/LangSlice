"""Backward-compatibility shim -- renamed to ap_single_slice.py."""

from langslice.estimation.google.common import (  # noqa: F401
    _emit_trace as _emit_trace,
)
from langslice.estimation.google.common import (  # noqa: F401
    _image_to_bytes as _image_to_bytes,
)
