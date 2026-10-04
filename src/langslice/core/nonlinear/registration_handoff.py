"""Host-side bridge from a written linear placement to nonlinear registration.

The geometry (the section image and the native atlas plane mapped onto it) is
the core's, :func:`langslice.core.handoff.prepare_linear_registration`; it is
re-exported here because the sibling repo SliceBench imports it from this path
(``slicebench/adapters/langslice_geometry.py``). On a job, the image-model
step on top of it is the ``trace_borders`` verb.
"""

from __future__ import annotations

from langslice.core.handoff import LinearRegistrationInput, prepare_linear_registration

__all__ = ["LinearRegistrationInput", "prepare_linear_registration"]
