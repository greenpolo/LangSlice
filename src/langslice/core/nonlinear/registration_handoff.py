"""Host-side bridge from a written linear placement to nonlinear registration.

The sibling methods remain independent. The geometry half (the section image
and the native atlas plane mapped onto it) is the core's,
:func:`langslice.core.handoff.prepare_linear_registration`; it is re-exported
here because the sibling repo SliceBench imports it from this path
(``slicebench/adapters/langslice_geometry.py``). :func:`run_linear_registration`
is the provider-using operation on top of it: it takes the image model as an
argument and never changes the linear checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langslice.core.handoff import LinearRegistrationInput, prepare_linear_registration
from langslice.core.workspace import Workspace

if TYPE_CHECKING:
    from langslice.core.nonlinear.types import RegistrationCandidate
    from langslice.core.state import StackState
    from langslice.providers.registry import ImageModel

__all__ = ["LinearRegistrationInput", "prepare_linear_registration", "run_linear_registration"]


def run_linear_registration(
    state: StackState,
    ctx: Workspace,
    section_id: str,
    *,
    image_model: ImageModel,
    long_edge: int = 2048,
    **options: Any,
) -> RegistrationCandidate:
    """Run nonlinear registration (route "supplied") from this section's placement.

    *image_model* is the model to call, resolved by the caller
    (:func:`langslice.providers.registry.resolve_image_model`). Output and
    generation options may be passed through. Geometry and the model are
    owned here, and attempts to override them are refused. Returned
    coordinates refer to the oriented rendered image, not the acquisition
    TIFF. The linear state is never changed by this operation.
    """
    owned = {
        "image", "atlas_name", "position_mm", "plane", "pitch_deg", "yaw_deg",
        "initial_atlas_to_slice", "initial_alignment_source", "image_axes", "atlas_mirror_lr",
        "provider", "image_model", "image_call",
    }
    conflicts = owned.intersection(options)
    if conflicts:
        raise ValueError("Handoff geometry and model cannot be overridden: "
                         + ", ".join(sorted(conflicts)))
    prepared = prepare_linear_registration(state, ctx, section_id, long_edge=long_edge)
    from langslice.core.nonlinear.image_gen_registration import generate_registration_candidate

    candidate = generate_registration_candidate(
        prepared.image, atlas_name=prepared.atlas_name, position_mm=prepared.position_mm,
        plane=prepared.plane, pitch_deg=prepared.pitch_deg, yaw_deg=prepared.yaw_deg,
        initial_atlas_to_slice=prepared.atlas_to_slice,
        initial_alignment_source="linear_agent", provider=image_model.provider,
        image_model=image_model.model, image_call=image_model.call, **options,
    )
    candidate.metadata["linear_handoff"] = prepared.metadata
    return candidate
