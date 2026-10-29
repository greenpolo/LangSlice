"""A section's parts in their JSON form, for tests that read them as data."""

from __future__ import annotations

from typing import Any

from langslice.core.stack import codec, landmarks_json
from langslice.core.stack.affine import StoredAffine
from langslice.core.stack.deformation import DeformationRef
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.landmarks import Endpoint, Landmarks, Pair
from langslice.core.stack.trace import BorderTrace

_WRITERS: dict[type, Any] = {
    StoredAffine: codec.affine_to_json,
    DeformationRef: codec.deformation_to_json,
    BorderTrace: codec.trace_to_json,
    Landmarks: landmarks_json.landmarks_to_json,
    Pair: landmarks_json.pair_to_json,
    Endpoint: landmarks_json.endpoint_to_json,
    ImageRef: codec.image_to_json,
    MaskRef: codec.mask_to_json,
    LayerRef: codec.layer_to_json,
}


def json_of(value: Any) -> Any:
    """*value* (a part, a list or mapping of parts, or None) in its JSON form."""
    if value is None:
        return None
    writer = _WRITERS.get(type(value))
    if writer is not None:
        return writer(value)
    if isinstance(value, dict):
        return {key: json_of(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_of(item) for item in value]
    return value
