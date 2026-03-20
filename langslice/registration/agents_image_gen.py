"""Image-generation two-shot registration workflow.

This workflow is exclusively for Gemini image-generation models:

- ``gemini-3-pro-image-preview``
- ``gemini-3.1-flash-image-preview``

These models cannot use tools, structured JSON output, or system
instructions.  Instead, we use a two-pass strategy:

1. **Atlas pass** — the model draws numbered landmark annotations directly
   on the atlas image.
2. **Slice pass** — a second call receives the model-annotated atlas
   alongside the raw histology slice.  The model draws matching numbered
   landmarks on the slice.

Landmark pixel positions are extracted from the generated images via
classical CV: colour thresholding and connected-component analysis.
The model never outputs text — only annotated images.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as _ndimage

import langslice.registration.agents as _agents
from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.registration.types import (
    LandmarkAnnotation,
    RegistrationAnnotationSession,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _species_from_atlas_name(atlas_name: str) -> str:
    """Extract a readable species name from a BrainGlobe atlas identifier."""
    lower = atlas_name.lower()
    for species in ("mouse", "rat", "human", "zebrafish", "fish"):
        if species in lower:
            return species
    return "animal"


def _build_image_gen_config() -> dict[str, object]:
    """Build generation config for image-gen models.

    Image-generation models reject most generation config fields with
    400 INVALID_ARGUMENT.  We pass an empty config.
    """
    return {}


def _extract_generated_image(response: object) -> Image.Image | None:
    """Pull the first generated image from a Gemini response."""
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        mime = getattr(inline, "mime_type", "")
        if isinstance(data, (bytes, bytearray)) and str(mime).startswith("image/"):
            return Image.open(io.BytesIO(data)).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Adaptive colour selection
# ---------------------------------------------------------------------------

_CANDIDATE_COLOURS: list[tuple[str, tuple[int, int, int]]] = [
    ("cyan", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("lime green", (0, 255, 0)),
    ("bright orange", (255, 165, 0)),
    ("bright yellow", (255, 255, 0)),
    ("hot pink", (255, 105, 180)),
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
]


def _choose_marker_colour(
    images: list[Image.Image],
    *,
    threshold: int = 80,
) -> tuple[str, tuple[int, int, int]]:
    """Pick the candidate colour least present in the input images.

    For each candidate, count how many pixels across all *images* have an
    Euclidean RGB distance less than *threshold* from the candidate colour.
    Return the candidate with the fewest nearby pixels.
    """
    # Stack all image pixels into one array for efficiency.
    all_pixels: list[np.ndarray] = []
    for img in images:
        arr = np.asarray(img.convert("RGB"), dtype=np.float64)
        all_pixels.append(arr.reshape(-1, 3))
    if not all_pixels:
        return _CANDIDATE_COLOURS[0]
    pixels = np.concatenate(all_pixels, axis=0)

    best_name, best_rgb = _CANDIDATE_COLOURS[0]
    best_count = pixels.shape[0] + 1  # worse than any real count

    threshold_sq = float(threshold * threshold)
    for name, rgb in _CANDIDATE_COLOURS:
        colour = np.array(rgb, dtype=np.float64)
        dist_sq = np.sum((pixels - colour) ** 2, axis=1)
        count = int(np.sum(dist_sq < threshold_sq))
        if count < best_count:
            best_count = count
            best_name = name
            best_rgb = rgb

    return best_name, best_rgb


# ---------------------------------------------------------------------------
# CV marker extraction
# ---------------------------------------------------------------------------


def _extract_colour_markers(
    image: Image.Image,
    colour_rgb: tuple[int, int, int],
    *,
    tolerance: int = 80,
    min_area: int = 15,
    max_area: int = 8000,
) -> list[tuple[float, float]]:
    """Find marker centroids by detecting regions matching *colour_rgb*.

    Filters pixels by Euclidean RGB distance from the target colour,
    performs connected-component labeling, filters blobs by area, and
    returns centroids sorted in scan order (top-to-bottom, left-to-right).

    Returns a list of ``(x, y)`` pixel positions.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    colour = np.array(colour_rgb, dtype=np.float64)
    dist_sq = np.sum((arr - colour) ** 2, axis=2)
    mask = (dist_sq < float(tolerance * tolerance)).astype(np.int32)

    labeled, n_features = _ndimage.label(mask)

    centroids: list[tuple[float, float]] = []
    for i in range(1, n_features + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if area < min_area or area > max_area:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        centroids.append((cx, cy))

    # Scan order: top-to-bottom, then left-to-right.
    centroids.sort(key=lambda p: (p[1], p[0]))
    return centroids


def _rescale_markers(
    markers: list[tuple[float, float]],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[tuple[float, float]]:
    """Scale marker coordinates from one image size to another.

    Both sizes are (width, height).  Returns markers in the target space.
    """
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        return markers
    sx = tw / fw
    sy = th / fh
    return [(x * sx, y * sy) for x, y in markers]


def _match_markers_nearest(
    atlas_markers: list[tuple[float, float]],
    slice_markers: list[tuple[float, float]],
    *,
    atlas_image_size: tuple[int, int],
    slice_image_size: tuple[int, int],
    max_pairs: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pair atlas and slice markers using nearest-neighbour in normalised space.

    Normalises both sets of markers to [0, 1] based on their respective image
    dimensions, then greedily pairs each atlas marker to the nearest unmatched
    slice marker.  Returns up to *max_pairs* ``(atlas_xy, slice_xy)`` tuples
    in original pixel coordinates.
    """
    aw, ah = atlas_image_size
    sw, sh = slice_image_size
    if aw <= 0 or ah <= 0 or sw <= 0 or sh <= 0:
        return []

    # Normalise to [0, 1].
    a_norm = np.array([(x / aw, y / ah) for x, y in atlas_markers], dtype=np.float64)
    s_norm = np.array([(x / sw, y / sh) for x, y in slice_markers], dtype=np.float64)

    used_slice: set[int] = set()
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for ai in range(len(a_norm)):
        best_si = -1
        best_dist = float("inf")
        for si in range(len(s_norm)):
            if si in used_slice:
                continue
            d = float(np.sum((a_norm[ai] - s_norm[si]) ** 2))
            if d < best_dist:
                best_dist = d
                best_si = si
        if best_si >= 0:
            used_slice.add(best_si)
            pairs.append((atlas_markers[ai], slice_markers[best_si]))
        if len(pairs) >= max_pairs:
            break

    return pairs


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _build_image_gen_atlas_request(
    *,
    atlas_prep: Any,
    species: str,
    target_count: int,
    colour_name: str,
) -> list[dict[str, object]]:
    border_count = max(1, target_count // 2)
    interior_count = max(1, target_count - border_count)
    prompt = (
        f"This is a coronal section of a {species} brain anatomical atlas. "
        "I'd like you to place landmark points as annotations on the atlas. "
        "Prioritize placing landmarks which are evenly distributed, visually "
        "distinct, and would be easily identifiable in a real brain section. "
        f"Please place {border_count} points on the outline/border of the "
        f"brain slice atlas, and {interior_count} points in the interior of "
        "the brain slice atlas. Label each annotation with a number. "
        f"Make both the points and their numbered labels {colour_name} with "
        "white outlines. Please output only the annotated image, no text."
    )
    return [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                _agents._image_to_inline_data(atlas_prep.image),
            ],
        }
    ]


def _build_image_gen_slice_request(
    *,
    annotated_atlas: Image.Image,
    slice_prep: Any,
    species: str,
    colour_name: str,
) -> list[dict[str, object]]:
    prompt = (
        f"This is a {species} brain slice and a corresponding anatomical atlas. "
        "The anatomical atlas is annotated with landmark points. "
        "I'd like you to place corresponding point annotations on the "
        "histological slice in the same relative anatomical positions as "
        "the atlas points. Ensure the numbers of the annotations are preserved. "
        f"Make both the points and their numbered labels {colour_name} with "
        "white outlines. Please output only the annotated histology slice image, no text."
    )
    return [
        {
            "role": "user",
            "parts": [
                _agents._image_to_inline_data(annotated_atlas),
                _agents._image_to_inline_data(slice_prep.image),
                {"text": prompt},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Main two-shot orchestration
# ---------------------------------------------------------------------------


def _estimate_correspondences_image_gen_two_shot(
    client: Any,
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
) -> list[dict[str, object]]:
    """Run the two-shot image-gen landmark workflow.

    The model draws landmark annotations directly on the images.  Marker
    positions are extracted via classical CV (colour thresholding + blob
    detection).  The model never outputs text — only annotated images.
    """
    species = _species_from_atlas_name(atlas_name)
    atlas_image = prepared.atlas_prep.image
    slice_image = prepared.slice_prep.image

    # ------------------------------------------------------------------
    # Choose adaptive marker colour
    # ------------------------------------------------------------------
    colour_name, colour_rgb = _choose_marker_colour([atlas_image, slice_image])
    logger.info("Chosen marker colour: %s %s", colour_name, colour_rgb)

    # ------------------------------------------------------------------
    # Atlas pass — model annotates the atlas image
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: atlas landmark annotation pass (image-gen)...")
    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen atlas request",
            summary=f"Atlas image sent to Gemini for landmark annotation (colour={colour_name})",
            parts=[
                image_part_from_pil(atlas_image, label="Atlas image sent to Gemini"),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 1, "marker_colour": colour_name},
        ),
    )

    atlas_response = _agents._retry_generate(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_atlas_request(
            atlas_prep=prepared.atlas_prep,
            species=species,
            target_count=prepared.target_count,
            colour_name=colour_name,
        ),
        config=_build_image_gen_config(),
        request_label="Registration image-gen atlas pass",
        on_progress=on_progress,
    )

    # Extract generated annotated atlas image.
    annotated_atlas = _extract_generated_image(atlas_response)
    if annotated_atlas is None:
        raise RuntimeError(
            "Image-gen atlas pass did not return an image — "
            "this workflow requires the model to generate annotated images"
        )

    # Extract marker positions from the annotated atlas via CV, then rescale
    # from generated-image space to the original atlas image space.
    raw_atlas_markers = _extract_colour_markers(annotated_atlas, colour_rgb)
    atlas_markers = _rescale_markers(
        raw_atlas_markers,
        from_size=annotated_atlas.size,
        to_size=prepared.atlas_prep.original_size,
    )
    logger.info("Atlas pass: extracted %d markers", len(atlas_markers))

    if not atlas_markers:
        raise RuntimeError(
            "No markers detected in model-generated atlas image. "
            f"Expected {colour_name} annotations."
        )

    atlas_annotations = [
        LandmarkAnnotation(
            image_role="atlas",
            pixel_xy=(mx, my),
            label=str(idx + 1),
            status="confirmed",
        )
        for idx, (mx, my) in enumerate(atlas_markers)
    ]

    if on_annotation_session is not None:
        on_annotation_session(
            RegistrationAnnotationSession(
                workflow="image_gen_two_shot",
                target_count=prepared.target_count,
                atlas_annotations=atlas_annotations,
                slice_annotations=[],
                metadata={
                    "atlas_name": atlas_name,
                    "position_mm": round(position_mm, 3),
                    "pass": 1,
                    "marker_colour": colour_name,
                },
            )
        )
    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen atlas landmarks",
            summary=f"Atlas pass extracted {len(atlas_markers)} markers via CV",
            parts=[
                image_part_from_pil(
                    annotated_atlas, label="Annotated atlas (model-generated)"
                ),
                json_part(
                    [{"label": str(i + 1), "xy": list(m)} for i, m in enumerate(atlas_markers)],
                    label="Atlas marker positions (CV-extracted)",
                    collapsible=True,
                ),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 1, "marker_count": len(atlas_markers)},
        ),
    )

    # ------------------------------------------------------------------
    # Slice pass — model annotates the histology slice
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: slice landmark transfer pass (image-gen)...")
    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen slice request",
            summary="Model-annotated atlas and slice sent to Gemini for transfer",
            parts=[
                image_part_from_pil(annotated_atlas, label="Annotated atlas sent"),
                image_part_from_pil(slice_image, label="Histology slice sent"),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 2},
        ),
    )

    slice_response = _agents._retry_generate(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_slice_request(
            annotated_atlas=annotated_atlas,
            slice_prep=prepared.slice_prep,
            species=species,
            colour_name=colour_name,
        ),
        config=_build_image_gen_config(),
        request_label="Registration image-gen slice pass",
        on_progress=on_progress,
    )

    annotated_slice = _extract_generated_image(slice_response)
    if annotated_slice is None:
        raise RuntimeError(
            "Image-gen slice pass did not return an image — "
            "this workflow requires the model to generate annotated images"
        )

    # Extract marker positions from the annotated slice via CV, then rescale
    # from generated-image space to the original slice image space.
    raw_slice_markers = _extract_colour_markers(annotated_slice, colour_rgb)
    slice_markers = _rescale_markers(
        raw_slice_markers,
        from_size=annotated_slice.size,
        to_size=prepared.slice_prep.original_size,
    )
    logger.info("Slice pass: extracted %d markers", len(slice_markers))

    if not slice_markers:
        raise RuntimeError(
            "No markers detected in model-generated slice image. "
            f"Expected {colour_name} annotations."
        )

    slice_annotations = [
        LandmarkAnnotation(
            image_role="slice",
            pixel_xy=(mx, my),
            label=str(idx + 1),
            status="confirmed",
        )
        for idx, (mx, my) in enumerate(slice_markers)
    ]

    if on_annotation_session is not None:
        on_annotation_session(
            RegistrationAnnotationSession(
                workflow="image_gen_two_shot",
                target_count=prepared.target_count,
                atlas_annotations=atlas_annotations,
                slice_annotations=slice_annotations,
                metadata={
                    "atlas_name": atlas_name,
                    "position_mm": round(position_mm, 3),
                    "pass": 2,
                    "marker_colour": colour_name,
                },
            )
        )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen slice transfer",
            summary=f"Slice pass extracted {len(slice_markers)} markers via CV",
            parts=[
                image_part_from_pil(annotated_slice, label="Model-generated annotated slice"),
                json_part(
                    [{"label": str(i + 1), "xy": list(m)} for i, m in enumerate(slice_markers)],
                    label="Slice marker positions (CV-extracted)",
                    collapsible=True,
                ),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 2, "marker_count": len(slice_markers)},
        ),
    )

    # ------------------------------------------------------------------
    # Pair by nearest-neighbour in normalised coordinate space
    # ------------------------------------------------------------------
    pairs = _match_markers_nearest(
        atlas_markers,
        slice_markers,
        atlas_image_size=prepared.atlas_prep.original_size,
        slice_image_size=prepared.slice_prep.original_size,
        max_pairs=prepared.target_count,
    )

    merged: list[dict[str, object]] = []
    for i, ((ax, ay), (sx, sy)) in enumerate(pairs):
        merged.append(
            {
                "label": str(i + 1),
                "status": "found",
                "atlas_point_2d": [ay, ax],   # [y, x] format
                "slice_point_2d": [sy, sx],   # [y, x] format
            }
        )

    if not merged:
        raise RuntimeError("No marker pairs could be formed from atlas and slice passes")

    return merged
