"""The atlas side of a fit: which picture of the atlas plane, and which regions count.

Every atlas image is first built on the native plane grid (exactly
``annotation_slice``'s pixels at the placement's position, plane and cutting
angles), with excluded regions blanked to background there, and only then
placed on the section's working grid through the linear placement. Blanking
before placement is what makes exclusion work the same for every image kind:
an excluded region's Nissl or template texture is gone, not just its lines.

Host portability: an ABBA host has ABBA's cached Allen volumes, so it may use
the Nissl channel; every other host gets only what BrainGlobe provides — the
atlas reference and borders drawn from its labels (:func:`atlas_images_for_host`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from langslice.atlas.render import annotation_slice, family_labels, placed_border_coverage
from langslice.deformable.abba_atlas import AbbaAtlas
from langslice.deformable.geometry import Placement, warp_affine
from langslice.deformable.settings import BORDER_IMAGES

#: Acronym of the ventricular system: it and all its descendants get the
#: looser area limits (a ventricle collapses or dilates far more than tissue).
VENTRICLE_ACRONYMS = ("VS",)
#: Fallback for atlases without a ``VS`` node: structure names containing any
#: of these (spelled out so periventricular nuclei are not swept in).
VENTRICLE_NAME_KEYWORDS = (
    "ventricle", "central canal", "cerebral aqueduct", "ventricular system",
)
#: Intensity percentile mapped to 1.0 when normalizing grayscale atlas planes.
INTENSITY_PERCENTILE = 99.5


def atlas_images_for_host(host: str, atlas: Any | None = None) -> tuple[str, ...]:
    """The atlas image kinds a host may offer.

    ABBA hosts get ``nissl`` when ABBA's cached Allen volume is present and
    matches the atlas; every host gets the BrainGlobe reference and borders.
    """
    kinds = ["ara", "borders", "borders_merged"]
    if host == "abba":
        found = AbbaAtlas.find()
        if found is not None and (atlas is None or found.compatible(atlas)):
            kinds.insert(1, "nissl")
    return tuple(kinds)


_RECORD_FIELDS = ("id", "acronym", "name", "structure_id_path")


def _structure_records(atlas: Any) -> list[dict[str, Any]]:
    """id/acronym/name/path per structure; never copies a BrainGlobe record
    whole, which would load its lazy ``mesh`` entry from disk or the network."""
    structures = getattr(atlas, "structures", None)
    if structures is None:
        return []
    values = structures.values() if hasattr(structures, "values") else structures
    return [{key: record[key] for key in _RECORD_FIELDS if key in record} for record in values]


def resolve_structures(atlas: Any, names: Iterable[str | int]) -> frozenset[int]:
    """Structure ids for acronyms or ids, refusing unknown names."""
    records = _structure_records(atlas)
    by_acronym = {str(r.get("acronym", "")).lower(): int(r["id"]) for r in records}
    known_ids = {int(r["id"]) for r in records}
    resolved: set[int] = set()
    unknown: list[str] = []
    for name in names:
        if isinstance(name, (int, np.integer)) or str(name).strip().isdigit():
            identifier = int(name)
            if identifier in known_ids:
                resolved.add(identifier)
            else:
                unknown.append(str(name))
            continue
        identifier = by_acronym.get(str(name).strip().lower())
        if identifier is None:
            unknown.append(str(name))
        else:
            resolved.add(identifier)
    if unknown:
        raise ValueError("Unknown atlas structures: " + ", ".join(unknown))
    return frozenset(resolved)


def with_descendants(atlas: Any, ids: Iterable[int]) -> frozenset[int]:
    """*ids* plus every structure whose ``structure_id_path`` passes through one."""
    roots = {int(i) for i in ids}
    if not roots:
        return frozenset()
    result = set(roots)
    for record in _structure_records(atlas):
        path = [int(p) for p in record.get("structure_id_path", [])]
        if roots.intersection(path):
            result.add(int(record["id"]))
    return frozenset(result)


def excluded_ids(atlas: Any, names: Iterable[str | int]) -> frozenset[int]:
    """Agent-chosen exclusions, each with all its descendants."""
    return with_descendants(atlas, resolve_structures(atlas, names))


def ventricle_ids(atlas: Any) -> frozenset[int]:
    """The ventricular system (``VS`` and descendants), or a name-based fallback."""
    records = _structure_records(atlas)
    roots = {int(r["id"]) for r in records if str(r.get("acronym", "")) in VENTRICLE_ACRONYMS}
    if roots:
        return with_descendants(atlas, roots)
    return frozenset(
        int(r["id"]) for r in records
        if any(k in str(r.get("name", "")).lower() for k in VENTRICLE_NAME_KEYWORDS)
    )


def structure_acronyms(atlas: Any, ids: Iterable[int]) -> dict[int, str]:
    """Acronym per id (the id itself when the atlas has no record)."""
    table = {int(r["id"]): str(r.get("acronym", r["id"])) for r in _structure_records(atlas)}
    return {int(i): table.get(int(i), str(int(i))) for i in ids}


def native_labels(atlas: Any, placement: Placement) -> np.ndarray:
    """Leaf annotation ids on the placement's native plane grid."""
    labels = np.asarray(annotation_slice(
        atlas, placement.position_mm, plane=placement.plane,
        pitch_deg=placement.pitch_deg, yaw_deg=placement.yaw_deg,
    ))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Atlas annotation must be a two-dimensional integer label map")
    return labels


def native_intensity(
    atlas: Any, placement: Placement, kind: str, *, abba: AbbaAtlas | None = None,
) -> np.ndarray:
    """The grayscale atlas plane (``ara`` or ``nissl``) on the native grid, float32."""
    if kind == "ara":
        if placement.pitch_deg or placement.yaw_deg:
            from langslice.oblique import sample_oblique_plane

            return sample_oblique_plane(
                atlas, placement.position_mm, placement.plane,
                placement.pitch_deg, placement.yaw_deg, volume="template", order=1,
            )
        from langslice.oblique import plane_index_coordinates

        coords = plane_index_coordinates(atlas, placement.position_mm, placement.plane)
        index = np.rint(coords).astype(np.intp)
        return np.asarray(np.asarray(atlas.template)[index[0], index[1], index[2]],
                          dtype=np.float32)
    if kind == "nissl":
        source = abba or AbbaAtlas.find()
        if source is None:
            raise ValueError("The nissl atlas image needs ABBA's cached Allen atlas")
        return source.sample_plane(
            "NISSL", atlas, placement.position_mm, placement.plane,
            placement.pitch_deg, placement.yaw_deg,
        )
    raise ValueError(f"Not a grayscale atlas image: {kind!r}")


def normalize_intensity(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] by a high percentile inside *region* (zero stays zero)."""
    values = image[region & (image > 0)]
    top = float(np.percentile(values, INTENSITY_PERCENTILE)) if values.size else 0.0
    if top <= 0:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip(image / top, 0.0, 1.0).astype(np.float32)


def soft_lines(lines: np.ndarray, sigma_px: float) -> np.ndarray:
    """A one-pixel line mask as a Gaussian ridge, exp(-d^2 / 2 sigma^2) of distance.

    Both the model's lines and the atlas borders go through this same profile,
    so a fit compares like with like; *sigma_px* sets the capture range.
    """
    mask = np.asarray(lines, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    distance = ndi.distance_transform_edt(~mask)
    return np.exp(-0.5 * (np.asarray(distance) / max(sigma_px, 1e-6)) ** 2).astype(np.float32)


#: Supersampling for the placed border render on the working grid.
BORDER_SUPERSAMPLE = 2
#: Border coverage at or above this counts as a line pixel.
BORDER_COVERAGE_THRESHOLD = 0.25


def placed_atlas_image(
    kind: str,
    labels: np.ndarray,
    atlas: Any,
    placement: Placement,
    atlas_to_working: np.ndarray,
    working_size: tuple[int, int],
    excluded: frozenset[int],
    *,
    softening_px: float,
    abba: AbbaAtlas | None = None,
) -> np.ndarray:
    """The moving image on the working grid, excluded regions blanked first."""
    keep = ~np.isin(labels, list(excluded)) if excluded else np.ones(labels.shape, bool)
    if kind in BORDER_IMAGES:
        source = family_labels(labels, atlas) if kind == "borders_merged" else labels
        # Excluded regions join the background, so their internal lines vanish
        # and their edge with kept tissue becomes an outline.
        source = np.where(keep, source, 0)
        coverage = placed_border_coverage(
            source, atlas_to_working, working_size, width_px=1.0,
            supersample=BORDER_SUPERSAMPLE,
        )
        return soft_lines(coverage >= BORDER_COVERAGE_THRESHOLD, softening_px)
    native = native_intensity(atlas, placement, kind, abba=abba)
    native = normalize_intensity(native, (labels > 0) & keep)
    native = np.where(keep, native, 0.0).astype(np.float32)
    return warp_affine(native, atlas_to_working, working_size)
