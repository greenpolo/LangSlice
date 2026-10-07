"""The atlas side of a fit: which picture of the atlas plane, and which regions count.

Every atlas image is first built on the native plane grid (exactly
``annotation_slice``'s pixels at the placement's position, plane and cutting
angles), with excluded regions blanked to background there, and only then
placed on the section's working grid through the linear placement. Blanking
before placement is what makes exclusion work the same for every image kind:
an excluded region's Nissl or template texture is gone, not just its lines.

The ``nissl`` image is the Nissl template aligned to the Allen CCFv3
(:class:`NisslAtlas`, an Allen mouse atlas only); the reference and the
borders come from the BrainGlobe atlas itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from langslice.core.atlas.render import annotation_slice, family_labels, placed_border_coverage
from langslice.core.deformable.geometry import Placement, warp_affine
from langslice.core.deformable.nissl import NisslAtlas
from langslice.core.deformable.settings import BORDER_IMAGES

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


def resolve_entries(
    atlas: Any, entries: Iterable[str | int],
) -> list[tuple[str, str | None, frozenset[int]]]:
    """``(region, side, ids with descendants)`` per entry; unknown names or sides refused.

    An entry is an acronym or id, optionally with a side (``"CTX:left"``,
    :mod:`langslice.core.atlas.sides`).
    """
    from langslice.core.atlas.sides import split_side

    parsed = [split_side(entry) for entry in entries]
    resolve_structures(atlas, [region for region, _side in parsed])
    return [(region, side, with_descendants(atlas, resolve_structures(atlas, [region])))
            for region, side in parsed]


def whole_region_ids(atlas: Any, entries: Iterable[str | int]) -> frozenset[int]:
    """Ids (with descendants) of the entries that name no side: both hemispheres."""
    return frozenset().union(*(
        ids for _region, side, ids in resolve_entries(atlas, entries) if side is None))


def regions_mask(
    atlas: Any, labels: np.ndarray, entries: Iterable[str | int],
    left: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels of *labels* (native plane) covered by *entries*, sides included.

    *left* is the native pixels on the section's displayed left
    (:func:`langslice.core.atlas.sides.native_left`); needed only when an entry
    names a side.
    """
    from langslice.core.atlas.sides import restrict

    mask = np.zeros(labels.shape, dtype=bool)
    for _region, side, ids in resolve_entries(atlas, entries):
        mask |= restrict(np.isin(labels, list(ids)), side, left)
    return mask


def placement_left(
    atlas: Any, placement: Placement, entries: Iterable[str | int],
) -> np.ndarray | None:
    """The placement's displayed-left native pixels, when an entry names a side."""
    from langslice.core.atlas.sides import has_sides, native_left

    if not has_sides(list(entries)):
        return None
    return native_left(atlas, placement.position_mm, placement.plane, placement.pitch_deg,
                       placement.yaw_deg, placement.atlas_to_section)


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
    atlas: Any, placement: Placement, kind: str, *, nissl: NisslAtlas | None = None,
) -> np.ndarray:
    """The grayscale atlas plane (``template`` or ``nissl``) on the native grid, float32."""
    if kind == "template":
        if placement.pitch_deg or placement.yaw_deg:
            from langslice.core.oblique import sample_oblique_plane

            return sample_oblique_plane(
                atlas, placement.position_mm, placement.plane,
                placement.pitch_deg, placement.yaw_deg, volume="template", order=1,
            )
        from langslice.core.oblique import plane_index_coordinates

        coords = plane_index_coordinates(atlas, placement.position_mm, placement.plane)
        index = np.rint(coords).astype(np.intp)
        return np.asarray(np.asarray(atlas.template)[index[0], index[1], index[2]],
                          dtype=np.float32)
    if kind == "nissl":
        source = nissl or NisslAtlas()
        return source.sample_plane(
            atlas, placement.position_mm, placement.plane,
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
    excluded: np.ndarray | None,
    *,
    softening_px: float,
    nissl: NisslAtlas | None = None,
) -> np.ndarray:
    """The moving image on the working grid, excluded regions blanked first.

    *excluded* is the native-plane mask of excluded pixels (a one-sided
    exclusion covers one half of its region), or None.
    """
    keep = ~excluded if excluded is not None else np.ones(labels.shape, bool)
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
    native = native_intensity(atlas, placement, kind, nissl=nissl)
    native = normalize_intensity(native, (labels > 0) & keep)
    native = np.where(keep, native, 0.0).astype(np.float32)
    return warp_affine(native, atlas_to_working, working_size)
