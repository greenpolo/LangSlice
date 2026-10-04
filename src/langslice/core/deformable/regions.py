"""Section-side label maps: the image model's lines as named regions.

Label-map mode registers region indicators rather than pictures, as the ANTsX
mouse pipeline does (Tustison et al. 2025, Nat Commun 16:11548: label-guided
deformable alignment beat intensity-only alignment on every modality they
report). On the image-model route the section's labels come from the model's
extracted lines: each area the lines enclose is named after the placed atlas
region it overlaps most. Names are overlap hypotheses inherited from the
linear placement, not anatomical recognition. Adapted from the experimental
partition on the ``registration-design`` branch (commit cdcadb0).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

#: Gaps in the model's lines shorter than about twice this are closed.
LINE_GAP_CLOSING_MM = 0.03
#: An enclosed area below this size that sits wholly inside one placed region
#: (purity above :data:`SMALL_LOOP_PURITY`) and is a small share of it
#: (:data:`SMALL_LOOP_SHARE`) is a loop around a bubble or speck, not a region.
SMALL_LOOP_MM2 = 0.04
SMALL_LOOP_PURITY = 0.98
SMALL_LOOP_SHARE = 0.05
#: An area overlapping its best placed region less than this is left unnamed.
MIN_NAME_PURITY = 0.1
#: Minimum normalized overlap for the joint name assignment to rename an area.
MIN_JOINT_SCORE = 0.02


def named_regions(
    lines: np.ndarray, placed: np.ndarray, tissue: np.ndarray, mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """(labels, unnamed, report) for the areas the model's lines enclose.

    *lines* is the model's one-pixel line mask, *placed* the linearly placed
    atlas labels (the merged set the model was shown) and *tissue* the filled
    tissue mask, all on one grid. Pixels outside tissue are background (0).
    Each enclosed tissue area is named by its largest placed-region overlap,
    then names are resolved jointly so an inner area that spilled into its
    surroundings keeps its own name instead of the surrounding one (otherwise
    the very nested boundary the fit should follow disappears). Unnamed areas
    come back in *unnamed* so the fit can leave them out of its masks.
    """
    lines = np.asarray(lines, dtype=bool)
    if lines.shape != placed.shape or tissue.shape != placed.shape:
        raise ValueError("Lines, placed labels and tissue must share one grid")
    radius = max(1, round(LINE_GAP_CLOSING_MM / mm_per_px))
    wall = np.asarray(ndi.binary_closing(lines, iterations=radius), dtype=bool) | lines
    # A one-pixel barrier joins diagonal segments without moving area interiors.
    wall = np.asarray(ndi.binary_dilation(wall), dtype=bool)
    cells, count = cast(tuple[np.ndarray, int], ndi.label(~wall & tissue))
    lut = np.zeros(count + 1, dtype=placed.dtype)
    named = np.zeros(count + 1, dtype=bool)
    region_sizes = dict(zip(*np.unique(placed, return_counts=True), strict=True))
    report: list[dict[str, Any]] = []
    area_px = np.bincount(cells.ravel(), minlength=count + 1)
    for cell, mask in _cell_masks(cells, count):
        ids, sizes = np.unique(placed[mask], return_counts=True)
        nonzero = ids != 0
        winner = int(ids[nonzero][np.argmax(sizes[nonzero])]) if nonzero.any() else 0
        purity = (float(sizes[nonzero].max()) / float(area_px[cell])) if winner else 0.0
        reasons = []
        if winner == 0 or purity < MIN_NAME_PURITY:
            reasons.append("no_placed_overlap")
        elif (area_px[cell] * mm_per_px ** 2 < SMALL_LOOP_MM2 and purity > SMALL_LOOP_PURITY
              and area_px[cell] / region_sizes[winner] < SMALL_LOOP_SHARE):
            reasons.append("small_loop_inside_one_region")
        named[cell] = not reasons
        lut[cell] = winner if not reasons else 0
        report.append({
            "area": cell, "region_id": winner if not reasons else 0,
            "area_mm2": float(area_px[cell] * mm_per_px ** 2),
            "overlap_fraction": purity, "dropped": reasons,
        })
    _joint_names(cells, count, placed, area_px, named, lut, report)
    # Fill the thin line barrier from the nearest area; unnamed stays unnamed.
    nearest = np.asarray(ndi.distance_transform_edt(
        cells == 0, return_distances=False, return_indices=True), dtype=int)
    filled = cells[tuple(nearest)]
    labels = np.where(tissue, lut[filled], 0).astype(placed.dtype)
    unnamed = tissue & ~named[filled]
    return labels, unnamed, report


def _cell_masks(cells: np.ndarray, count: int):
    slices = ndi.find_objects(cells)
    for index, box in enumerate(slices, start=1):
        if box is None:
            continue
        mask = np.zeros(cells.shape, dtype=bool)
        mask[box] = cells[box] == index
        yield index, mask


def _joint_names(
    cells: np.ndarray, count: int, placed: np.ndarray, area_px: np.ndarray,
    named: np.ndarray, lut: np.ndarray, report: list[dict[str, Any]],
) -> None:
    """Assign names one-to-one by overlap normalized by both sizes (Hungarian)."""
    components = []
    for uid in np.unique(placed):
        if not uid:
            continue
        parts, number = cast(tuple[np.ndarray, int], ndi.label(placed == uid))
        for part in range(1, number + 1):
            mask = parts == part
            if np.count_nonzero(mask) >= 32:
                components.append((int(uid), mask))
    candidates = [r for r in report if named[r["area"]] and r["region_id"]]
    if not candidates or not components:
        return
    score = np.zeros((len(candidates), len(components)))
    for j, (_, mask) in enumerate(components):
        overlap = np.bincount(cells[mask], minlength=count + 1)
        size = np.count_nonzero(mask)
        for i, record in enumerate(candidates):
            score[i, j] = overlap[record["area"]] / np.sqrt(size * area_px[record["area"]])
    rows, columns = linear_sum_assignment(-score)
    for i, j in zip(rows, columns, strict=True):
        if score[i, j] < MIN_JOINT_SCORE:
            continue
        record = candidates[i]
        record["majority_region_id"] = record["region_id"]
        record["region_id"] = components[j][0]
        lut[record["area"]] = record["region_id"]
