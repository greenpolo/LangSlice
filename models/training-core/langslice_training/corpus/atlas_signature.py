from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from math import pi
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import ndimage

from langslice_harness.atlas.core import (
    _lookup_structure_record,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

_UNAVAILABLE = "Visible clues: atlas signature unavailable."
_DEFAULT_OFFSETS_MM = (-0.50, -0.25, 0.25, 0.50)
_DEFAULT_TOP_K = 16
# Landmark priority tables live in the Gemma-4 data dir. This module was
# relocated here from models/synthdata/synthdata/; anchor on the repo root
# (parents[4] = repo root from langslice_training/corpus/) so the path holds.
_GEMMA_DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "models"
    / "langslice-gemma-4"
    / "data"
)


@dataclass(frozen=True)
class RegionFeatures:
    area_fraction: float
    centroid_x: float
    centroid_y: float
    major_axis_len: float
    minor_axis_len: float
    elongation: float
    orientation_deg: float
    compactness: float
    component_count: int
    largest_component_fraction: float


@dataclass(frozen=True)
class RegionScore:
    region_id: int
    name: str
    score: float
    local_change_magnitude: float
    centroid_shift_x: float
    centroid_shift_y: float
    area_trend: float


def compose_visible_clues(
    atlas_name: str,
    plane: str,
    position_mm: float,
    *,
    comparison_positions: list[float] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> str:
    try:
        top_k = max(1, int(top_k))
        atlas = load_atlas(atlas_name)
        base_slice = _annotation_slice(atlas, plane, position_mm)
        base_features = _extract_slice_features(base_slice)
        if not base_features:
            return _UNAVAILABLE

        if comparison_positions is None:
            positions = [position_mm + delta for delta in _DEFAULT_OFFSETS_MM]
        else:
            positions = list(comparison_positions)

        comparisons: dict[float, dict[int, RegionFeatures]] = {}
        for pos in positions:
            try:
                comp_slice = _annotation_slice(atlas, plane, float(pos))
            except Exception:
                continue
            comparisons[float(pos) - float(position_mm)] = _extract_slice_features(comp_slice)

        names = _region_name_map(atlas, base_features.keys())
        ranked = _score_regions(
            base_features,
            comparisons,
            names,
            atlas=atlas,
            atlas_name=atlas_name,
            plane=plane,
            top_k=top_k,
        )
        if not ranked:
            return _UNAVAILABLE

        clues = [
            _render_clue(rank, base_features.get(rank.region_id))
            for rank in ranked
        ]
        clues = [c for c in clues if c]
        if not clues:
            return _UNAVAILABLE
        return "Visible clues: " + "; ".join(clues) + "."
    except Exception:
        return _UNAVAILABLE


def _annotation_slice(atlas: Any, plane: str, position_mm: float) -> np.ndarray:
    plane_typed = plane  # runtime validation happens inside slice_axis_index
    idx = position_mm_to_index(atlas, position_mm, plane=plane_typed)  # type: ignore[arg-type]
    axis = slice_axis_index(atlas_space_context(atlas), plane_typed)  # type: ignore[arg-type]
    arr = np.take(np.asarray(atlas.annotation), idx, axis=axis)
    return np.asarray(arr, dtype=np.int32)


def _extract_slice_features(annotation_slice: np.ndarray) -> dict[int, RegionFeatures]:
    if annotation_slice.ndim != 2:
        raise ValueError(f"Expected 2D annotation slice, got shape={annotation_slice.shape}")

    h, w = annotation_slice.shape
    total_px = float(h * w)
    features: dict[int, RegionFeatures] = {}
    ids, counts = np.unique(annotation_slice, return_counts=True)

    for sid_raw, count_raw in zip(ids, counts, strict=True):
        sid = int(sid_raw)
        if sid == 0:
            continue
        count = int(count_raw)
        if count <= 0:
            continue

        mask = annotation_slice == sid
        ys, xs = np.nonzero(mask)

        centroid_x = float(np.mean(xs) / max(w - 1, 1))
        centroid_y = float(np.mean(ys) / max(h - 1, 1))

        major_axis_len = 0.0
        minor_axis_len = 0.0
        orientation_deg = 0.0
        if count >= 2:
            points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
            cov = np.cov(points.T, bias=True)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            major_eval = float(max(evals[order[0]], 0.0))
            minor_eval = float(max(evals[order[1]], 0.0))
            major_axis_len = float(np.sqrt(major_eval))
            minor_axis_len = float(np.sqrt(minor_eval))
            vec = evecs[:, order[0]]
            orientation_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
            while orientation_deg <= -90.0:
                orientation_deg += 180.0
            while orientation_deg > 90.0:
                orientation_deg -= 180.0

        elongation = float((major_axis_len + 1e-9) / (minor_axis_len + 1e-9))

        eroded = np.asarray(
            ndimage.binary_erosion(
                mask,
                structure=np.ones((3, 3), dtype=bool),
                border_value=0,
            ),
            dtype=bool,
        )
        perimeter_px = int(np.count_nonzero(mask & np.logical_not(eroded)))
        if perimeter_px > 0:
            compactness = float((4.0 * pi * count) / float(perimeter_px * perimeter_px))
            compactness = max(0.0, min(1.0, compactness))
        else:
            compactness = 0.0

        label_result = cast(
            tuple[np.ndarray, int],
            ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8)),
        )
        labeled = np.asarray(label_result[0])
        n_components = int(label_result[1])
        largest_fraction = 1.0
        if n_components > 0:
            comp_sizes = np.bincount(labeled.ravel())
            if comp_sizes.size > 1:
                largest = float(np.max(comp_sizes[1:]))
                largest_fraction = largest / float(count)

        features[sid] = RegionFeatures(
            area_fraction=float(count / total_px),
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            major_axis_len=major_axis_len,
            minor_axis_len=minor_axis_len,
            elongation=elongation,
            orientation_deg=orientation_deg,
            compactness=compactness,
            component_count=int(n_components),
            largest_component_fraction=float(largest_fraction),
        )
    return features


def _region_name_map(atlas: Any, region_ids: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    for sid in region_ids:
        record = _lookup_structure_record(atlas, int(sid))
        name = record.get("name") or record.get("acronym") or ""
        if not name:
            name = f"region {int(sid)}"
        out[int(sid)] = name
    return out


def _score_regions(
    base: dict[int, RegionFeatures],
    comparisons: dict[float, dict[int, RegionFeatures]],
    names: dict[int, str],
    *,
    atlas: Any | None = None,
    atlas_name: str = "",
    plane: str = "",
    top_k: int,
) -> list[RegionScore]:
    scored: list[RegionScore] = []

    for sid, feat in base.items():
        area_deltas: list[float] = []
        dxs: list[float] = []
        dys: list[float] = []
        elong_deltas: list[float] = []
        compact_deltas: list[float] = []
        neg_areas: list[float] = []
        pos_areas: list[float] = []

        for offset, comp_map in sorted(comparisons.items(), key=lambda kv: kv[0]):
            comp = comp_map.get(sid)
            if comp is None:
                comp_area = 0.0
                comp_cx = feat.centroid_x
                comp_cy = feat.centroid_y
                comp_elong = 1.0
                comp_compact = 0.0
            else:
                comp_area = comp.area_fraction
                comp_cx = comp.centroid_x
                comp_cy = comp.centroid_y
                comp_elong = comp.elongation
                comp_compact = comp.compactness

            area_delta = comp_area - feat.area_fraction
            area_deltas.append(area_delta)
            dxs.append(comp_cx - feat.centroid_x)
            dys.append(comp_cy - feat.centroid_y)
            elong_deltas.append(comp_elong - feat.elongation)
            compact_deltas.append(comp_compact - feat.compactness)
            if offset < 0:
                neg_areas.append(comp_area)
            elif offset > 0:
                pos_areas.append(comp_area)

        if area_deltas:
            local_change = (
                float(np.mean(np.abs(area_deltas)))
                + 0.35 * float(np.mean(np.hypot(dxs, dys)))
                + 0.20 * float(np.mean(np.abs(elong_deltas)))
                + 0.10 * float(np.mean(np.abs(compact_deltas)))
            )
        else:
            local_change = 0.0

        area_trend = 0.0
        if neg_areas and pos_areas:
            area_trend = float(np.mean(pos_areas) - np.mean(neg_areas))

        size_weight = min(1.0, (feat.area_fraction / 0.015) ** 0.60)
        if feat.area_fraction <= 0.0025:
            size_weight *= 0.20
        if feat.component_count >= 5 and feat.largest_component_fraction < 0.45:
            size_weight *= 0.35

        shape_salience = (
            max(0.0, feat.elongation - 1.1)
            + 0.40 * max(0.0, 0.60 - feat.compactness)
            + 0.30 * max(0.0, feat.component_count - 1) * (1.0 - feat.largest_component_fraction)
        )
        orientation_salience = min(abs(feat.orientation_deg) / 90.0, 1.0)

        score = size_weight * (
            0.95 * local_change
            + 0.55 * shape_salience
            + 0.25 * orientation_salience
        )
        if score <= 0.0:
            continue

        scored.append(
            RegionScore(
                region_id=sid,
                name=names.get(sid, f"region {sid}"),
                score=float(score),
                local_change_magnitude=float(local_change),
                centroid_shift_x=float(np.mean(dxs) if dxs else 0.0),
                centroid_shift_y=float(np.mean(dys) if dys else 0.0),
                area_trend=area_trend,
            )
        )

    scored.sort(key=lambda s: (-s.score, s.name.lower(), s.region_id))
    return _select_ranked_regions(
        scored,
        atlas=atlas,
        atlas_name=atlas_name,
        plane=plane,
        top_k=top_k,
    )


def _select_ranked_regions(
    scored: list[RegionScore],
    *,
    atlas: Any,
    atlas_name: str,
    plane: str,
    top_k: int,
) -> list[RegionScore]:
    cap = max(1, top_k)
    if not scored:
        return []

    priority_ranks = (
        _priority_region_ranks(atlas, atlas_name, plane)
        if atlas is not None and atlas_name
        else {}
    )
    priority = [score for score in scored if score.region_id in priority_ranks]
    priority.sort(
        key=lambda s: (
            priority_ranks.get(s.region_id, 10_000),
            -s.score,
            s.name.lower(),
            s.region_id,
        )
    )
    priority_cap = max(1, cap // 2)

    selected: list[RegionScore] = []
    seen: set[int] = set()

    def add(score: RegionScore) -> None:
        if len(selected) >= cap or score.region_id in seen:
            return
        selected.append(score)
        seen.add(score.region_id)

    for score in priority[:priority_cap]:
        add(score)
    for score in scored:
        add(score)
    return selected


def _priority_region_ranks(
    atlas: Any,
    atlas_name: str,
    plane: str,
) -> dict[int, int]:
    try:
        loader = _landmark_loader()
        landmarks = loader.landmarks_for_orientation(plane.lower())
    except Exception:
        return {}

    ranks: dict[int, int] = {}
    for rank, landmark_name in enumerate(landmarks):
        try:
            region_ids = loader.resolve(landmark_name, atlas, atlas_name)
        except Exception:
            continue
        for region_id in region_ids:
            ranks.setdefault(int(region_id), rank)
    return ranks


@lru_cache(maxsize=1)
def _landmark_loader() -> Any:
    if str(_GEMMA_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(_GEMMA_DATA_DIR))
    from landmarks import LandmarkLoader  # noqa: PLC0415

    return LandmarkLoader()


def _render_clue(score: RegionScore, feat: RegionFeatures | None) -> str:
    if feat is None:
        return ""

    descriptors: list[str] = []
    if feat.elongation >= 1.8:
        descriptors.append("elongated")
    if feat.compactness <= 0.38:
        descriptors.append("irregular")
    if feat.component_count >= 3 and feat.largest_component_fraction <= 0.70:
        descriptors.append("fragmented")
    if abs(feat.orientation_deg) <= 20.0 and feat.elongation >= 1.6:
        descriptors.append("horizontally oriented")
    elif abs(feat.orientation_deg) >= 70.0 and feat.elongation >= 1.6:
        descriptors.append("vertically oriented")

    if score.area_trend >= 0.01:
        descriptors.append("expanding across nearby slices")
    elif score.area_trend <= -0.01:
        descriptors.append("shrinking across nearby slices")

    if score.centroid_shift_x >= 0.03:
        descriptors.append("centroid shifts rightward")
    elif score.centroid_shift_x <= -0.03:
        descriptors.append("centroid shifts leftward")

    if score.centroid_shift_y >= 0.03:
        descriptors.append("centroid shifts downward")
    elif score.centroid_shift_y <= -0.03:
        descriptors.append("centroid shifts upward")

    if not descriptors:
        descriptors.append("has a stable footprint")
    return f"{score.name} " + " and ".join(descriptors)
