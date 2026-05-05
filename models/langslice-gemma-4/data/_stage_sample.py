"""--stage sample: enumerate viable tuples, render examples, write draft manifest."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))
sys.path.insert(0, str(REPO_ROOT / "_local" / "eval" / "lib"))

import bbox_io  # noqa: E402
import build_bbox_data  # noqa: E402
import region_bbox  # noqa: E402
from augmentation.oblique import get_oblique_slice  # noqa: E402
from landmarks import LandmarkLoader  # noqa: E402

from langslice_harness.atlas.core import load_atlas  # noqa: E402

log = logging.getLogger("stage_sample")


def _iter_viable_tuples(
    *,
    atlases: list[str],
    orientations: list[str],
    loader: LandmarkLoader,
    atlas_objs: dict[str, object],
) -> Iterable[tuple[str, str, str]]:
    for atlas_name in atlases:
        atlas = atlas_objs[atlas_name]
        for orientation in orientations:
            for landmark in loader.landmarks_for_orientation(orientation):
                ids = loader.resolve(landmark, atlas, atlas_name)
                if not ids:
                    continue
                yield (atlas_name, orientation, landmark)


def _region_mm_extent(
    atlas, orientation: str, region_ids: set[int]
) -> tuple[float, float] | None:
    """Walk the annotation volume along the position axis; return (min, max) mm
    where any region voxel exists, else None."""
    annotation = atlas.annotation
    mask = np.isin(annotation, list(region_ids))
    axis = {"coronal": 0, "horizontal": 1, "sagittal": 2}[orientation]
    along = mask.any(axis=tuple(a for a in (0, 1, 2) if a != axis))
    indices = np.where(along)[0]
    if len(indices) == 0:
        return None
    res_um = float(atlas.resolution[axis])
    return (
        float(indices.min()) * res_um / 1000.0,
        float(indices.max()) * res_um / 1000.0,
    )


def _render_example_atlas(
    *,
    atlas_name: str,
    orientation: str,
    landmark: str,
    region_ids: set[int],
    n_sections: int,
    spacings_mm: Sequence[float],
    anchor_mm: float,
    source_type: str,  # "augmented_atlas" or "reference_atlas"
    rng: np.random.Generator,
    example_id: str,
    out_dir: Path,
) -> dict | None:
    """Render N atlas sections at the chosen positions; compute bboxes; return
    a draft-manifest record or None if any section's bbox fails."""
    from synth_dataset import render, sample_spec  # type: ignore

    atlas = load_atlas(atlas_name)

    section_paths: list[str] = []
    bboxes: list[Any] = []
    is_hemisphere = False  # whole-brain by default for atlas rendering

    positions_mm = [
        anchor_mm + sum(spacings_mm[:k])
        for k in range(n_sections)
    ]

    modality = None
    for k, pos_mm in enumerate(positions_mm):
        if source_type == "augmented_atlas":
            spec = sample_spec(
                rng=rng, atlases=[atlas_name],
                position_strata="uniform", oblique_prob=0.0,
            )
            object.__setattr__(spec, "plane", orientation)
            object.__setattr__(spec, "position_mm", pos_mm)
            image_f32, _ = render(spec)
            modality = spec.modality
        else:
            ref_u8, _ann = get_oblique_slice(
                atlas, base_position_mm=pos_mm, plane=orientation,
                yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            )
            image_f32 = (np.repeat(ref_u8[..., None], 3, axis=2) / 255.0).astype(np.float32)
            modality = None

        # Annotation slice for bbox.
        _ref, ann_slice = get_oblique_slice(
            atlas, base_position_mm=pos_mm, plane=orientation,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        )
        hemi_slice = None if is_hemisphere else _slice_hemispheres(
            atlas, base_position_mm=pos_mm, plane=orientation,
        )
        bbox = region_bbox.bbox_from_atlas_slice(
            annotation_slice=ann_slice,
            hemisphere_slice=hemi_slice,
            region_ids=region_ids,
            is_hemisphere=is_hemisphere,
        )
        if bbox is None:
            return None
        bboxes.append(bbox)

        # Save the section PNG.
        section_dir = out_dir / "section_images" / example_id
        section_dir.mkdir(parents=True, exist_ok=True)
        section_path = section_dir / f"section_{k:02d}.png"
        Image.fromarray(np.clip(image_f32 * 255.0, 0, 255).astype(np.uint8)).save(section_path)
        section_paths.append(str(section_path.relative_to(REPO_ROOT)))

    atlas_version = (atlas.metadata or {}).get("version") or "unknown"
    return {
        "id": example_id,
        "atlas": atlas_name,
        "atlas_version": atlas_version,
        "orientation": orientation,
        "region": landmark,
        "source_type": source_type,
        "source_brain": None,
        "modality": modality,
        "is_hemisphere": is_hemisphere,
        "section_image_paths": section_paths,
        "section_positions_mm": positions_mm,
        "bboxes": bboxes,
    }


def _slice_hemispheres(atlas, *, base_position_mm: float, plane: str) -> np.ndarray:
    """Project atlas.hemispheres through get_oblique_slice into the section frame."""
    hemi_atlas = type("HemisphereAtlas", (), {})()
    hemi_atlas.reference = atlas.reference
    hemi_atlas.annotation = atlas.hemispheres
    hemi_atlas.resolution = atlas.resolution
    hemi_atlas.orientation = atlas.orientation
    hemi_atlas.atlas_name = getattr(atlas, "atlas_name", "hemisphere_proxy")
    _ref, hemi_slice = get_oblique_slice(
        hemi_atlas, base_position_mm=base_position_mm, plane=plane,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )
    return hemi_slice


def _render_overlay_strip(record: dict, out_dir: Path) -> Path:
    """Compose N section PNGs into a horizontal strip with bbox rectangles drawn.
    Cyan = left, magenta = right (or single bbox in cyan for sagittal/hemisphere)."""
    images = [Image.open(REPO_ROOT / p) for p in record["section_image_paths"]]
    h = max(im.height for im in images)
    w = sum(im.width for im in images)
    strip = Image.new("RGB", (w, h), (24, 24, 28))

    cursor_x = 0
    for im, bbox in zip(images, record["bboxes"], strict=False):
        strip.paste(im, (cursor_x, 0))
        draw = ImageDraw.Draw(strip)
        if record["is_hemisphere"]:
            x1, y1, x2, y2 = bbox
            draw.rectangle(
                [cursor_x + x1, y1, cursor_x + x2, y2], outline="cyan", width=3
            )
        else:
            for side, color in (("left", "cyan"), ("right", "magenta")):
                if bbox.get(side) is not None:
                    x1, y1, x2, y2 = bbox[side]
                    draw.rectangle(
                        [cursor_x + x1, y1, cursor_x + x2, y2], outline=color, width=3
                    )
        cursor_x += im.width

    overlays_dir = out_dir / "draft_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlays_dir / f"{record['id']}_strip.png"
    strip.save(out_path)
    return out_path


def _load_atlas_cached(name: str) -> object:
    """Cache atlas loads; called from _render_example_real."""
    if not hasattr(_load_atlas_cached, "_cache"):
        _load_atlas_cached._cache = {}
    cache = _load_atlas_cached._cache
    if name not in cache:
        cache[name] = load_atlas(name)
    return cache[name]


def _load_real_section_record(*, section_id: str, manifest_path: Path) -> dict:
    """Load a section's image path, anchoring/markers, and registration helper.

    Reads the canonical manifest, picks the section's record, calls
    `section_to_atlas_voxel` to build the pixel->voxel function, and computes
    the projected midline-line function.
    """
    from registration import section_to_atlas_voxel  # type: ignore

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("section_id") == section_id:
                break
        else:
            raise KeyError(f"section {section_id!r} not in manifest {manifest_path}")

    atlas = _load_atlas_cached(rec["atlas"])
    slice_record = rec["slice_record"]
    pixel_to_voxel = section_to_atlas_voxel(slice_record, atlas)

    H = int(slice_record["height"])
    W = int(slice_record["width"])

    # Project the atlas ML-midline plane to a line in section pixel space.
    # BrainGlobe annotation order is (AP, DV, ML), while pixel_to_voxel returns
    # QuickNII order (x_ml, y_ap, z_dv). Midline is therefore voxel[0].
    midline_voxel_x = float(atlas.annotation.shape[2]) / 2.0

    def midline_x_at_row(i: float) -> float:
        # Bisection along the row to find the (i, j) where x_ml crosses midline.
        lo, hi = 0.0, float(W - 1)
        for _ in range(20):
            mid = (lo + hi) / 2.0
            voxel = pixel_to_voxel(mid, i)
            if voxel[0] < midline_voxel_x:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return {
        "section_id": section_id,
        "image_path": REPO_ROOT / rec["image_path"],
        "image_shape": (H, W),
        "pixel_to_voxel": pixel_to_voxel,
        "midline_x_at_row": midline_x_at_row,
        "is_hemisphere": bool(rec.get("is_hemisphere", False)),
        "position_mm": float(rec["position_mm"]),
        "atlas": rec["atlas"],
    }


def _render_example_real(
    *,
    atlas_name: str,
    orientation: str,
    landmark: str,
    region_ids: set[int],
    brain_id: str,
    section_ids: list[str],
    out_dir: Path,
    example_id: str,
    manifest_path: Path | None = None,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Render a real-histology example: sample N in {4..8}, choose spacings and
    anchor to fit available sections, compute bboxes."""
    if manifest_path is None:
        manifest_path = REPO_ROOT / "_local" / "eval" / "data" / "manifest.jsonl"
    if rng is None:
        rng = np.random.default_rng()

    section_records = [
        _load_real_section_record(section_id=sid, manifest_path=manifest_path)
        for sid in section_ids
    ]
    section_records.sort(key=lambda r: r["position_mm"])

    chosen = _choose_real_section_subset(section_records, rng=rng)
    if chosen is None:
        return None

    atlas = _load_atlas_cached(atlas_name)
    is_hemisphere = chosen[0]["is_hemisphere"]
    section_paths: list[str] = []
    bboxes: list[Any] = []
    positions_mm: list[float] = []

    for k, rec in enumerate(chosen):
        bbox = region_bbox.bbox_from_real_section(
            section_image_shape=rec["image_shape"],
            pixel_to_voxel=rec["pixel_to_voxel"],
            annotation_volume=atlas.annotation,
            region_ids=region_ids,
            midline_x_at_row=rec["midline_x_at_row"],
            is_hemisphere=is_hemisphere,
            grid_step=8,
        )
        if bbox is None:
            return None

        # Resize and save into the example's output dir for QC + submission.
        from PIL import Image as _Image
        section_dir = out_dir / "section_images" / example_id
        section_dir.mkdir(parents=True, exist_ok=True)
        section_path = section_dir / f"section_{k:02d}.png"
        im = _Image.open(rec["image_path"])
        original_w, original_h = im.size
        im.thumbnail((768, 768))
        scale_x = im.width / float(original_w)
        scale_y = im.height / float(original_h)
        bboxes.append(region_bbox.scale_bbox(bbox, scale_x=scale_x, scale_y=scale_y))
        im.save(section_path)
        try:
            section_paths.append(str(section_path.relative_to(REPO_ROOT)))
        except ValueError:
            section_paths.append(str(section_path))
        positions_mm.append(rec["position_mm"])

    atlas_version = (atlas.metadata or {}).get("version") or "unknown"
    return {
        "id": example_id,
        "atlas": atlas_name,
        "atlas_version": atlas_version,
        "orientation": orientation,
        "region": landmark,
        "source_type": "real_histology",
        "source_brain": brain_id,
        "modality": None,
        "is_hemisphere": is_hemisphere,
        "section_image_paths": section_paths,
        "section_positions_mm": positions_mm,
        "bboxes": bboxes,
    }


def _choose_real_section_subset(
    section_records: list[dict],
    rng: np.random.Generator,
) -> list[dict] | None:
    """Sample N uniformly from {4..8}, then choose per-gap spacings/anchor to fit records."""
    if len(section_records) < 4:
        return None
    by_pos = sorted(section_records, key=lambda r: r["position_mm"])
    for _attempt in range(100):
        n = build_bbox_data.sample_section_count(rng)
        if len(by_pos) < n:
            continue
        spacings = build_bbox_data.sample_spacings_mm(rng, n_gaps=n - 1)
        positions = np.array([float(r["position_mm"]) for r in by_pos])
        min_pos = float(positions.min())
        max_pos = float(positions.max())
        anchor = build_bbox_data.sample_anchor_mm(
            rng=rng, region_mm_min=min_pos, region_mm_max=max_pos,
            spacings_mm=spacings,
        )
        if anchor is None:
            continue
        targets = np.array([anchor + sum(spacings[:k]) for k in range(n)])
        chosen: list[dict] = []
        used: set[int] = set()
        for target in targets:
            order = np.argsort(np.abs(positions - target))
            idx = next((int(i) for i in order if int(i) not in used), None)
            if idx is None:
                break
            used.add(idx)
            chosen.append(by_pos[idx])
        chosen.sort(key=lambda r: r["position_mm"])
        if len(chosen) != n:
            continue
        gaps = [
            b["position_mm"] - a["position_mm"]
            for a, b in zip(chosen, chosen[1:], strict=False)
        ]
        if all(0.2 <= gap <= 0.8 for gap in gaps):
            return chosen
    return None


def run_stage_sample(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    source_counts: dict[tuple[str, str], int] = {}
    loader = LandmarkLoader()

    atlases = ["allen_mouse_25um", "whs_sd_rat_39um"]
    orientations = ["coronal", "sagittal", "horizontal"]
    atlas_objs = {name: load_atlas(name) for name in atlases}

    if args.coverage_index.exists():
        coverage_index = json.loads(args.coverage_index.read_text(encoding="utf-8"))
    else:
        coverage_index = {}

    viable = list(_iter_viable_tuples(
        atlases=atlases, orientations=orientations,
        loader=loader, atlas_objs=atlas_objs,
    ))
    log.info("Viable (atlas, orientation, region) tuples: %d", len(viable))
    if not viable:
        log.error("No viable tuples - check landmark_atlas_map.json")
        return 1

    target_per_tuple = max(1, args.target_total // len(viable))

    records: list[dict] = []
    next_id = 0
    for atlas_name, orientation, landmark in viable:
        atlas = atlas_objs[atlas_name]
        region_ids = loader.resolve(landmark, atlas, atlas_name)
        extent = _region_mm_extent(atlas, orientation, region_ids)
        if extent is None:
            continue
        rmin, rmax = extent
        produced = 0
        attempts = 0
        while produced < target_per_tuple and attempts < target_per_tuple * 5:
            attempts += 1
            decision = build_bbox_data.pick_source(
                atlas=atlas_name, orientation=orientation, landmark=landmark,
                coverage_index=coverage_index, rng=rng,
                source_counts=source_counts,
            )
            if decision["source_type"] == "real_histology":
                example_id = f"bbox_{next_id:06d}"
                next_id += 1
                rec = _render_example_real(
                    atlas_name=atlas_name, orientation=orientation,
                    landmark=landmark, region_ids=region_ids,
                    brain_id=decision["source_brain"],
                    section_ids=decision["available_section_ids"],
                    out_dir=args.out_dir, example_id=example_id,
                    rng=rng,
                )
                if rec is None:
                    continue
                _render_overlay_strip(rec, args.out_dir)
                records.append(rec)
                source_counts[(decision["source_brain"], landmark)] = (
                    source_counts.get((decision["source_brain"], landmark), 0) + 1
                )
                produced += 1
                continue

            n = build_bbox_data.sample_section_count(rng)
            spacings = build_bbox_data.sample_spacings_mm(rng, n_gaps=n - 1)
            anchor = build_bbox_data.sample_anchor_mm(
                rng=rng, region_mm_min=rmin, region_mm_max=rmax,
                spacings_mm=spacings,
            )
            if anchor is None:
                continue

            example_id = f"bbox_{next_id:06d}"
            next_id += 1
            rec = _render_example_atlas(
                atlas_name=atlas_name, orientation=orientation,
                landmark=landmark, region_ids=region_ids,
                n_sections=n, spacings_mm=spacings, anchor_mm=anchor,
                source_type=decision["source_type"], rng=rng,
                example_id=example_id, out_dir=args.out_dir,
            )
            if rec is None:
                continue
            _render_overlay_strip(rec, args.out_dir)
            records.append(rec)
            produced += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bbox_io.write_jsonl(args.out_dir / "draft_manifest.jsonl", records)
    log.info("Wrote %d records to draft_manifest.jsonl", len(records))
    return 0
