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
                # Real-histology rendering is implemented separately; for the
                # first iteration, fall back to atlas if real path is not yet
                # ready. The orchestrator's hook is `_render_example_real`.
                log.debug("real-histology path not yet implemented for %s/%s/%s",
                          atlas_name, orientation, landmark)
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
