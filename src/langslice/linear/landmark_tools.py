"""Pixel-addressed paired landmarks and an undoable native-compatible TPS warp."""

from __future__ import annotations

import copy
import json
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import denormalized_affine
from langslice.landmark_warp import fit_spline
from langslice.linear.render import (
    OVERLAY_LONG_EDGE,
    PREVIEW_LONG_EDGE,
    canvas_geometry,
    image_to_part,
    normalize_border_style,
    physical_views,
    render_slice,
)
from langslice.linear.state import StackState
from langslice.linear.transform import affine_fit, calibrate, physical_params
from langslice.space import Plane


def geometry_signature(state: StackState, record: Any) -> str:
    return json.dumps(
        {
            "id": record.id,
            "atlas": state.atlas,
            "plane": state.plane,
            "angles": state.cutting_angles_deg,
            "position": record.position_mm,
            "rotation": record.rotation_deg,
            "flip": record.flip,
        },
        sort_keys=True,
    )


def image_to_section(points: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    """Returned-image pixel centres -> section pixel centres, undoing caption/crop/resize."""
    content = np.asarray(frame["content_box"], dtype=float)
    crop = np.asarray(frame["canvas_box"], dtype=float)
    if (
        not np.isfinite(points).all()
        or np.any(points < content[:2])
        or np.any(points >= content[2:])
    ):
        raise ValueError("Landmarks must be inside image content, excluding its caption")
    scale = (crop[2:] - crop[:2]) / (content[2:] - content[:2])
    return (points - content[:2] + 0.5) * scale - 0.5 + crop[:2] - frame["section_offset"]


def section_to_image(points: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    content = np.asarray(frame["content_box"], dtype=float)
    crop = np.asarray(frame["canvas_box"], dtype=float)
    scale = (content[2:] - content[:2]) / (crop[2:] - crop[:2])
    return (points + frame["section_offset"] - crop[:2] + 0.5) * scale - 0.5 + content[:2]


def make_landmark_tools(
    state: StackState, ctx: Any, snapshot: Callable[[], None], commit: Callable[..., dict[str, Any]]
) -> list[Callable[..., Any]]:
    # View identities are session-local and bounded; checkpoints retain canonical points.
    views: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def geometry(record: Any) -> tuple[Any, float, str]:
        if record.position_mm is None:
            raise ValueError("The section requires a position")
        section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
        um, source = calibrate(state, ctx, record, section)
        return section, um, source

    def render(
        record: Any, section: Any, um: float, transform: dict[str, Any] | None, **kwargs: Any
    ) -> tuple[list[Any], float]:
        matrix: Any = (
            denormalized_affine(transform["params"], section.size) if transform else
            {"rotation_deg": 0., "scale_x": 1., "scale_y": 1.,
             "translate_x_mm": 0., "translate_y_mm": 0.}
        )
        return physical_views(
            section,
            um,
            ctx.atlas,
            record.position_mm,
            cast(Plane, state.plane),
            state.pitch_deg,
            state.yaw_deg,
            matrix,
            spline=(transform or {}).get("spline"),
            long_edge=OVERLAY_LONG_EDGE,
            **kwargs,
        )

    def pair_state(record: Any) -> tuple[list[dict[str, Any]], bool]:
        if record.landmark_pairs is not None:
            stale = record.landmark_frame != geometry_signature(state, record)
            return copy.deepcopy(record.landmark_pairs), stale
        spline = (record.transform or {}).get("spline")
        if not spline:
            return [], False
        count = len(spline["source"])
        return [
            {"id": int(ident), "label": str(label), "source": list(a), "target": list(b)}
            for ident, label, a, b in zip(
                spline.get("ids", range(1, count + 1)),
                spline.get("labels", [""] * count), spline["source"], spline["target"],
                strict=True,
            )
        ], False

    def signature(record: Any) -> str:
        pairs, stale = pair_state(record)
        return json.dumps([geometry_signature(state, record), pairs, stale], sort_keys=True)

    def project_pairs(
        pairs: list[dict[str, Any]], size: Any, frames: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {"id": p["id"], "label": p["label"],
             "slice_xy": section_to_image(np.asarray(p["source"]) * size, frames[0]).tolist(),
             "atlas_xy": section_to_image(np.asarray(p["target"]) * size, frames[1]).tolist()}
            for p in pairs
        ]

    def mark(image: Any, pairs: list[dict[str, Any]], key: str, frame: Any) -> Any:
        result = image.copy()
        draw = ImageDraw.Draw(result)
        font = ImageFont.load_default(size=14)
        x0, y0, x1, y1 = frame["content_box"]
        for pair in pairs:
            x, y = pair[key]
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            draw.ellipse((x-4, y-4, x+4, y+4), outline="#00ffff", width=1)
            label = str(pair["id"])
            bx, by = min(x+6, x1-20), max(y0, y-17)
            bounds = draw.textbbox((bx, by), label, font=font)
            draw.rectangle(bounds, fill="black")
            draw.text((bx, by), label, font=font, fill="#00ffff")
        return result

    def build_view(
        record: Any, zoom: list[float] | None, border_color: str, border_thickness: float
    ) -> dict[str, Any]:
        normalize_border_style(border_color, border_thickness)
        window = [float(v) for v in (zoom or [])]
        if window and (
            len(window) != 4 or not np.isfinite(window).all()
            or not (0 <= window[0] < window[2] <= 1 and 0 <= window[1] < window[3] <= 1)
        ):
            raise ValueError("zoom must be [x0,y0,x1,y1] inside 0..1")
        section, um, source = geometry(record)
        frames: list[dict[str, Any]] = []
        # The correspondence panels never include an applied affine or spline.
        images, _ = render(
            record, section, um, None, mode="side_by_side", outlines="none",
            zoom=window, frames=frames, label=f"{record.id} unwarped",
        )
        overlay, iou = render(
            record, section, um, record.transform, mode="overlay", zoom=window,
            border_color=border_color, border_thickness=border_thickness,
            frames=frames, label=record.id,
        )
        pairs, stale = pair_state(record)
        existing = project_pairs([] if stale else pairs, section.size, frames)
        images = [mark(im, existing, field, frame) for im, field, frame in zip(
            images, ["slice_xy", "atlas_xy"], frames[:2], strict=True,
        )]
        parts = [image_to_part(im) for im in images + overlay]
        key = uuid.uuid4().hex
        views[key] = {
            "signature": signature(record), "id": record.id, "frames": frames,
            "zoom": window, "calibration_source": source,
        }
        while len(views) > 32:
            views.popitem(last=False)
        return {
            "status": "ok", "id": record.id, "view_id": key,
            "coordinate_system": "image pixels [x,y]; top-left origin; x right, y down",
            "images_info": [
                {"image_index": i, "role": role, **frame}
                for i, (role, frame) in enumerate(zip(
                    ["unwarped_slice", "atlas", "current_overlay"], frames, strict=True,
                ))
            ],
            "existing_pairs": existing, "landmark_frame_stale": stale,
            "iou": iou, "image_detail": "high", TOOL_MEDIA_PARTS_KEY: parts,
        }

    def view_landmarks(
        slice_id: str, zoom: list[float] | None = None,
        border_color: str = "yellow", border_thickness: float = 0.5,
    ) -> dict[str, Any]:
        """Show stable unwarped slice and atlas with numbered correspondence pairs.

        Args:
            slice_id: Filename or corrected index.
            zoom: Optional [x0,y0,x1,y1] fractions of the full canvas.
            border_color: Atlas outline color, named or #RRGGBB.
            border_thickness: Outline width in pixels, 0.25..8.

        Returns:
            image0 is the unwarped oriented section; image1 is the atlas. Match
            corresponding numbered points using image pixels [x,y], top-left
            origin, x right, y down, inside content_box. image2 is the current
            applied overlay. Applying a warp never moves the first two images.
            edit_landmarks adds/moves/deletes individual pairs; warp_landmarks
            applies the saved pairs. A stale landmark frame requires reset=True
            in edit_landmarks before reusing drafts. Cropped-out pairs remain
            listed but cannot be selected in this view.
        """
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE", "id": slice_id}
        try:
            return build_view(record, zoom, border_color, border_thickness)
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            return {"status": "error", "error": "LANDMARK_VIEW_FAILED", "message": str(exc)}

    def checked_view(slice_id: str, view_id: str) -> tuple[Any, Any, Any]:
        record = state.resolve(slice_id)
        view = views.get(view_id)
        if record is None or view is None or view["id"] != record.id:
            return record, view, {"status": "error", "error": "UNKNOWN_LANDMARK_VIEW"}
        if view["signature"] != signature(record):
            return record, view, {"status": "refused", "error": "STALE_LANDMARK_VIEW"}
        return record, view, None

    def coordinate(value: Any, frame: Any, size: Any, *, source: bool) -> list[float]:
        point = np.asarray(value, dtype=float)
        if point.shape != (2,):
            raise ValueError("Each point must be [x,y]")
        raw = image_to_section(point, frame)
        if source and (np.any(raw < -0.5) or np.any(raw >= np.asarray(size) - 0.5)):
            raise ValueError("Slice landmarks must be inside the unwarped section image")
        return (raw / size).tolist()

    def make_pair(value: dict[str, Any], ident: int, view: Any, size: Any) -> dict[str, Any]:
        return {"id": ident, "label": str(value.get("label", ""))[:120],
                "source": coordinate(value["slice_xy"], view["frames"][0], size, source=True),
                "target": coordinate(value["atlas_xy"], view["frames"][1], size, source=False)}

    def edit_landmarks(
        slice_id: str, view_id: str, add: list[dict[str, Any]] | None = None,
        move: list[dict[str, Any]] | None = None, delete: list[int] | None = None,
        reset: bool = False, border_color: str = "yellow", border_thickness: float = 0.5,
    ) -> dict[str, Any]:
        """Edit persistent numbered landmark pairs without changing the applied warp.

        Args:
            slice_id: Filename or corrected index.
            view_id: Current view_landmarks or edit_landmarks view_id.
            add: New {slice_xy:[x,y],atlas_xy:[x,y],label:optional} pairs in
                unwarped image0 and atlas image1 pixels. IDs are assigned.
            move: Existing {id, slice_xy:optional, atlas_xy:optional, label:optional}.
                Supply one endpoint to move just that endpoint; others stay fixed.
            delete: Pair IDs to remove. Drafts may contain fewer than four pairs.
            reset: Clear all draft pairs, including stale geometry, before edits.
            border_color: Atlas outline color, named or #RRGGBB.
            border_thickness: Outline width in pixels, 0.25..8.

        Returns:
            One undoable draft checkpoint, fresh view_id, numbered unwarped slice
            and atlas, and current applied overlay. Call warp_landmarks without
            pairs to apply the edited set. Any invalid edit refuses the whole batch.
        """
        record, view, error = checked_view(slice_id, view_id)
        if error:
            return error
        try:
            pairs, stale = pair_state(record)
            if stale and not reset:
                return {"status": "refused", "error": "STALE_LANDMARK_FRAME"}
            section, _, _ = geometry(record)
            next_id = max(record.landmark_next_id, max((p["id"] for p in pairs), default=0)+1)
            if reset:
                pairs = []
            by_id = {p["id"]: p for p in pairs}
            deletions = delete or []
            if len(set(deletions)) != len(deletions):
                raise ValueError("Each deleted pair ID must occur once")
            for ident in deletions:
                if isinstance(ident, bool) or ident not in by_id:
                    raise ValueError(f"Unknown pair ID: {ident}")
                del by_id[ident]
            moved: set[int] = set()
            for update in move or []:
                ident = update["id"]
                if isinstance(ident, bool) or ident not in by_id or ident in moved:
                    raise ValueError(f"Unknown or repeated pair ID: {ident}")
                moved.add(ident)
                pair = by_id[ident]
                for field, endpoint, index in [
                    ("slice_xy", "source", 0), ("atlas_xy", "target", 1),
                ]:
                    if field in update:
                        pair[endpoint] = coordinate(
                            update[field], view["frames"][index], section.size, source=index == 0,
                        )
                if "label" in update:
                    pair["label"] = str(update["label"])[:120]
            pairs = list(by_id.values())
            for value in add or []:
                pairs.append(make_pair(value, next_id, view, section.size))
                next_id += 1
            if len(pairs) > 64:
                raise ValueError("At most 64 landmark pairs are supported")
            prospective = copy.deepcopy(record)
            prospective.landmark_pairs = pairs
            prospective.landmark_frame = geometry_signature(state, record)
            prospective.landmark_next_id = next_id
            result = build_view(prospective, view["zoom"], border_color, border_thickness)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return {"status": "refused", "error": "INVALID_LANDMARK_EDIT", "message": str(exc)}
        snapshot()
        record.landmark_pairs = pairs
        record.landmark_frame = prospective.landmark_frame
        record.landmark_next_id = next_id
        commit(record.id)
        result.update(written=True, applied=False, landmarks=len(pairs))
        return result

    def warp_landmarks(
        slice_id: str,
        view_id: str,
        pairs: list[dict[str, Any]] | None = None,
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        template_opacity: float = 0.0,
        note: str = "",
        method: str = "affine",
    ) -> dict[str, Any]:
        """Fit an affine or spline using the same persistent landmark correspondences.

        Args:
            slice_id: Filename or corrected index.
            view_id: The unchanged view_landmarks view used to choose coordinates.
            pairs: Omit to apply saved edited pairs. Optional complete set of
                3..64 {slice_xy:[x,y],atlas_xy:[x,y],label:optional} pairs in
                UNWARPED image0 and atlas image1 pixels. Replaces the
                complete landmark warp, including its affine component; does not
                stack another deformation. Include unchanged correspondences to
                constrain anatomy that is already aligned.
            border_color: Atlas outline color, named or #RRGGBB.
            border_thickness: Outline width in output pixels, 0.25..8.
            template_opacity: Atlas intensity overlay, 0..1.
            note: Optional registration note.
            method: "affine" (default, at least three pairs) or "spline" (four).
                Start with affine; reuse the same pairs for local spline correction.

        Returns:
            Writes/checkpoints one undoable transform and returns before/after
            full-canvas overlays, numbered unwarped slice/atlas, landmark residuals
            and silhouette IoU. Images 0/1 are before/after; images 2/3 are the
            correspondence panels. Their pixel coordinates use the returned view_id.
            Invalid/stale coordinates or detected folding are refused before any
            write. Correspondence panels stay unwarped after applying.
        """
        record, view, error = checked_view(slice_id, view_id)
        if error:
            return error
        try:
            normalize_border_style(border_color, border_thickness)
            opacity = float(template_opacity)
            if not np.isfinite(opacity) or not 0 <= opacity <= 1:
                raise ValueError("template_opacity must be 0..1")
            section, um, source = geometry(record)
            canonical, stale = pair_state(record)
            next_id = max(record.landmark_next_id, max((p["id"] for p in canonical), default=0)+1)
            if pairs is not None:
                if not isinstance(pairs, list):
                    raise ValueError("pairs must be a list")
                canonical = [
                    make_pair(p, next_id+i, view, section.size) for i, p in enumerate(pairs)
                ]
                next_id += len(canonical)
            elif stale:
                return {"status": "refused", "error": "STALE_LANDMARK_FRAME"}
            if method not in {"affine", "spline"}:
                raise ValueError("method must be affine or spline")
            minimum = 3 if method == "affine" else 4
            if not minimum <= len(canonical) <= 64:
                raise ValueError(f"Supply {minimum}..64 paired landmarks for {method}")
            raw = np.asarray([p["source"] for p in canonical]) * section.size
            target = np.asarray([p["target"] for p in canonical]) * section.size
            previous = copy.deepcopy(record.transform)
            original_um, _ = ctx.calibration(record.id)
            if original_um is not None:
                from pathlib import Path

                with Image.open(Path(state.image_folder) / record.id) as original:
                    dimensions = np.asarray(original.size, dtype=float)
                if record.rotation_deg % 180:
                    dimensions = dimensions[::-1]
                extent = dimensions * original_um / 1000
            else:
                extent = np.asarray(section.size) * um / 1000
            labels = [p["label"] for p in canonical]
            spline = {
                "source": (raw / section.size).tolist(),
                "target": (target / section.size).tolist(),
                "extent_mm": extent.tolist(),
                "labels": labels,
                "ids": [p["id"] for p in canonical],
            }
            source_mm = np.asarray(spline["source"]) * extent
            target_mm = np.asarray(spline["target"]) * extent
            written = (
                copy.deepcopy(previous) if previous else {
                    "params": [1, 0, 0, 0, 1, 0],
                    "physical": {"rotation_deg": 0., "scale_x": 1., "scale_y": 1.,
                                 "translate_x_mm": 0., "translate_y_mm": 0.,
                                 "pivot": [.5, .5]},
                }
            )
            if method == "spline":
                fit = fit_spline(spline)
                predicted = fit.forward(source_mm)
                written["spline"] = spline
            else:
                for points in (source_mm, target_mm):
                    if (not np.isfinite(points).all()
                            or len(np.unique(points, axis=0)) != len(points)
                            or np.linalg.matrix_rank(
                                np.column_stack([points, np.ones(len(points))])
                            ) < 3):
                        raise ValueError(
                            "Affine landmarks must be finite, distinct and non-collinear"
                        )
                matrix = affine_fit(source_mm, target_mm)
                determinant = float(np.linalg.det(matrix[:, :2]))
                if not np.isfinite(matrix).all() or determinant <= 1e-10:
                    raise ValueError("Landmark affine must be nonsingular and preserve orientation")
                # Convert from original physical millimetres to the normalized
                # section frame; preserves shear and avoids preview-size rounding.
                normalized = matrix.copy()
                normalized[:, :2] *= extent[None, :] / extent[:, None]
                normalized[:, 2] /= extent
                written["params"] = normalized.reshape(-1).tolist()
                written.pop("spline", None)
                canvas = canvas_geometry(
                    section.size, um, ctx.atlas, record.position_mm,
                    cast(Plane, state.plane), state.pitch_deg, state.yaw_deg,
                )
                pivot_mm = (
                    np.asarray(canvas.size) / 2 - np.asarray(canvas.section_offset)
                ) / np.asarray(section.size) * extent
                written["physical"] = {
                    **physical_params(
                        matrix, pivot=(float(pivot_mm[0]), float(pivot_mm[1])), um_per_px=1000.,
                    ),
                    "pivot": [.5, .5],
                }
                predicted = source_mm @ matrix[:, :2].T + matrix[:, 2]
            written.update(
                kind="interactive", note=str(note),
                calibration={"section_um_per_px": um, "source": source},
            )
            residuals = np.linalg.norm(predicted - target_mm, axis=1)
            common = dict(
                mode="overlay",
                border_color=border_color,
                border_thickness=border_thickness,
                template_opacity=opacity,
            )
            before_images, before_iou = render(
                record, section, um, previous, label=f"{record.id} before", **common
            )
            after_images, after_iou = render(
                record, section, um, written, label=f"{record.id} after", **common
            )
            written["iou"] = after_iou
            written["mirrored"] = False
            # Encode all visual feedback before checkpointing. The numbered raw
            # panels also make direct full-pair submissions visually inspectable.
            prospective = copy.deepcopy(record)
            prospective.transform = written
            prospective.landmark_pairs = canonical
            prospective.landmark_frame = geometry_signature(state, record)
            prospective.landmark_next_id = next_id
            references = build_view(prospective, view["zoom"], border_color, border_thickness)
            parts = [image_to_part(im) for im in before_images + after_images]
            parts.extend(references[TOOL_MEDIA_PARTS_KEY][:2])
        except (ValueError, TypeError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
            return {"status": "refused", "error": "INVALID_LANDMARK_WARP", "message": str(exc)}
        snapshot()
        record.transform = written
        record.landmark_pairs = canonical
        record.landmark_frame = geometry_signature(state, record)
        record.landmark_next_id = next_id
        commit(record.id)
        return {
            "status": "ok",
            "id": record.id,
            "written": True,
            "landmarks": len(canonical),
            "method": method,
            "view_id": references["view_id"],
            "existing_pairs": references["existing_pairs"],
            "images_info": [
                {"image_index": 0, "role": "before_overlay"},
                {"image_index": 1, "role": "after_overlay"},
                *[{**info, "image_index": i + 2}
                  for i, info in enumerate(references["images_info"][:2])],
            ],
            "landmark_residuals_mm": residuals.tolist(),
            "before_iou": before_iou,
            "after_iou": after_iou,
            "description": (
                "Image0 before; image1 after; image2 numbered unwarped slice; "
                "image3 numbered atlas. IoU is tissue silhouette overlap, "
                "not internal anatomical accuracy."
            ),
            "image_detail": "high",
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    return [view_landmarks, edit_landmarks, warp_landmarks]
