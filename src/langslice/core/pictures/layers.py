"""A picture's layers: what the model saw, and where every pixel sits in the atlas.

Every placement picture (a section on its physical canvas,
:func:`langslice.core.pictures.placement.draw_canvas`) is drawn in one frame:
the canvas, cropped and sized into the picture below its caption. Its layers
are on the SAME pixel grid as the picture (caption band included, empty there):

- **labels** — the atlas structure id under every pixel (uint32: the larger
  Allen ids exceed 16 bits), nearest neighbour through the mapping the
  borders are drawn with;
- **borders** — how much of each pixel the drawn atlas borders cover (uint8,
  0..255; the picture's own rasteriser,
  :func:`langslice.core.pictures.canvas.line_coverage`, at full strength);
- **frame** — a small JSON record: the plane (position, pitch, yaw), the
  micrometres per pixel, the placement, and ``pixel_to_atlas_um``, the 3x3
  matrix taking a picture pixel ``[row, col, 1]`` to BrainGlobe atlas
  micrometres (atlas axis order; voxel ``i``'s centre at ``i * resolution``;
  pixel centres at integers).

No dense coordinate map is stored per picture (three float32 channels per
pixel is the largest layer by far); :func:`coordinate_map` computes it from
the frame record on demand, exactly (the map is affine: a canvas pixel is a
point on one atlas plane).

Which pictures a tool call returned, and of what, is noted here while the call
runs (:func:`collecting`, :func:`note`), so the job can save them
(:class:`langslice.job.pictures.ViewStore`) without the doors changing what
they send. Core code notes the pictures it draws; with no collection running a
note is free.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tifffile

if TYPE_CHECKING:
    from langslice.core.pictures.canvas import PanelFrame

from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.fits.deformable.render import resampled_record
from langslice.core.jsonable import JsonObject
from langslice.core.pictures.drawn import PictureNote
from langslice.core.stack.json_reads import as_object, numbers_at, text_at

#: The frame record's format.
FRAME_FORMAT_VERSION = 1
#: Stated in every frame record, so a script reading one needs nothing else.
CONVENTION = (
    "pixel_to_atlas_um maps a picture pixel [row, col, 1] (pixel centres at "
    "integers, row 0 at the top, caption band included) to BrainGlobe atlas "
    "micrometres in the atlas's own axis order (voxel i's centre at i * "
    "resolution_um); the labels layer is the atlas annotation through the "
    "same map, nearest neighbour; section_to_picture maps the section "
    "render's pixel [row, col, 1] to the picture before any deformation."
)
#: Added to a deformable-fit picture's frame record.
WARP_CONVENTION = (
    " For a deformable-fit picture, residual says how the deformation record "
    "named by deformation was resized (grid_size) and cropped (box) to the "
    "picture; its field, (drow, dcol) in picture pixels, maps pixel [row, col] "
    "to the atlas point pixel_to_atlas_um @ [row + drow, col + dcol, 1] (no "
    "residual: the linear placement alone); langslice.coordinate_map applies it."
)


# --- the layers -----------------------------------------------------------------------


def _native_rows_cols(panel: PanelFrame) -> tuple[np.ndarray, np.ndarray]:
    """Native plane row and column of each picture row and column (floats)."""
    x0, y0, x1, y1 = panel.content_box
    bx, by = panel.crop_box[0], panel.crop_box[1]
    scale = panel.geometry.atlas_scale
    ox, oy = panel.geometry.atlas_offset
    rows = ((np.arange(y0, y1, dtype=np.float64) - y0) / panel.factor + by - oy) / scale
    cols = ((np.arange(x0, x1, dtype=np.float64) - x0) / panel.factor + bx - ox) / scale
    return rows, cols


def labels_layer(panel: PanelFrame) -> np.ndarray:
    """The atlas id under every picture pixel, uint32, the picture's size."""
    width, height = panel.size
    out = np.zeros((height, width), dtype=np.uint32)
    annotation = np.asarray(panel.geometry.annotation)
    rows, cols = _native_rows_cols(panel)
    r = np.rint(rows).astype(np.int64)
    c = np.rint(cols).astype(np.int64)
    keep_r = (r >= 0) & (r < annotation.shape[0])
    keep_c = (c >= 0) & (c < annotation.shape[1])
    x0, y0, _x1, _y1 = panel.content_box
    block = annotation[np.ix_(r[keep_r], c[keep_c])].astype(np.uint32)
    target = out[y0:y0 + len(r), x0:x0 + len(c)]
    target[np.ix_(np.nonzero(keep_r)[0], np.nonzero(keep_c)[0])] = block
    return out


def borders_layer(panel: PanelFrame) -> np.ndarray:
    """The drawn atlas borders' coverage of every picture pixel, uint8."""
    from langslice.core.atlas.borders import HIGHLIGHT_WIDTH
    from langslice.core.pictures.canvas import line_coverage

    width, height = panel.size
    out = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = panel.content_box
    shape = (y1 - y0, x1 - x0)
    kwargs = {"thickness": panel.line_width, "scale": panel.geometry.atlas_scale,
              "offset": panel.geometry.atlas_offset, "origin": panel.crop_box[:2],
              "factor": panel.factor}
    coverage = np.zeros(shape, dtype=np.float32)
    for polys, width in ((panel.lines, 1.0), (panel.highlighted, HIGHLIGHT_WIDTH)):
        if polys:
            coverage = np.maximum(coverage, line_coverage(
                shape, list(polys), **{**kwargs, "thickness": panel.line_width * width}))
    out[y0:y1, x0:x1] = np.rint(coverage * 255.0).astype(np.uint8)
    return out


def picture_to_native(panel: PanelFrame) -> np.ndarray:
    """3x3: picture pixel ``[row, col, 1]`` -> native plane ``[row, col, 1]``."""
    x0, y0, _x1, _y1 = panel.content_box
    bx, by = panel.crop_box[0], panel.crop_box[1]
    scale = panel.geometry.atlas_scale
    ox, oy = panel.geometry.atlas_offset
    step = 1.0 / (panel.factor * scale)
    return np.array([
        [step, 0.0, (-y0 / panel.factor + by - oy) / scale],
        [0.0, step, (-x0 / panel.factor + bx - ox) / scale],
        [0.0, 0.0, 1.0],
    ])


def section_to_picture(panel: PanelFrame) -> np.ndarray:
    """3x3: section render pixel ``[row, col, 1]`` -> picture ``[row, col, 1]``
    (the linear placement; a deformation drawn on top is not in it)."""
    x0, y0, _x1, _y1 = panel.content_box
    bx, by = panel.crop_box[0], panel.crop_box[1]
    canvas_to_picture = np.array([
        [panel.factor, 0.0, x0 - bx * panel.factor],
        [0.0, panel.factor, y0 - by * panel.factor],
        [0.0, 0.0, 1.0],
    ])
    xy = canvas_to_picture @ np.asarray(panel.section_matrix, dtype=np.float64)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return swap @ xy @ swap  # x/y -> row/col on both sides


def atlas_facts(atlas: Any) -> dict[str, Any]:
    """The atlas as a frame record names it."""
    from langslice.core.space import atlas_space_context

    context = atlas_space_context(atlas)
    metadata = getattr(atlas, "metadata", None)
    version = metadata.get("version") if isinstance(metadata, dict) else None
    return {
        "name": context.atlas_name,
        "version": None if version is None else str(version),
        "orientation": context.orientation,
        "shape": [int(v) for v in context.shape],
        "resolution_um": [float(v) for v in context.resolution_um],
    }


def frame_record(atlas: Any, note: PictureNote) -> dict[str, Any]:
    """The frame of a placement picture as plain JSON (see the module text)."""
    from langslice.core.geometry.oblique import plane_index_affine

    frame, panel = note.frame, note.panel
    if frame is None or panel is None:
        raise ValueError("frame_record needs a placement picture's frame and panel")
    facts = atlas_facts(atlas)
    index = plane_index_affine(atlas, frame.position_mm, frame.plane,  # type: ignore[arg-type]
                               frame.pitch_deg, frame.yaw_deg)
    to_um = np.diag(facts["resolution_um"]) @ index @ picture_to_native(panel)
    params = frame.params
    placement: dict[str, Any] = (
        {"matrix": np.asarray(params, dtype=np.float64).tolist()}
        if isinstance(params, np.ndarray)
        else {"knobs": {key: float(value) for key, value in params.items()}}
    )
    if frame.pivot is not None:
        placement["pivot_canvas"] = [float(v) for v in frame.pivot]
    if frame.pivot_in_section is not None:
        placement["pivot_section"] = [float(v) for v in frame.pivot_in_section]
    return {
        "format_version": FRAME_FORMAT_VERSION,
        "atlas": facts,
        "plane": {"name": frame.plane, "position_mm": float(frame.position_mm),
                  "pitch_deg": float(frame.pitch_deg), "yaw_deg": float(frame.yaw_deg)},
        "picture_size": [int(v) for v in panel.size],
        "content_box": [int(v) for v in panel.content_box],
        "um_per_px": {"canvas": float(frame.um_per_px),
                      "picture": float(frame.um_per_px) / panel.factor},
        "canvas": {"size": [int(v) for v in panel.geometry.size],
                   "crop_box": [int(v) for v in panel.crop_box],
                   "factor": float(panel.factor),
                   "atlas_scale": float(panel.geometry.atlas_scale),
                   "atlas_offset": [float(v) for v in panel.geometry.atlas_offset],
                   "section_offset": [int(v) for v in panel.geometry.section_offset]},
        "section": {"id": frame.section_id, "render_size": [int(v) for v in frame.section.size],
                    "placement": placement},
        "deformation": note.deformation,
        "zoom": None if frame.zoom is None else [float(v) for v in frame.zoom],
        "pixel_to_atlas_um": to_um.tolist(),
        "section_to_picture": section_to_picture(panel).tolist(),
        "convention": CONVENTION,
    }


def picture_layers(atlas: Any, note: PictureNote) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(labels, borders, frame record)`` of one placement picture."""
    if note.panel is None:
        raise ValueError("picture_layers needs a placement picture's panel")
    return labels_layer(note.panel), borders_layer(note.panel), frame_record(atlas, note)


def warp_layers(atlas: Any, note: PictureNote, picture_size: tuple[int, int]) -> tuple[
        np.ndarray, np.ndarray, dict[str, Any]]:
    """``(labels, borders, frame record)`` of a ``fit_deformable`` picture
    (a :class:`WarpNote`), on the picture's own pixel grid.

    The frame's ``pixel_to_atlas_um`` is the record's linear placement on
    the picture; its ``residual`` (None when the picture shows the linear
    placement alone) names how the record (the frame's ``deformation``) was
    resized and cropped to the picture, so :func:`coordinate_map` draws the
    residual again from the record: a pixel ``[r, c]`` shows atlas point
    ``pixel_to_atlas_um @ [r + drow, c + dcol, 1]``. Labels are the
    native atlas plane's ids under that map (nearest) at every content pixel,
    as a placement picture's labels: no tissue rule cuts them (the drawn
    lines stop at the tissue; the labels do not); borders the drawn lines'
    coverage at full strength.
    """
    from langslice.core.fits.deformable.geometry import sample_native
    from langslice.core.fits.deformable.render import drawn_border_coverage
    from langslice.core.geometry.oblique import plane_index_affine

    warp = note.warp
    if warp is None:
        raise ValueError("warp_layers needs a deformable-fit picture's note")
    record = warp.record
    placement = record.placement
    width, height = (int(v) for v in picture_size)
    rows, cols = record.field_mm.shape[:2]
    band = int(warp.band)
    facts = atlas_facts(atlas)
    index = plane_index_affine(atlas, placement.position_mm, placement.plane,
                               placement.pitch_deg, placement.yaw_deg)
    # picture [row, col, 1] -> record grid [x, y, 1] -> native [x, y, 1] -> [row, col, 1]
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    down = np.array([[1.0, 0.0, -band], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    picture_to_grid = swap @ down  # [row, col, 1] -> [x, y, 1] on the record's grid
    grid_to_native = np.linalg.inv(np.asarray(placement.atlas_to_section, dtype=np.float64))
    to_um = (np.diag(facts["resolution_um"]) @ index @ swap @ grid_to_native
             @ picture_to_grid)
    yy, xx = np.indices((rows, cols), dtype=np.float64)
    shift_x = shift_y = np.zeros((rows, cols))
    if warp.warped:
        shift_x = record.field_mm[..., 0].astype(np.float64) / record.mm_per_px
        shift_y = record.field_mm[..., 1].astype(np.float64) / record.mm_per_px
    gx, gy = xx + shift_x, yy + shift_y
    native_x = grid_to_native[0, 0] * gx + grid_to_native[0, 1] * gy + grid_to_native[0, 2]
    native_y = grid_to_native[1, 0] * gx + grid_to_native[1, 1] * gy + grid_to_native[1, 2]
    native = warp.native if warp.native is not None else None
    if native is None:
        from langslice.core.fits.deformable.plane_images import native_labels

        native = native_labels(atlas, placement)
    ids = sample_native(np.asarray(native), np.stack([native_x, native_y], axis=-1))
    labels = np.zeros((height, width), dtype=np.uint32)
    labels[band:band + rows, :cols] = ids.astype(np.uint32)
    coverage = drawn_border_coverage(
        record, atlas, highlight=warp.highlight, marked=warp.marked, warped=warp.warped,
        outlines=warp.outlines, depth=warp.depth, width_px=warp.width_px, native=native)
    borders = np.zeros((height, width), dtype=np.uint8)
    borders[band:band + rows, :cols] = np.rint(np.clip(coverage, 0.0, 1.0) * 255.0)
    residual = None
    if warp.warped:
        residual = {"grid_size": list(warp.grid_size), "box": list(warp.box)}
    frame = {
        "format_version": FRAME_FORMAT_VERSION,
        "atlas": facts,
        "plane": {"name": placement.plane, "position_mm": float(placement.position_mm),
                  "pitch_deg": float(placement.pitch_deg),
                  "yaw_deg": float(placement.yaw_deg)},
        "picture_size": [width, height],
        "content_box": [0, band, cols, band + rows],
        "um_per_px": {"picture": float(record.mm_per_px) * 1000.0},
        "section": {"id": note.sections[0] if note.sections else None,
                    "grid": "the section's oriented, unframed render the fit ran on, "
                            "resized and cropped to the picture"},
        "deformation": note.deformation,
        "warped": bool(warp.warped),
        "residual": residual,
        "pixel_to_atlas_um": to_um.tolist(),
        "convention": CONVENTION + WARP_CONVENTION,
    }
    return labels, borders, frame


# --- on demand: the coordinate map ------------------------------------------------------


def coordinate_map(
    view: dict[str, Any] | str | os.PathLike[str], *,
    folder: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Every picture pixel's atlas position, ``(rows, cols, 3)`` float32 µm.

    *view* is a saved picture's ``view.json`` (its record or its path). The
    three channels are the atlas axes in the atlas's own order, BrainGlobe
    micrometres (voxel ``i``'s centre at ``i * resolution``); the caption
    band, which shows no canvas, is NaN. A placement picture shows one atlas
    plane, so the map is ``pixel_to_atlas_um`` applied to every pixel
    centre, exactly. A deformable-fit picture adds its residual: its
    deformation record (the frame's ``deformation``, in the job folder above
    the ``view.json``, or above *folder* when *view* is a record) resized and
    cropped as the frame's ``residual`` says, ``pixel_to_atlas_um`` at each
    pixel plus its displacement. (A picture saved with a ``residual.tif``
    layer beside it, which the frame's ``residual`` names, reads that.)
    """
    base: str | None = None
    if isinstance(view, dict):
        record: dict[str, Any] = view
    else:
        import json

        with open(os.fspath(view), encoding="utf-8") as handle:
            record = json.load(handle)
        base = os.path.dirname(os.path.abspath(os.fspath(view)))
    if folder is not None:
        base = os.fspath(folder)
    frame = record.get("frame", record)
    if not isinstance(frame, dict) or "pixel_to_atlas_um" not in frame:
        raise ValueError("This picture has no atlas frame (not a placement picture)")
    matrix = np.asarray(frame["pixel_to_atlas_um"], dtype=np.float64)
    width, height = (int(v) for v in frame["picture_size"])
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    if frame.get("residual"):
        if base is None:
            raise ValueError("This picture's map needs its deformation record: pass the "
                             "view.json path, or folder= with the record")
        shift = _residual_shift(cast("JsonObject", frame), base)  # view.json is JSON
        rows = rows + shift[0]
        cols = cols + shift[1]
    out = (np.tensordot(matrix[:, :2], np.stack([rows, cols]), axes=1)
           + matrix[:, 2][:, None, None])
    result = np.moveaxis(out, 0, -1).astype(np.float32)
    x0, y0, x1, y1 = (int(v) for v in frame["content_box"])
    outside = np.ones((height, width), dtype=bool)
    outside[y0:y1, x0:x1] = False
    result[outside] = np.nan
    return result


def _residual_shift(frame: JsonObject, base: str) -> np.ndarray:
    """Return a deformable-fit picture's residual as it was drawn with it.

    ``(2, rows, cols)`` ``(drow, dcol)`` in picture pixels, zero outside the
    content: the frame's record resampled as its ``residual`` says, or the
    ``residual.tif`` beside a picture saved with one.
    """
    residual = frame["residual"]
    if isinstance(residual, str):
        layer = tifffile.imread(Path(base) / residual)
        return np.asarray(layer, dtype=np.float64)
    resample = as_object("residual", residual)
    folder = _record_folder(Path(base), text_at(frame, "deformation"))
    record = DeformableRecord.load(folder)
    width, height = (int(v) for v in numbers_at(resample, "grid_size"))
    x0, y0, x1, y1 = (int(v) for v in numbers_at(resample, "box"))
    resampled = resampled_record(record, (width, height), (x0, y0, x1, y1))
    field = resampled.field_mm.astype(np.float64) / resampled.mm_per_px
    columns, rows = (int(v) for v in numbers_at(frame, "picture_size"))
    content = numbers_at(frame, "content_box")
    left, top, right, bottom = (int(v) for v in content)
    shift = np.zeros((2, rows, columns), dtype=np.float32)
    shift[0, top:bottom, left:right] = field[..., 1]
    shift[1, top:bottom, left:right] = field[..., 0]
    return shift.astype(np.float64)


def _record_folder(base: Path, relative: str) -> Path:
    """Return the folder of the record a job-relative path names, in the
    job folder above *base* (a picture's folder).
    """
    for folder in (base.resolve(), *base.resolve().parents):
        if (folder / relative / "record.json").exists():
            return folder / relative
    raise ValueError(
        f"The deformation record {relative} this picture was drawn with is "
        f"not in a job folder above {base}"
    )
