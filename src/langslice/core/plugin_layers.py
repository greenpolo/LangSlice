"""The layers tool plugins leave on sections: their files, and how ``look`` draws them.

A layer (:class:`langslice.core.plugins.Layer`) is stored as one file with a
fixed format per kind (``docs/reference/file_formats.md``):

=========  ============================================  =====================
kind       file                                          content
=========  ============================================  =====================
points     ``.csv``                                      ``x,y[,z],label,value``
lines      ``.json``                                     GeoJSON-like features
labels     ``.png`` (16-bit), ``.npz`` past 65535 ids    integer raster
scalar     ``.npz`` (``values``, ``level``)              float raster
picture    ``.png``                                      a finished picture
=========  ============================================  =====================

The job layer decides where the files go and keeps the manifest; this module
writes and reads one file, and draws layers on the pictures ``look`` makes of
a section (mode ``section`` or ``overlay``), over the panel geometry the
pictures already carry for the landmark markers: a layer in ``section_px`` is
placed through the section's panel (with the applied deformation when the
overlay draws it), a layer in ``atlas_um`` through the atlas plane of the
overlay or, in a section picture, the section's linear placement. Layers are
drawn as outlines and marks with a dark halo, never as translucent fills.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from langslice.core.landmarks import transform
from langslice.core.plugins import Layer, ToolError
from langslice.core.state import SliceState, StackState

logger = logging.getLogger(__name__)

#: Marks a layer's colours cycle through (a dark halo is drawn under each).
PALETTE: tuple[str, ...] = ("#ffa000", "#00e5a0", "#ff4081", "#40c4ff", "#ffee00", "#b388ff")


# --- files -------------------------------------------------------------------------------


def save_layer(layer: Layer, stem: Path) -> Path:
    """Write *layer* to ``<stem><extension>`` and return the path."""
    layer.check()
    stem.parent.mkdir(parents=True, exist_ok=True)
    kind = layer.kind
    if kind == "points":
        path = stem.with_suffix(".csv")
        assert layer.xy is not None
        header = ["x", "y"] if layer.frame == "section_px" else ["x_um", "y_um", "z_um"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*header, "label", "value"])
            for index, row in enumerate(layer.xy):
                writer.writerow([*(f"{float(v):.6g}" for v in row),
                                 layer.tags[index] if layer.tags else "",
                                 f"{layer.values[index]:.6g}" if layer.values else ""])
        return path
    if kind == "lines":
        path = stem.with_suffix(".json")
        features = []
        for index, (coordinates, closed) in enumerate(layer.paths):
            points = [[float(v) for v in row] for row in coordinates]
            if closed:
                features.append({"type": "Polygon", "coordinates": [points + [points[0]]]})
            else:
                features.append({"type": "LineString", "coordinates": points})
            if layer.tags:
                features[-1]["properties"] = {"label": layer.tags[index]}
        path.write_text(json.dumps({"kind": "lines", "frame": layer.frame,
                                    "features": features}), encoding="utf-8")
        return path
    if kind == "labels":
        assert layer.raster is not None
        if int(layer.raster.max(initial=0)) < 65536:
            path = stem.with_suffix(".png")
            Image.fromarray(layer.raster.astype(np.uint16)).save(path)
        else:
            path = stem.with_suffix(".npz")
            np.savez_compressed(path, labels=layer.raster.astype(np.uint32))
        return path
    if kind == "scalar":
        assert layer.raster is not None
        path = stem.with_suffix(".npz")
        np.savez_compressed(path, values=layer.raster.astype(np.float32),
                            level=np.float32(layer.level))
        return path
    path = stem.with_suffix(".png")
    image = layer.image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image.save(path)
    return path


def load_layer(path: Path, ref: Mapping[str, Any]) -> Layer:
    """The layer the file *path* holds; *ref* is its reference (``kind``,
    ``frame``, ``name``)."""
    kind, frame, name = str(ref["kind"]), str(ref["frame"]), str(ref.get("name", ""))
    if kind == "points":
        rows = list(csv.reader(path.open(encoding="utf-8")))
        body = rows[1:]
        width = 2 if frame == "section_px" else 3
        points = np.array([[float(v) for v in row[:width]] for row in body],
                          dtype=np.float64).reshape(-1, width)
        labels = tuple(row[width] for row in body) if any(row[width] for row in body) else ()
        values = (tuple(float(row[width + 1]) for row in body)
                  if body and all(row[width + 1] for row in body) else ())
        return Layer("points", frame, name=name, xy=points, tags=labels, values=values)
    if kind == "lines":
        held = json.loads(path.read_text(encoding="utf-8"))
        paths: list[tuple[np.ndarray, bool]] = []
        tags: list[str] = []
        for feature in held["features"]:
            closed = feature["type"] == "Polygon"
            coordinates = feature["coordinates"][0][:-1] if closed else feature["coordinates"]
            paths.append((np.asarray(coordinates, dtype=np.float64), closed))
            tags.append(str((feature.get("properties") or {}).get("label", "")))
        return Layer("lines", frame, name=name, paths=tuple(paths),
                     tags=tuple(tags) if any(tags) else ())
    if kind == "labels":
        if path.suffix == ".npz":
            with np.load(path) as held:
                raster = np.asarray(held["labels"]).astype(np.int64)
        else:
            raster = np.asarray(Image.open(path)).astype(np.int64)
        return Layer("labels", frame, name=name, raster=raster)
    if kind == "scalar":
        with np.load(path) as held:
            return Layer("scalar", frame, name=name, raster=np.asarray(held["values"]),
                         level=float(held["level"]))
    return Layer("picture", frame, name=name, image=Image.open(path).copy())


def layer_path(folder: Path | str, ref: Mapping[str, Any]) -> Path:
    """Where a layer reference's file is (``file`` is relative to the job folder)."""
    return Path(folder) / str(ref["file"])


def visible(record: SliceState, wanted: Sequence[str]) -> list[dict[str, Any]]:
    """The references of *record*'s layers named by *wanted* (``"all"``: every
    one), in the order the section holds them."""
    held = list(record.layers)
    if "all" in wanted:
        return held
    names = set(wanted)
    return [ref for ref in held if ref.get("name") in names]


def known_names(state: StackState, sections: Sequence[str] = ()) -> list[str]:
    """Every layer name on *sections* (all sections when empty), first seen first."""
    names: list[str] = []
    for record in state.slices:
        if sections and record.id not in sections:
            continue
        for ref in record.layers:
            if ref["name"] not in names:
                names.append(str(ref["name"]))
    return names


# --- geometry -------------------------------------------------------------------------------


def _colour(index: int) -> tuple[int, int, int]:
    value = PALETTE[index % len(PALETTE)].lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def fractions_to_picture(panel: Mapping[str, Any], fractions: np.ndarray,
                         warp: Any = None) -> np.ndarray:
    """Section-file edge fractions ``(N, 2)`` to picture pixels ``(N, 2)``
    through a section *panel* (the inverse of
    :func:`langslice.core.landmarks.section_point`, for many points)."""
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1, 2)
    if "pixel_to_file_fraction" in panel:
        return transform(np.linalg.inv(panel["pixel_to_file_fraction"]), fractions)
    oriented = transform(np.linalg.inv(np.asarray(panel["orientation"])), fractions)
    size = np.asarray(panel["render_size"], dtype=np.float64)
    point = oriented * size - 0.5
    if warp is not None:
        from scipy.ndimage import map_coordinates

        scaled = (point + 0.5) * np.asarray(warp.section_size) / size - 0.5
        shift = np.stack([map_coordinates(warp.field_mm[..., k], [scaled[:, 1], scaled[:, 0]],
                                          order=1, mode="nearest") for k in range(2)], axis=1)
        point = point + shift / warp.mm_per_px * size / np.asarray(warp.section_size)
    return transform(np.linalg.inv(np.asarray(panel["pixel_to_render"])), point)


def picture_to_fractions(panel: Mapping[str, Any], xy: np.ndarray,
                         warp: Any = None) -> np.ndarray:
    """Picture pixels ``(N, 2)`` to section-file edge fractions ``(N, 2)``
    (:func:`langslice.core.landmarks.section_point`, for many points)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if "pixel_to_file_fraction" in panel:
        return transform(np.asarray(panel["pixel_to_file_fraction"]), xy)
    point = transform(np.asarray(panel["pixel_to_render"]), xy)
    width, height = panel["render_size"]
    if warp is not None:
        from langslice.core.deformable.render import warp_source_points

        # cv2.remap takes maps under 32767 on a side: lay the points in rows of 1024.
        count = len(point)
        padded = np.zeros((-(-count // 1024) * 1024, 2))
        padded[:count] = point
        x, y = warp_source_points(warp, (width, height), padded[:, 0].reshape(-1, 1024),
                                  padded[:, 1].reshape(-1, 1024))
        point = np.stack([x.ravel()[:count], y.ravel()[:count]], axis=1)
    return transform(np.asarray(panel["orientation"]),
                     (point + 0.5) / np.array([width, height], dtype=np.float64))


class _Mapper:
    """Places one section's layers on one picture: file fractions and atlas
    micrometres to picture pixels."""

    def __init__(self, state: StackState, record: SliceState, panels: Sequence[Mapping[str, Any]],
                 ws: Any, warp: Any) -> None:
        self.state, self.record, self.ws, self.warp = state, record, ws, warp
        self.section = next((p for p in panels if p["role"] == "section"), None)
        self.atlas = next((p for p in panels if p["role"] == "atlas"), None)
        self._file_size: tuple[int, int] | None = None
        self._frame: Any = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        panel = self.section or self.atlas
        assert panel is not None
        x0, y0, x1, y1 = panel["bounds"]
        return float(x0), float(y0), float(x1), float(y1)

    def file_size(self) -> tuple[int, int]:
        if self._file_size is None:
            from langslice.core.maps import render_sizes

            self._file_size = render_sizes(self.ws, self.record.id).file_size
        return self._file_size

    def frame(self) -> Any:
        if self._frame is None:
            from langslice.core.maps import section_frame

            self._frame = section_frame(self.state, self.ws, self.record)
        return self._frame

    def section_warp(self) -> Any:
        return self.warp if self.section is not None and self.section.get("warp_key") else None

    def from_section_px(self, xy: np.ndarray) -> np.ndarray:
        if self.section is None:
            raise ToolError("LAYER_NOT_DRAWABLE", "this picture has no section to place a "
                            "section layer on")
        width, height = self.file_size()
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array([width, height])
        return fractions_to_picture(self.section, fractions, self.section_warp())

    def from_atlas_um(self, xyz: np.ndarray) -> np.ndarray:
        frame = self.frame()
        xyz = np.asarray(xyz, dtype=np.float64)
        if self.atlas is not None:
            matrix = np.asarray(frame.native_to_um, dtype=np.float64)
            linear, offset = matrix[:, :2], matrix[:, 2]
            rowcol = np.linalg.lstsq(linear, (xyz - offset).T, rcond=None)[0].T
            native_xy = rowcol[:, ::-1]
            return transform(np.linalg.inv(np.asarray(self.atlas["pixel_to_native"])), native_xy)
        if self.section is None:
            raise ToolError("LAYER_NOT_DRAWABLE", "this picture shows no plane to place an "
                            "atlas layer on")
        width, height = frame.file_size
        matrix = np.asarray(frame.pixel_to_atlas_um(), dtype=np.float64)
        linear, offset = matrix[:, :2], matrix[:, 2]
        rowcol = np.linalg.lstsq(linear, (xyz - offset).T, rcond=None)[0].T
        fractions = (rowcol[:, ::-1] + 0.5) / np.array([width, height])
        return fractions_to_picture(self.section, fractions, None)

    def to_picture(self, frame: str, coords: np.ndarray) -> np.ndarray:
        return self.from_section_px(coords) if frame == "section_px" \
            else self.from_atlas_um(coords)

    def sample_raster(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """For every picture pixel in the section panel's bounds: its row and
        column in a raster of *shape* covering the section file, and whether
        it lies on the section (``inside``). ``(rows, cols, inside)`` stacked as
        two arrays of the bounds' shape plus the mask."""
        if self.section is None:
            raise ToolError("LAYER_NOT_DRAWABLE", "this picture has no section to place a "
                            "raster layer on")
        x0, y0, x1, y1 = self.bounds
        xs, ys = np.meshgrid(np.arange(int(x0), int(x1)), np.arange(int(y0), int(y1)))
        fractions = picture_to_fractions(self.section, np.stack([xs.ravel(), ys.ravel()], 1),
                                         self.section_warp())
        rows = np.rint(fractions[:, 1] * shape[0] - 0.5).astype(int)
        cols = np.rint(fractions[:, 0] * shape[1] - 0.5).astype(int)
        inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        index = np.stack([np.clip(rows, 0, shape[0] - 1), np.clip(cols, 0, shape[1] - 1)])
        return index.reshape(2, *xs.shape), inside.reshape(xs.shape)


# --- drawing --------------------------------------------------------------------------------


def _halo_line(draw: ImageDraw.ImageDraw, path: list[tuple[float, float]],
               colour: tuple[int, int, int]) -> None:
    draw.line(path, fill=(0, 0, 0), width=5)
    draw.line(path, fill=colour, width=3)


def _draw_points(image: Image.Image, mapper: _Mapper, layer: Layer, colour: tuple[int, int, int]
                 ) -> int:
    assert layer.xy is not None
    xy = mapper.to_picture(layer.frame, layer.xy)
    x0, y0, x1, y1 = mapper.bounds
    draw = ImageDraw.Draw(image)
    shown = 0
    for index, (x, y) in enumerate(xy):
        if not (x0 <= x < x1 and y0 <= y < y1):
            continue
        shown += 1
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(0, 0, 0))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
        text = layer.tags[index] if layer.tags else ""
        if text:
            box = draw.textbbox((x + 9, y - 7), text)
            draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill="black")
            draw.text((x + 9, y - 7), text, fill=colour)
    return shown


def _draw_lines(image: Image.Image, mapper: _Mapper, layer: Layer, colour: tuple[int, int, int]
                ) -> int:
    draw = ImageDraw.Draw(image)
    for index, (coordinates, closed) in enumerate(layer.paths):
        xy = mapper.to_picture(layer.frame, coordinates)
        path = [(float(x), float(y)) for x, y in xy]
        if closed:
            path.append(path[0])
        _halo_line(draw, path, colour)
        if layer.tags and layer.tags[index]:
            draw.text((path[0][0] + 5, path[0][1] + 5), layer.tags[index], fill=colour)
    return len(layer.paths)


def _outline(mask: np.ndarray) -> np.ndarray:
    """The pixels of *mask* that touch a pixel outside it (4-neighbourhood)."""
    padded = np.pad(mask, 1, constant_values=False)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~inner


def _paint(image: Image.Image, mapper: _Mapper, boundary: np.ndarray,
           colour: np.ndarray) -> None:
    """Paint *boundary* (a mask over the panel bounds) in *colour* (a colour,
    or one per pixel) with a one-pixel dark halo."""
    x0, y0 = int(mapper.bounds[0]), int(mapper.bounds[1])
    height, width = boundary.shape
    halo = np.pad(boundary, 1, constant_values=False)
    thick = np.zeros_like(boundary)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            thick |= halo[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
    array = np.asarray(image.convert("RGB")).copy()
    window = array[y0:y0 + height, x0:x0 + width]
    window[thick] = (0, 0, 0)
    window[boundary] = colour if colour.ndim == 1 else colour[boundary]
    image.paste(Image.fromarray(array))


def _draw_labels(image: Image.Image, mapper: _Mapper, layer: Layer, index: int) -> int:
    assert layer.raster is not None
    (rows, cols), inside = mapper.sample_raster(layer.raster.shape)
    ids = np.where(inside, layer.raster[rows, cols], 0)
    padded = np.pad(ids, 1, constant_values=0)
    edge = ((padded[:-2, 1:-1] != ids) | (padded[2:, 1:-1] != ids)
            | (padded[1:-1, :-2] != ids) | (padded[1:-1, 2:] != ids))
    boundary = edge & (ids > 0)
    colours = np.array([_colour(index + int(item)) for item in range(len(PALETTE))])
    paint = colours[(ids % len(PALETTE)).astype(int)]
    _paint(image, mapper, boundary, paint)
    return int(len(np.unique(ids[ids > 0])))


def _draw_scalar(image: Image.Image, mapper: _Mapper, layer: Layer,
                 colour: tuple[int, int, int]) -> int:
    assert layer.raster is not None
    (rows, cols), inside = mapper.sample_raster(layer.raster.shape)
    mask = inside & (layer.raster[rows, cols] >= layer.level)
    _paint(image, mapper, _outline(mask), np.array(colour, dtype=np.uint8))
    return int(mask.any())


def draw_layers(
    image: Image.Image, panels: Sequence[Mapping[str, Any]], state: StackState,
    record: SliceState, wanted: Sequence[str], *, ws: Any, warp: Any = None,
) -> list[dict[str, Any]]:
    """Draw the layers of *record* that *wanted* names (``"all"``: every one)
    on *image* in place, over the panels of the picture. Returns one note per
    layer asked for: ``{"name", "tool", "kind", "drawn"}`` and, for a layer
    that could not be drawn, ``"reason"``. A layer is never drawn
    translucently: points and lines with a dark halo, rasters as outlines."""
    folder = getattr(ws, "job_folder", None)
    notes: list[dict[str, Any]] = []
    refs = visible(record, wanted)
    if not refs:
        return notes
    mapper = _Mapper(state, record, panels, ws, warp)
    for index, ref in enumerate(refs):
        note: dict[str, Any] = {"name": ref["name"], "tool": ref.get("tool", ""),
                                "kind": ref["kind"]}
        notes.append(note)
        if ref["kind"] == "picture":
            note.update(drawn=False, reason="a picture layer is a finished picture; it is "
                        "not drawn on other pictures")
            continue
        if folder is None:
            note.update(drawn=False, reason="the job folder is unknown here")
            continue
        try:
            layer = load_layer(layer_path(folder, ref), ref)
            colour = _colour(index)
            if layer.kind == "points":
                shown = _draw_points(image, mapper, layer, colour)
            elif layer.kind == "lines":
                shown = _draw_lines(image, mapper, layer, colour)
            elif layer.kind == "labels":
                shown = _draw_labels(image, mapper, layer, index)
            else:
                shown = _draw_scalar(image, mapper, layer, colour)
            note.update(drawn=True, shown=shown)
        except ToolError as exc:
            note.update(drawn=False, reason=exc.message)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("layer %s of %s could not be drawn", ref.get("name"), record.id,
                           exc_info=True)
            note.update(drawn=False, reason=f"the layer could not be read or placed ({exc})")
    return notes
