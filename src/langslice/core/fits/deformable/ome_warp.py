"""A deformable record's arrays as an OME-Zarr 0.6 scene (``warp.ome.zarr``).

The record's folder holds ``record.json`` (its metadata, written last) and
this Zarr v3 directory store. The store is self-describing for other
programs through the OME-NGFF 0.6 coordinate transformations (RFC-5):

- the root group's ``ome.scene`` names three coordinate systems, ``section``
  (the fit grid's pixel centres times its pixel size, ``(y, x)`` in mm),
  ``placed_atlas`` (the linearly placed atlas plane drawn on that grid, mm)
  and ``atlas`` (the BrainGlobe volume in micrometres, its own axis order,
  voxel ``i``'s centre at ``i * resolution``), and the transformations
  between them: a ``bijection`` ``section`` -> ``placed_atlas`` whose forward
  is the ``displacements`` of ``field`` and whose inverse is the
  ``displacements`` of ``inverse_field`` (a record without an inverse has
  the forward ``displacements`` alone), then a ``sequence`` ``placed_atlas``
  -> ``atlas``: a ``scale`` from mm to placed pixels, an ``affine`` from
  placed pixels to native atlas-plane pixels (the inverse of the
  placement's ``atlas_to_section``), an ``affine`` from the native plane to
  volume indices (``atlas["native_to_volume_index"]``) and a ``scale`` by
  the atlas resolution; and an ``identity`` from each image group's own
  ``section`` system to the scene's;
- the fields as ``record.json``'s ``fields`` says (the ``displacement``
  axis first, as the spec orders it, ``(dy, dx)`` in mm; the record holds
  ``(rows, cols, 2)`` ``(dx, dy)`` float32):

  - ``engine_grid``: ``coordinateTransformations/step_field`` and
    ``.../step_inverse_field``, the engine's own fields of this step on its
    working grid, float32;
  - ``landmark_mesh``: none (the mesh is in ``record.json``);
  - ``section_grid`` (a map made elsewhere, a restricted fit, and every
    format-2 record): ``coordinateTransformations/field`` and
    ``.../inverse_field`` on the section grid, float16;

  each a multiscale group of one level ``s0``;
- ``warped_labels`` (the leaf ids under the composed warp), ``footprint``
  ``left_out`` and ``weights`` (the engine's own P(match) sorting of the
  section, float16; absent when the engine estimated nothing): multiscale
  images of one level on the section grid, stored losslessly.

Every array is one chunk, Blosc zstd with byte shuffle. A record's fields
are kept at float16 precision (rounding of at most 0.25 µm on a
displacement under 0.5 mm, 0.5 µm under 1 mm) and read back as float32;
:func:`at_storage_precision` rounds a field the same way, so a record kept
in memory equals the one read back. The scene's deformation is the stored
fields when they are the whole map (fields stored whole, or an engine
step on the linear placement); a step to compose with its parent and a
mesh are left out of it. The scene metadata is derived from the
record for other readers; LangSlice reads only the arrays (the placement
comes from ``record.json``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.fits.deformable.fields import EngineStep
from langslice.core.fits.deformable.geometry import WorkingGrid

if TYPE_CHECKING:
    from langslice.core.fits.deformable.record import DeformableRecord

#: The store's folder name inside a record's folder.
WARP_STORE = "warp.ome.zarr"
OME_VERSION = "0.6"
#: How the section-grid displacement fields are stored (read back as float32).
FIELD_DTYPE = np.float16
#: How an engine's own working-grid fields are stored: as it produced them.
STEP_DTYPE = np.float32
FIELD_GROUP = "coordinateTransformations/field"
INVERSE_GROUP = "coordinateTransformations/inverse_field"
STEP_GROUP = "coordinateTransformations/step_field"
STEP_INVERSE_GROUP = "coordinateTransformations/step_inverse_field"
#: The fields :func:`read` returns, by the group that holds each.
FIELD_GROUPS = {"field_mm": FIELD_GROUP, "inverse_field_mm": INVERSE_GROUP,
                "step_field_mm": STEP_GROUP,
                "step_inverse_field_mm": STEP_INVERSE_GROUP}
#: The record's other arrays, by the image group that holds each.
IMAGE_GROUPS = {"labels": "warped_labels", "footprint": "footprint", "left_out": "left_out",
                "weights": "weights"}
LEVEL = "s0"
SECTION, PLACED, ATLAS = "section", "placed_atlas", "atlas"
BLOSC_LEVEL = 5

#: The atlas axis each orientation letter starts from (BrainGlobe's origin
#: convention: the axis runs away from that side).
_AXIS_RUNS = {
    "a": "anterior to posterior", "p": "posterior to anterior",
    "s": "superior to inferior", "i": "inferior to superior",
    "r": "right to left", "l": "left to right",
}


def at_storage_precision(field_mm: np.ndarray) -> np.ndarray:
    """*field_mm* as the store keeps it: rounded to float16, as float32."""
    return np.asarray(field_mm).astype(FIELD_DTYPE).astype(np.float32)


# --- the metadata ---------------------------------------------------------------------


def _plane_axes(unit: str) -> list[dict[str, Any]]:
    return [{"name": "y", "type": "space", "unit": unit},
            {"name": "x", "type": "space", "unit": unit}]


def _system(name: str, axes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "axes": axes}


def _atlas_axes(orientation: str) -> list[dict[str, Any]]:
    axes = []
    for index in range(3):
        axis: dict[str, Any] = {"name": f"axis{index}", "type": "space", "unit": "micrometer"}
        letter = orientation[index] if len(orientation) == 3 else ""
        if letter in _AXIS_RUNS:
            axis["longName"] = f"atlas volume axis {index}, {_AXIS_RUNS[letter]}"
        axes.append(axis)
    return axes


def _yx(matrix_xy: np.ndarray) -> list[list[float]]:
    """A 2-D affine on ``[x, y, 1]`` (3x3) as the spec's ``(y, x)`` affine rows."""
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return (swap @ matrix_xy @ swap)[:2].tolist()


def _to_volume_yx(native_to_volume: np.ndarray) -> list[list[float]]:
    """``volume_index = M @ [nx, ny, 1]`` as the spec's affine on ``(y, x)``."""
    m = np.asarray(native_to_volume, dtype=np.float64)
    return np.stack([m[:, 1], m[:, 0], m[:, 2]], axis=1).tolist()


def _image_attributes(name: str, system: dict[str, Any], scale: list[float],
                      *, origin: list[float] | None = None) -> dict[str, Any]:
    to_system = {"input": {"path": LEVEL}, "output": {"name": system["name"]}}
    placing = {"type": "scale", "scale": scale, **to_system}
    if origin is not None:  # a working grid: its first pixel is off zero
        steps = [{"type": "scale", "scale": scale},
                 {"type": "translation", "translation": origin}]
        placing = {"type": "sequence", "transformations": steps, **to_system}
    return {"ome": {"version": OME_VERSION, "multiscales": [{
        "name": name,
        # One level: nothing was downsampled (the spec's SHOULD fields).
        "type": "none",
        "metadata": {"description": "a single full-resolution level"},
        "coordinateSystems": [system],
        "datasets": [{"path": LEVEL, "coordinateTransformations": [placing]}],
    }]}}


def _field_attributes(name: str, frame: str, mm: float,
                      *, grid: WorkingGrid | None = None) -> dict[str, Any]:
    axes = [{"name": "c", "type": "displacement", "discrete": True}, *_plane_axes("millimeter")]
    if grid is None:
        return _image_attributes(name, _system(frame, axes), [1.0, mm, mm])
    spacing_x, spacing_y = grid.spacing_mm
    origin_x, origin_y = grid.origin_mm
    return _image_attributes(name, _system(frame, axes),
                             [1.0, spacing_y, spacing_x],
                             origin=[0.0, origin_y, origin_x])


def scene_attributes(record: DeformableRecord, images: list[str]) -> dict[str, Any]:
    """The root group's attributes: the ``ome.scene`` of the module text,
    *images* the image groups written."""
    mm = float(record.mm_per_px)
    groups = _whole_map_groups(record)
    deformation = []
    if groups is not None:
        forward = {"type": "displacements", "path": groups[0],
                   "interpolation": "linear", "name": "section to placed atlas"}
        warp: dict[str, Any] = forward
        if groups[1] is not None:
            inverse = {"type": "displacements", "path": groups[1],
                       "interpolation": "linear",
                       "name": "placed atlas to section"}
            warp = {"type": "bijection", "forward": forward, "inverse": inverse}
        ends = {"input": {"name": SECTION}, "output": {"name": PLACED}}
        deformation = [{**warp, "name": "the deformation", **ends}]
    atlas = record.atlas
    resolution = [float(v) for v in atlas["resolution_um"]]
    placement = {
        "type": "sequence", "name": "the linear placement",
        "input": {"name": PLACED}, "output": {"name": ATLAS},
        "transformations": [
            {"type": "scale", "scale": [1.0 / mm, 1.0 / mm],
             "name": "millimetres to placed-atlas pixels"},
            {"type": "affine", "affine": _yx(np.linalg.inv(record.placement.atlas_to_section)),
             "name": "placed-atlas pixels to native atlas-plane pixels"},
            {"type": "affine", "affine": _to_volume_yx(atlas["native_to_volume_index"]),
             "name": "native atlas-plane pixels to atlas volume indices"},
            {"type": "scale", "scale": resolution, "name": "volume indices to micrometres"},
        ],
    }
    links = [{"type": "identity", "input": {"name": SECTION, "path": group},
              "output": {"name": SECTION}} for group in images]
    return {"ome": {"version": OME_VERSION, "scene": {
        "coordinateSystems": [
            _system(SECTION, _plane_axes("millimeter")),
            _system(PLACED, _plane_axes("millimeter")),
            _system(ATLAS, _atlas_axes(str(atlas.get("orientation", "")))),
        ],
        "coordinateTransformations": [*deformation, placement, *links],
    }}}


def _whole_map_groups(
    record: DeformableRecord,
) -> tuple[str, str | None] | None:
    """Return the field groups when they hold the whole deformation.

    The forward group and the inverse's (None without one); None when the
    stored arrays are a step to compose with the parent, or a landmark
    mesh's (no arrays).
    """
    step = record.stored_step
    if step is None:
        if record.inverse_field_mm is None:
            return FIELD_GROUP, None
        return FIELD_GROUP, INVERSE_GROUP
    if isinstance(step, EngineStep) and record.parent is None:
        return STEP_GROUP, STEP_INVERSE_GROUP
    return None


# --- writing and reading --------------------------------------------------------------


def _blosc(dtype: np.dtype[Any]) -> Any:
    from zarr.codecs import BloscCodec

    shuffle = "bitshuffle" if dtype.itemsize == 1 else "shuffle"
    return BloscCodec(cname="zstd", clevel=BLOSC_LEVEL, shuffle=shuffle)


def _array(group: Any, name: str, data: np.ndarray, dimensions: tuple[str, ...]) -> None:
    data = np.ascontiguousarray(data)
    group.create_array(name, data=data, chunks=data.shape, compressors=_blosc(data.dtype),
                       dimension_names=dimensions)


def _field_planes(field_mm: np.ndarray) -> np.ndarray:
    """``(rows, cols, 2)`` ``(dx, dy)`` as the stored ``(2, rows, cols)`` ``(dy, dx)``."""
    return np.stack([field_mm[..., 1], field_mm[..., 0]])


def write(directory: str | Path, record: DeformableRecord) -> Path:
    """Write *record*'s arrays as ``<directory>/warp.ome.zarr`` (replacing one
    left there); return its path."""
    import zarr

    path = Path(directory) / WARP_STORE
    if path.exists():
        shutil.rmtree(path)
    mm = float(record.mm_per_px)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    transforms = root.create_group("coordinateTransformations")

    # a record without a step keeps its fields whole, on the section grid
    step = record.stored_step
    whole = [("field", record.field_mm, SECTION)] if step is None else []
    if record.inverse_field_mm is not None and step is None:
        whole.append(("inverse_field", record.inverse_field_mm, PLACED))
    for name, field_mm, frame in whole:
        attributes = _field_attributes(name, frame, mm)
        group = transforms.create_group(name, attributes=attributes)
        planes = _field_planes(field_mm).astype(FIELD_DTYPE)
        _array(group, LEVEL, planes, ("c", "y", "x"))

    # an engine's step on its working grid, as the engine produced it
    if isinstance(step, EngineStep):
        engine_fields = [("step_field", step.field_mm, SECTION),
                         ("step_inverse_field", step.inverse_field_mm, PLACED)]
        for name, field_mm, frame in engine_fields:
            attributes = _field_attributes(name, frame, mm, grid=step.grid)
            group = transforms.create_group(name, attributes=attributes)
            planes = _field_planes(field_mm).astype(STEP_DTYPE)
            _array(group, LEVEL, planes, ("c", "y", "x"))

    images: list[str] = []
    for attribute, name in IMAGE_GROUPS.items():
        data = getattr(record, attribute)
        if data is None:
            continue
        group = root.create_group(name, attributes=_image_attributes(
            name, _system(SECTION, _plane_axes("millimeter")), [mm, mm]))
        _array(group, LEVEL, np.asarray(data), ("y", "x"))
        images.append(name)
    root.attrs.put(scene_attributes(record, images))
    return path


def _field(root: Any, group: str) -> np.ndarray:
    planes = np.asarray(root[f"{group}/{LEVEL}"][...])
    return np.ascontiguousarray(np.stack([planes[1], planes[0]], axis=-1), dtype=np.float32)


def read(directory: str | Path) -> dict[str, np.ndarray | None]:
    """The arrays of ``<directory>/warp.ome.zarr``: the fields of
    :data:`FIELD_GROUPS` (``(rows, cols, 2)`` float32 ``(dx, dy)``),
    ``labels``, ``footprint``, ``left_out`` and ``weights``, each None when
    not stored. ``FileNotFoundError`` without the store."""
    import zarr

    path = Path(directory) / WARP_STORE
    if not (path / "zarr.json").exists():
        raise FileNotFoundError(f"{path} does not exist")
    root: Any = zarr.open_group(str(path), mode="r")
    members = {name for name, _ in root.members(max_depth=None)}
    arrays: dict[str, np.ndarray | None] = {}
    for key, group in FIELD_GROUPS.items():
        arrays[key] = _field(root, group) if f"{group}/{LEVEL}" in members else None
    for attribute, name in IMAGE_GROUPS.items():
        key = f"{name}/{LEVEL}"
        arrays[attribute] = np.asarray(root[key][...]) if key in members else None
    return arrays
