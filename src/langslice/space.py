from dataclasses import dataclass
from typing import Literal

import numpy as np
from brainglobe_space import AnatomicalSpace

Plane = Literal["coronal", "sagittal", "horizontal"]


@dataclass(frozen=True)
class AtlasSpaceContext:
    atlas_name: str
    orientation: str
    shape: tuple[int, int, int]
    resolution_um: tuple[float, float, float]
    space: AnatomicalSpace
    ap_axis_index: int
    dv_axis_index: int
    ml_axis_index: int
    ap_resolution_mm: float


def _shape3d(volume: object) -> tuple[int, int, int]:
    shape = getattr(volume, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"Expected a 3D atlas volume, got shape={shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _resolution3d_um(resolution: object) -> tuple[float, float, float]:
    if not isinstance(resolution, (tuple, list)) or len(resolution) != 3:
        raise ValueError(f"Expected 3-axis atlas resolution, got {resolution!r}")
    return float(resolution[0]), float(resolution[1]), float(resolution[2])


def atlas_space_context(atlas: object) -> AtlasSpaceContext:
    atlas_name = str(getattr(atlas, "atlas_name", "unknown"))
    orientation = str(getattr(atlas, "orientation", ""))
    shape = _shape3d(getattr(atlas, "template", None))
    resolution_um = _resolution3d_um(getattr(atlas, "resolution", None))
    space = AnatomicalSpace(
        origin=orientation,
        shape=shape,
        resolution=resolution_um,
    )
    ap_axis_index = space.get_axis_idx("sagittal")
    dv_axis_index = space.get_axis_idx("vertical")
    ml_axis_index = space.get_axis_idx("frontal")
    ap_axis_letter = space.origin[ap_axis_index]
    if ap_axis_letter != "a":
        raise ValueError(
            f"Atlas '{atlas_name}' orientation '{space.origin_string}' is not supported. "
            + "LangSlice currently requires AP axis to increase from anterior to posterior."
        )
    return AtlasSpaceContext(
        atlas_name=atlas_name,
        orientation=space.origin_string,
        shape=shape,
        resolution_um=resolution_um,
        space=space,
        ap_axis_index=ap_axis_index,
        dv_axis_index=dv_axis_index,
        ml_axis_index=ml_axis_index,
        ap_resolution_mm=resolution_um[ap_axis_index] / 1000.0,
    )


def slice_axis_index(context: AtlasSpaceContext, plane: Plane) -> int:
    """Return the axis index normal to the given slicing plane."""
    if plane == "coronal":
        return context.ap_axis_index
    if plane == "sagittal":
        return context.ml_axis_index
    if plane == "horizontal":
        return context.dv_axis_index
    raise ValueError(f"Unknown plane: {plane!r}")


#: Anatomical direction pair for a brainglobe origin letter (the letter names
#: where the axis STARTS): 'a' means the axis runs anterior -> posterior.
_AXIS_PAIR = {"a": "ap", "p": "pa", "s": "si", "i": "is", "l": "lr", "r": "rl"}
#: Direction pair -> the anatomical axis it lives on.
_AXIS_KIND = {"ap": "ap", "pa": "ap", "si": "si", "is": "si", "lr": "lr", "rl": "lr"}


#: Anatomical name of each end of an axis, keyed by the letter naming where
#: the axis STARTS: 'a' means it runs from anterior to posterior.
_AXIS_ENDS = {
    "a": ("anterior", "posterior"),
    "p": ("posterior", "anterior"),
    "s": ("superior", "inferior"),
    "i": ("inferior", "superior"),
    "l": ("left", "right"),
    "r": ("right", "left"),
}


def slice_axis_ends(context: AtlasSpaceContext, plane: Plane) -> tuple[str, str]:
    """(low, high) anatomical ends of the axis normal to *plane*.

    Positions along that axis are measured from index 0, so the first name is
    what 0 mm sits at and the second is what positions increase toward. A
    coronal plane on a supported atlas always answers ("anterior",
    "posterior") — :func:`atlas_space_context` refuses any other AP order.
    """
    return _AXIS_ENDS[context.space.origin[slice_axis_index(context, plane)]]


def native_slice_axes(context: AtlasSpaceContext, plane: Plane) -> tuple[str, str]:
    """(rows, cols) anatomical directions of a rendered slice.

    Directions are 2-letter pairs read start->end: a sagittal render of an
    'asr' atlas returns ("si", "ap") — rows run superior->inferior, columns
    anterior->posterior. Accounts for the in-plane axis swap
    ``orient_slice_for_display`` applies to sagittal and horizontal slices.
    """
    normal = slice_axis_index(context, plane)
    in_plane = [i for i in range(3) if i != normal]
    pairs = [_AXIS_PAIR[context.space.origin[i]] for i in in_plane]
    if plane in ("sagittal", "horizontal"):
        pairs.reverse()
    return pairs[0], pairs[1]


def orient_slice_to_axes(
    slice_arr: np.ndarray,
    context: AtlasSpaceContext,
    plane: Plane,
    image_axes: str,
) -> np.ndarray:
    """Rotate/flip a rendered slice into the user's image frame.

    ``image_axes`` is "rows,cols" in anatomical direction pairs, mirroring
    ABBA's atlas-slicing initialization — e.g. "ap,lr" for a horizontal image
    with anterior at the top and the animal's left on the image's left. Only
    right-angle transforms are applied; pixels are never resampled.
    """
    import numpy as np

    try:
        rows_req, cols_req = (t.strip().lower() for t in image_axes.split(","))
    except ValueError:
        raise ValueError(f"image_axes must be 'rows,cols', got {image_axes!r}") from None
    for token in (rows_req, cols_req):
        if token not in _AXIS_KIND:
            raise ValueError(f"Unknown anatomical direction {token!r} in image_axes")

    rows_nat, cols_nat = native_slice_axes(context, plane)
    arr = np.asarray(slice_arr)
    if _AXIS_KIND[rows_req] == _AXIS_KIND[cols_nat]:
        arr = np.swapaxes(arr, 0, 1)
        rows_nat, cols_nat = cols_nat, rows_nat
    if _AXIS_KIND[rows_req] != _AXIS_KIND[rows_nat] or _AXIS_KIND[cols_req] != _AXIS_KIND[
        cols_nat
    ]:
        raise ValueError(
            f"image_axes {image_axes!r} does not name the in-plane axes of a "
            f"{plane} slice (native: {rows_nat},{cols_nat})"
        )
    if rows_req != rows_nat:
        arr = arr[::-1]
    if cols_req != cols_nat:
        arr = arr[:, ::-1]
    return np.ascontiguousarray(arr)
