"""Images and transforms in the ITK world (SimpleITK): what an outside engine
reads and writes.

An engine run outside LangSlice (``antsRegistration``, ``elastix`` /
``transformix``, SimpleITK, FireANTs in another environment) reads images
from files and answers with a transform file (an affine ``.mat`` / ``.txt`` /
``.tfm``, a composite ``.h5``) or a displacement-field image (``.nii.gz``,
``.mha``, ``.nrrd``). Here those are written and read on a 2-D pixel grid
with ``spacing`` per pixel and ``origin`` at pixel ``(0, 0)`` (the defaults,
1 and 0, make physical units pixels), identity direction.

ITK's convention throughout: a transform maps a point of the FIXED image to
the point of the MOVING image it corresponds to, and a displacement field
holds, at each fixed point, that point's offset ``(dx, dy)`` in physical
units (``x`` the column, ``y`` the row).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: File suffixes read as transforms (the rest as images).
TRANSFORM_SUFFIXES = (".mat", ".txt", ".tfm", ".h5", ".hdf5", ".xfm")
#: Image formats :func:`write_image` writes (by suffix).
IMAGE_FORMATS = ("tif", "nii.gz", "nii", "mha", "nrrd")
#: How far from affine a transform may be, in pixels at the grid's corners, to
#: be read as one (:func:`linear_matrix`).
LINEAR_TOLERANCE_PX = 1e-3


def write_image(path: str | os.PathLike[str], array: np.ndarray, *,
                spacing: float = 1.0) -> Path:
    """Write *array* (``(H, W)`` or ``(H, W, C)``) at *path*: a TIFF with
    tifffile (RGB stays RGB), any other suffix with SimpleITK on the grid
    (``spacing`` per pixel, origin 0, identity direction; a multi-channel
    array as a vector image)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(array)
    if target.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(target, data, photometric="rgb" if data.ndim == 3 and
                         data.shape[-1] == 3 else None)
        return target
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(data, isVector=data.ndim == 3)
    image.SetSpacing((float(spacing), float(spacing)))
    image.SetOrigin((0.0, 0.0))
    sitk.WriteImage(image, str(target))
    return target


def _transform(source: Any) -> Any:
    """*source* as a SimpleITK transform: a transform file, a displacement
    field image file, a ``sitk.Transform`` or a ``sitk.Image`` (vector)."""
    import SimpleITK as sitk

    if isinstance(source, sitk.Transform):
        return source
    if isinstance(source, sitk.Image):
        image = source
    else:
        path = Path(os.fspath(source))
        if not path.is_file():
            raise ValueError(f"no such file: {path}")
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in TRANSFORM_SUFFIXES):
            return sitk.ReadTransform(str(path))
        image = sitk.ReadImage(str(path))
    if image.GetNumberOfComponentsPerPixel() != 2 or image.GetDimension() != 2:
        raise ValueError("a displacement field is a 2-D image of 2-vectors (dx, dy); got "
                         f"{image.GetDimension()}-D with "
                         f"{image.GetNumberOfComponentsPerPixel()} components")
    return sitk.DisplacementFieldTransform(sitk.Cast(image, sitk.sitkVectorFloat64))


def _grid(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    yy, xx = np.indices((height, width), dtype=np.float64)
    return np.stack([xx, yy], axis=-1)


def mapped_points(source: Any, size: tuple[int, int], *, spacing: float = 1.0,
                  origin: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Where *source* sends each pixel centre of a grid of *size* ``(width,
    height)``: ``(H, W, 2)`` ``[x, y]`` pixels of the same grid units
    (physical ``origin + pixel * spacing`` there and back). *source* is a
    transform or displacement-field file, a ``sitk.Transform`` or
    ``sitk.Image``, or a float array ``(H, W, 2)`` of displacements ``(dx,
    dy)`` in pixels."""
    width, height = (int(v) for v in size)
    if isinstance(source, np.ndarray):
        array = np.asarray(source, dtype=np.float64)
        if array.shape != (height, width, 2):
            raise ValueError(f"a displacement array is (H, W, 2) = ({height}, {width}, 2); "
                             f"got {array.shape}")
        return _grid((width, height)) + array
    import SimpleITK as sitk

    transform = _transform(source)
    start = [float(v) for v in origin]
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, [width, height], start,
        [float(spacing), float(spacing)], [1.0, 0.0, 0.0, 1.0])
    offsets = sitk.GetArrayFromImage(field) / float(spacing)
    return _grid((width, height)) + offsets


def inverted(points: np.ndarray, *, iterations: int = 50) -> np.ndarray:
    """The inverse of a map *points* ``(H, W, 2)`` (where each pixel centre
    of a grid goes, on the same grid): for each pixel centre, the point that
    goes there (SimpleITK's ``InvertDisplacementField``, fixed-point)."""
    import SimpleITK as sitk

    height, width = points.shape[:2]
    displacement = np.asarray(points, dtype=np.float64) - _grid((width, height))
    image = sitk.GetImageFromArray(displacement, isVector=True)
    inverse = sitk.InvertDisplacementField(
        image, maximumNumberOfIterations=int(iterations), maxErrorToleranceThreshold=0.01,
        meanErrorToleranceThreshold=0.001, enforceBoundaryCondition=False)
    return _grid((width, height)) + sitk.GetArrayFromImage(inverse)


def linear_matrix(source: Any, size: tuple[int, int], *, spacing: float = 1.0,
                  origin: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """*source* (a transform file or ``sitk.Transform``) as a 3x3 affine on
    ``[x, y, 1]`` pixels of a grid of *size*; ``ValueError`` when it is not
    affine there (a deformable transform)."""
    transform = _transform(source)
    width, height = (int(v) for v in size)
    probe = np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0],
                      [width - 1.0, height - 1.0], [(width - 1) / 2.0, (height - 1) / 2.0]])
    start = np.asarray(origin, dtype=np.float64)
    moved = np.array([
        (np.asarray(transform.TransformPoint(tuple(start + p * spacing))) - start) / spacing
        for p in probe])
    ones = np.ones((len(probe), 1))
    solved, *_ = np.linalg.lstsq(np.hstack([probe, ones]), moved, rcond=None)
    matrix = np.vstack([solved.T, [0.0, 0.0, 1.0]])
    if np.abs(np.hstack([probe, ones]) @ solved - moved).max() > LINEAR_TOLERANCE_PX:
        raise ValueError("the transform is not affine over the grid")
    return matrix


__all__ = ["IMAGE_FORMATS", "TRANSFORM_SUFFIXES", "inverted", "linear_matrix",
           "mapped_points", "write_image"]
