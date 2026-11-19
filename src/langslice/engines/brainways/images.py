"""Make the images Brainways' network reads: the thumbnail and its input.

Vendored from Brainways 0.1.16.3 (Kantor et al. 2025, Neuropsychopharmacology;
https://github.com/bkntr/brainways, commit
1c43b0b708564974ce057167b6f83ed967dd3b04). Copyright (C) Ben Kantor and the
Brainways contributors. This program is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. It is distributed WITHOUT ANY
WARRANTY; see the GNU General Public License in ``LICENSE`` beside this file.

:func:`section_thumbnail` is Brainways' thumbnail of one channel
(``brainways.utils.image``: ``get_resize_size``, ``resize_image`` and
``slice_to_uint8``, copied). :func:`network_input` is the albumentations
chain ``brainways.model.model_utils.load_model`` builds (``Resize`` to
224 x 224 by area, ``NormalizePercentile`` at 0.1 and 99.8, ``FromFloat`` to
8 bits, ``ToRGB``, ``Normalize`` with the ImageNet mean and deviation,
``ToTensorV2``), written out with numpy and OpenCV.

Brainways pins numpy below 2, and the chain's results depend on numpy 1's
type promotion: its percentiles take the quantile index in float32 and
interpolate in float64, a section's stretch runs in float64 and an atlas
plane's in float32. Under numpy 2 the same calls run in float32 and move
some positions by a plane or two, so every step here names its types and
gives numpy 1's result under either numpy.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

THUMBNAIL_HW = (1024, 1024)
SATURATION = 0.001
HISTOGRAM_BINS = 1024
INPUT_PX = 224
PERCENTILES = np.array((0.1, 99.8), dtype=np.float32)
MEAN = np.multiply((0.485, 0.456, 0.406), 255.0, dtype=np.float32)
DENOMINATOR = np.reciprocal(np.multiply((0.229, 0.224, 0.225), 255.0, dtype=np.float32))


def section_thumbnail(
    channel: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Make Brainways' thumbnail of one channel of a section.

    Parameters
    ----------
    channel : array, shape (H, W), uint8
        One channel of the section, oriented as it is placed.

    Returns
    -------
    array, uint8
        The channel fitted into 1024 x 1024 with its aspect kept, its
        contrast stretched to 0-255 with 0.1 % of the pixels saturated at
        each end.
    """
    height, width = _height_width(channel)
    size = resize_size((height, width), THUMBNAIL_HW)
    image = channel
    if (height, width) != size:
        image = _resize_image(image, size)
    return _stretch_to_uint8(image)


def resize_size(input_hw: tuple[int, int], output_hw: tuple[int, int]) -> tuple[int, int]:
    """Fit a size into another, keeping its aspect (``get_resize_size``).

    Parameters
    ----------
    input_hw
        The image's height and width, pixels.
    output_hw
        The height and width it must fit in, pixels.

    Returns
    -------
    tuple of int
        The fitted height and width, pixels.
    """
    factor = max(input_hw[0] / output_hw[0], input_hw[1] / output_hw[1])
    return round(input_hw[0] / factor), round(input_hw[1] / factor)


def _resize_image(image: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
    height, width = _height_width(image)
    output = resize_size((height, width), size)
    if output <= (height, width):
        factor = min(height / output[0], width / output[1])
        halvings = int(cast("float", np.log2(factor)))
        for _ in range(halvings):
            height, width = _height_width(image)
            half = (int(width / 2), int(height / 2))
            image = np.asarray(cv2.resize(image, dsize=half), dtype=np.uint8)
    resized = cv2.resize(image, dsize=(output[1], output[0]))
    return np.asarray(resized, dtype=np.uint8)


def _height_width(image: npt.NDArray[np.generic]) -> tuple[int, int]:
    shape = cast("tuple[int, ...]", image.shape)
    return shape[0], shape[1]


def _stretch_to_uint8(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    low, high = _contrast_limits(image)
    clipped = np.clip(image.astype(np.float64), low, high)
    smallest, largest = np.min(clipped), np.max(clipped)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = (clipped - smallest) / (largest - smallest)
    return np.round(unit * 255).astype(np.uint8)


def _contrast_limits(image: npt.NDArray[np.uint8]) -> tuple[float, float]:
    counts, edges = np.histogram(image.ravel(), bins=HISTOGRAM_BINS)
    hist = cast("list[int]", counts.tolist())
    bounds = cast("list[float]", edges.tolist())
    if hist[0] > hist[1]:
        hist, bounds = hist[1:], bounds[1:]
    if hist[-1] > hist[-2]:
        hist, bounds = hist[:-1], bounds[:-1]
    saturated = float(sum(hist)) * SATURATION
    return _low_limit(hist, bounds, saturated), _high_limit(hist, bounds, saturated)


def _low_limit(hist: list[int], bounds: list[float], saturated: float) -> float:
    count = saturated
    for index in range(len(hist) - 1):
        if count < hist[index]:
            width = bounds[index + 1] - bounds[index]
            return bounds[index] + (count / hist[index]) * width
        count -= hist[index]
    return bounds[0]


def _high_limit(hist: list[int], bounds: list[float], saturated: float) -> float:
    count = saturated
    for index in range(len(hist) - 1, -1, -1):
        if count < hist[index]:
            width = bounds[index + 1] - bounds[index]
            return bounds[index + 1] - (count / hist[index]) * width
        count -= hist[index]
    return bounds[-1]


def section_input(thumbnail: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Turn a section's thumbnail into the network's input.

    Parameters
    ----------
    thumbnail : array, shape (H, W), uint8
        The section's thumbnail (:func:`section_thumbnail`).

    Returns
    -------
    array, shape (3, 224, 224), float32
        The thumbnail at 224 x 224, stretched between its 0.1 and 99.8
        percentiles in float64, at 8 bits, repeated in three channels and
        normalized with the ImageNet mean and deviation.
    """
    resized = np.asarray(_to_input_size(thumbnail), dtype=np.uint8)
    lower, upper = _percentiles(resized.astype(np.float64), float32=False)

    clipped = np.clip(resized.astype(np.float64), lower, upper)
    unit = (clipped - lower) / (upper - lower + 1e-6)
    levels = np.clip(np.rint(unit * 255), 0, 255).astype(np.float32)
    return _normalized(levels)


def plane_input(plane: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Turn an atlas plane into the network's input.

    Parameters
    ----------
    plane : array, shape (H, W), float32
        The atlas plane, 0 to 1 (``network.ReferenceVolume.plane``).

    Returns
    -------
    array, shape (3, 224, 224), float32
        The plane at 224 x 224, stretched between its 0.1 and 99.8
        percentiles in float32, at 8 bits, repeated in three channels and
        normalized with the ImageNet mean and deviation.
    """
    resized = np.asarray(_to_input_size(plane), dtype=np.float32)
    lower, upper = _percentiles(resized.astype(np.float64), float32=True)

    low, high = np.float32(lower), np.float32(upper)
    span = np.float32(upper - lower + 1e-6)
    unit = (np.clip(resized, low, high) - low) / span
    levels = np.clip(np.rint(unit * np.float32(255)), 0, 255)
    return _normalized(levels.astype(np.uint8).astype(np.float32))


def _to_input_size(
    image: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
) -> object:
    if _height_width(image) == (INPUT_PX, INPUT_PX):
        return image
    size = (INPUT_PX, INPUT_PX)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _normalized(levels: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    channels = np.broadcast_to(levels, (3, INPUT_PX, INPUT_PX))
    centred = channels - MEAN[:, np.newaxis, np.newaxis]
    scaled = centred * DENOMINATOR[:, np.newaxis, np.newaxis]
    return np.asarray(scaled, dtype=np.float32)


def _percentiles(image: npt.NDArray[np.float64], *, float32: bool) -> tuple[float, float]:
    values = np.sort(image, axis=None)
    quantiles = PERCENTILES / np.float32(100)
    index = np.float32(values.size - 1) * quantiles
    below = np.floor(index).astype(np.intp)
    above = below + 1
    last = index >= values.size - 1
    below[last] = values.size - 1
    above[last] = values.size - 1

    weight = index.astype(np.float64) - below
    low, high = values[below], values[above]
    difference = high - low
    if float32:
        native = high.astype(np.float32) - low.astype(np.float32)
        difference = native.astype(np.float64)
    from_low = low + difference * weight
    from_high = high - difference * (1 - weight)
    result = np.where(weight >= 0.5, from_high, from_low)
    lower, upper = cast("list[float]", result.tolist())
    return lower, upper
