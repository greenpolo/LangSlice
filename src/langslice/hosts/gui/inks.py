"""What the window's looks share: the tone curve and the border ink.

Every look (:mod:`.calm`, :mod:`.paper`) turns a stain's signal into a
brightness with one :class:`ToneCurve` (:func:`tone`) and lays the atlas
borders over the picture with one :class:`BorderInk` (:func:`lay_borders`),
from the border coverage the core's layers carry
(:attr:`langslice.core.pictures.canvas.PlacementLayers.borders`). Nothing is
read back out of a finished picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

Rgb = tuple[int, int, int]
GREY_MAX = 255.0


@dataclass(frozen=True)
class ToneCurve:
    """How a stain's signal becomes a brightness from 0 to 1.

    The signal is stretched between its *low_percentile* and
    *high_percentile* over the tissue (the pixels whose weight is over
    *inside_weight*), at least *min_span* grey levels apart. A soft knee
    compresses the brightest spots instead of clipping them: the stretched
    value ``v`` becomes ``shoulder * v / (1 + knee * v)``, clipped to 1 and
    raised to *gamma*.
    """

    low_percentile: float = 4.0
    high_percentile: float = 99.0
    inside_weight: float = 0.5
    min_span: float = 1.0
    shoulder: float = 1.35
    knee: float = 0.35
    gamma: float = 0.95


@dataclass(frozen=True)
class BorderInk:
    """The atlas borders as a soft layer of one colour over a picture.

    A border pixel takes *colour* at *strength* times how much of the pixel
    the line covers, full from *full_coverage* of it. A small picture's lines
    are fainter: the strength is scaled by ``min(1, small_floor + long side /
    full_size_px)``.
    """

    colour: Rgb
    strength: float
    full_coverage: float = 0.43
    small_floor: float = 0.45
    full_size_px: float = 1600.0


TONE = ToneCurve()


def tone(
    signal: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    curve: ToneCurve = TONE,
) -> npt.NDArray[np.float32]:
    """Map a stain's signal to a brightness, 0 to 1.

    Parameters
    ----------
    signal
        The picture's one channel, 0 to 255.
    weight
        The tissue's weight (:func:`langslice.core.sections.tissue.tissue_mask`).
    curve
        The curve's constants.

    Returns
    -------
    np.ndarray
        The brightness of every pixel, the picture's size.
    """
    inside = signal[weight > curve.inside_weight]
    low = float(np.percentile(inside, curve.low_percentile)) if inside.size else 0.0
    high = float(np.percentile(inside, curve.high_percentile)) if inside.size else GREY_MAX
    stretched = np.clip((signal - low) / max(curve.min_span, high - low), 0, None)
    kneed = curve.shoulder * stretched / (1 + curve.knee * stretched)
    return (np.clip(kneed, 0, 1) ** curve.gamma).astype(np.float32)


def lay_borders(
    rgb: npt.NDArray[np.float32],
    lines: npt.NDArray[np.float32] | None,
    ink: BorderInk,
) -> npt.NDArray[np.float32]:
    """Lay the atlas borders over a picture in *ink*.

    Parameters
    ----------
    rgb
        The picture, float, rows by columns by three, 0 to 255.
    lines
        The borders' coverage on the same grid, 0 to 1; None: none.
    ink
        The borders' colour and strength.

    Returns
    -------
    np.ndarray
        The picture with the borders laid on it.
    """
    if lines is None:
        return rgb
    size = max(rgb.shape[:2])
    strength = ink.strength * min(1.0, ink.small_floor + size / ink.full_size_px)
    alpha = (np.clip(lines / ink.full_coverage, 0, 1) * strength)[..., None]
    return (rgb * (1 - alpha) + np.array(ink.colour, np.float32) * alpha).astype(np.float32)
