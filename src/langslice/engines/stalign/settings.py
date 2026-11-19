"""The numbers the STalign tools keep: constants and options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from langslice.engines._options import device_option, known_options
from langslice.engines.stalign import engine

Result = TypeVar("Result")


OPTIONS = ("device", "threads")


FITS = ("warp", "both")


SORTING_METHODS = ("classic", "em", "none")


ENGINE_VERSION = "STalign 2.0.0 8a84bf1f"


SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


BRIGHTNESS = "slide 0, tissue 1"


SAME_LEVEL = 1e-6


MIN_SCALE = 1e-12


LOSS_COLUMNS = ("iteration", "total", "image_mismatch", "smoothness", "landmarks")


LOSS_ROW_UM = 1


@dataclass(frozen=True)
class Resolutions:
    """Hold the micrometres the two images are averaged to.

    Parameters
    ----------
    target_um
        The section image's pixel, um.
    source_um
        The atlas image's pixel, um (the published atlas voxel).
    """

    target_um: float = 10.0
    source_um: float = 50.0


@dataclass(frozen=True)
class SearchStarts:
    """Hold the plane search's start grid.

    Parameters
    ----------
    tilt_deg
        The largest tilt about either in-plane axis (starts are -tilt, 0, tilt).
    shift_um
        The largest shift along the slicing axis (starts are -shift, 0, shift).
    """

    tilt_deg: float = 6.0
    shift_um: float = 300.0


@dataclass(frozen=True)
class LossPlot:
    """Hold the size of the mismatch plot, px.

    Parameters
    ----------
    height, width
        The image's size.
    pad
        The margin around the line.
    """

    height: int = 240
    width: int = 640
    pad: int = 30


@dataclass(frozen=True)
class Pictures:
    """Hold the sizes of the sorting picture.

    Parameters
    ----------
    gap_px
        The white strip between the section and the sorting.
    long_edge
        The longest side of the picture sent.
    """

    gap_px: int = 8
    long_edge: int = 1600


RESOLUTIONS = Resolutions()


STARTS = SearchStarts()


LOSS_PLOT = LossPlot()


PICTURES = Pictures()


PUBLISHED_SORTING: dict[str, Any] = {}


MASK_ENGINES = ("fireants_syn", "elastix_affine", "emlddmm_stack", "spomialign")


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    threads
        The threads STalign may use, and the sections searched at once.
    """

    device: str = "auto"
    threads: int = 4

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> Options:
        """Read the options a job's ``plugin_config`` gives.

        Parameters
        ----------
        config
            The plugin's entry of ``plugin_config``; None gives the defaults.

        Returns
        -------
        Options
            The options.

        Raises
        ------
        PluginError
            For an option the plugin does not take.
        """
        given = known_options("STalign", config, OPTIONS)
        return cls(
            device=device_option(given.get("device", cls.device)),
            threads=int(str(given.get("threads", cls.threads))),
        )

    @property
    def machine(self) -> engine.Machine:
        """Return where STalign runs."""
        return engine.Machine(self.device, self.threads)
