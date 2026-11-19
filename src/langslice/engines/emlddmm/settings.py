"""The numbers the emlddmm tools keep: constants, options and the steps a call may give."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import numpy as np

from langslice.engines._devices import resolve_device
from langslice.engines._options import device_option, known_options
from langslice.engines._sorting import SORTING_KEYS
from langslice.job.settings.tool_table import TOOLS
from langslice.plugin_api import (
    PluginError,
    RefusalCode,
    ToolError,
)

Array = np.ndarray


Result = TypeVar("Result")


OPTIONS = ("device", "dtype", "section_um", "section_weight", "memory_limit_gb")


DTYPES = ("float32", "float64")


FITS = ("transform", "warp", "both")


ENGINE_VERSION = "emlddmm 9c4dc80 + e426ece + 6ff62b0"


NEEDS_TRANSFORM = "needs a transform first"


ANGLE_TOLERANCE_DEG = 1.0


LATTICE_PX = 8


UM_PER_MM = 1000.0


OVERLAYS = TOOLS["emlddmm_stack"].limits["pictures"]


SCALE_FIRST = 1.0


SCALES_HALF, SCALES_QUARTER = 2, 4


MIN_ITERATIONS = 4


GETS = {
    "deformation": "its transform kept, the rigid change folded into its deformation",
    "none": "nothing (its transform or deformation is frozen)",
}


@dataclass(frozen=True)
class PublishedExample:
    """Hold the constants of emlddmm's published mouse serial-section example.

    Each section is weighed as the gap it stands for: the warp's sigma times
    sqrt(k), the warp's and the affine's steps times k, with k the section's
    thickness over the slot spacing, and the slice step times the example's
    slice thickness over the slot spacing.

    Parameters
    ----------
    iterations
        The steps at its three scales.
    kernel_um
        The warp's kernel width, um.
    warp_freedom
        The warp's freedom (``sigmaR`` before the weighing).
    velocity_size_um
        The velocity field's grid spacing (``dv``), um.
    warp_step
        The warp's gradient step (``ev``).
    affine_step
        The stack affine's gradient step (``eA``).
    slice_step
        Each slice's in-plane gradient step (``eA2d``), at ``slice_um``.
    slice_um
        The example's slice spacing, um.
    no_warp
        A warp start past every step: the warp stays off.
    """

    iterations: tuple[int, int, int] = (100, 50, 25)
    kernel_um: float = 500.0
    warp_freedom: float = 10000.0
    velocity_size_um: float = 1000.0
    warp_step: float = 1e-2
    affine_step: float = 1e7
    slice_step: float = 1e5
    slice_um: float = 20.0
    no_warp: int = 10**6


@dataclass(frozen=True)
class Memory:
    """Hold the memory model of the stack's arrays and the slot search.

    Parameters
    ----------
    gb_per_megavoxel
        The memory one million voxels of the stack take at float64.
    search_max_um, search_step_um
        The slot spacings tried, to say which one would fit.
    """

    gb_per_megavoxel: float = 0.4
    search_max_um: float = 5000.0
    search_step_um: float = 5.0


@dataclass(frozen=True)
class Geometry:
    """Hold the numbers of the stack grid and of reading a section onto it.

    Parameters
    ----------
    edge_samples
        The points sampled along each side of a section's fit grid to find
        the stack's footprint.
    fixed_point_steps
        The iterations that invert a deformation's map.
    margin_pixels
        The empty margin around the sections' footprints, in grid pixels.
    byte_range
        A section image whose maximum is above this is read as 0-255, else
        as 0-1.
    white_slide
        The slide's level over its range above which it is white.
    """

    edge_samples: int = 24
    fixed_point_steps: int = 20
    margin_pixels: int = 2
    byte_range: float = 1.5
    white_slide: float = 0.5


@dataclass(frozen=True)
class SheetLayout:
    """Hold the layout of a section's sheet of the images the engine saw.

    Parameters
    ----------
    display_max
        The brightness shown as white for the section as sent.
    percentiles
        The atlas image's window, percentiles inside the tissue.
    enlarge
        How many times the sheet is enlarged.
    band
        The caption band's height, px.
    gap
        The white strip between the three tiles, px.
    font
        The caption font's size.
    """

    display_max: float = 1.6
    percentiles: tuple[float, float] = (1.0, 99.0)
    enlarge: int = 3
    band: int = 22
    gap: int = 4
    font: int = 11


@dataclass(frozen=True)
class EnergyPlot:
    """Hold the size of the energy plot, px, and its font.

    Parameters
    ----------
    height, width
        The image's size.
    pad
        The margin around the lines.
    font
        The label font's size.
    """

    height: int = 260
    width: int = 720
    pad: int = 34
    font: int = 9


EXAMPLE = PublishedExample()


MEMORY = Memory()


GEOMETRY = Geometry()


SHEET = SheetLayout()


PLOT = EnergyPlot()


PUBLISHED_SORTING: dict[str, Any] = {}


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    dtype
        ``float32`` or ``float64``.
    section_um
        The pixel of the stack grid, um.
    section_weight
        The thickness one section stands for, um; None is the gap between
        neighbouring slots.
    memory_limit_gb
        The most memory a stack may need.
    """

    device: str = "auto"
    dtype: str = "float64"
    section_um: float = 58.88
    section_weight: float | None = None
    memory_limit_gb: float = 24.0

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
            For an option the plugin does not take, or a dtype that is not
            ``float32`` or ``float64``.
        """
        given = known_options("emlddmm", config, OPTIONS)
        dtype = str(given.get("dtype", cls.dtype))
        if dtype not in DTYPES:
            raise PluginError(f"emlddmm's dtype is one of {', '.join(DTYPES)}; got {dtype!r}")
        weight = given.get("section_weight", "gap")
        return cls(
            device=device_option(given.get("device", cls.device)),
            dtype=dtype,
            section_um=float(str(given.get("section_um", cls.section_um))),
            section_weight=None if weight == "gap" else float(str(weight)),
            memory_limit_gb=float(str(given.get("memory_limit_gb", cls.memory_limit_gb))),
        )

    def settings(self) -> dict[str, object]:
        """Return the options as the call's files record them."""
        return {
            "device": self.device,
            "dtype": self.dtype,
            "section_um": self.section_um,
            "section_weight": "gap" if self.section_weight is None else self.section_weight,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def resolved(self) -> Options:
        """Return the options with ``auto`` replaced by the device it names."""
        return replace(self, device=resolve_device(self.device))


def scale_iterations(iterations: int) -> list[int]:
    """Return emlddmm's steps at its three scales (200, 100, 50 um) from the first scale's: that,
    half of it and a quarter, as the published example's [100, 50, 25] runs."""
    first = int(iterations)
    return [first, max(1, first // 2), max(1, first // 4)]


def check_iterations(iterations: Any) -> list[int]:
    """The three scales' steps of *iterations*, the first scale's (:func:`scale_iterations`)."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < MIN_ITERATIONS
    ):
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "iterations is a whole number of steps at the coarsest "
            "scale, 4 or more (the finer scales take half and a quarter)",
        )
    return scale_iterations(iterations)


def check_sorting(sorting: dict[str, Any]) -> dict[str, float]:
    held = [key for key in sorting if key in ("method", "artifact")]
    if held:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"emlddmm's sorting is its own EM with an artifact class: "
            f"it takes no {' or '.join(held)}; its keys are "
            f"{', '.join(SORTING_KEYS)}",
        )
    unknown = [key for key in sorting if key not in SORTING_KEYS]
    if unknown:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"unknown sorting keys {unknown}; the keys are {', '.join(SORTING_KEYS)}",
        )
    return {key: float(value) for key, value in sorting.items()}
