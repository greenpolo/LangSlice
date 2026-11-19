"""The numbers the SPOmiAlign tool keeps: its constants, options and the mode of a call."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langslice.engines import _blocks
from langslice.engines._devices import AUTO
from langslice.engines._options import device_option, known_options
from langslice.engines.spomialign import engine, matching

OPTIONS = ("device",)


MISSING = (
    "SPOmiAlign cannot run here: its matcher, RoMa (the romatch package), "
    "is not installed; install LangSlice's engines extra: "
    'pip install "langslice[engines]". Use another method, or ask the user to install it.'
)


CANVAS_EDGE = 1536


FITS = ("transform", "warp", "both")


WRITES = {
    engine.RIGID: ("affine",),
    engine.AFFINE: ("affine",),
    engine.WARP: ("deformation",),
    engine.AFFINE_WARP: ("affine", "deformation"),
}


ENGINE_VERSION = "SPOmiAlign c7f8b7a"


MATCHER = "RoMa roma_outdoor, coarse resolution 560"


MIN_MATCHES = 3


BRIGHTNESS = "8-bit, Nissl rendered dark on white"


NISSL = _blocks.NISSL


WORK_W, WORK_H = matching.W, matching.H


@dataclass(frozen=True)
class AtlasWindow:
    """Hold the percentile window the atlas plate is shown through.

    Parameters
    ----------
    low, high
        The percentiles of the atlas image inside the brain that map to
        black ink and to white.
    """

    low: float = 0.5
    high: float = 99.5


@dataclass(frozen=True)
class MatchesPicture:
    """Hold the layout of the matches picture saved with a call.

    Parameters
    ----------
    drawn_pairs
        How many of the most confident pairs are drawn.
    gap_px
        The white strip between the two canvases.
    long_edge
        The longest side of the saved picture.
    cross_colour
        The colour of the crosses marking passed-over pairs.
    """

    drawn_pairs: int = 300
    gap_px: int = 16
    long_edge: int = 2400
    cross_colour: tuple[int, int, int] = (220, 30, 30)


ATLAS_WINDOW = AtlasWindow()


MATCHES_PICTURE = MatchesPicture()


def mode_of(fit: str, rigid: bool) -> str:
    """The engine mode of a call's *fit* and *rigid*: ``rigid`` or ``affine``
    for the transform, ``warp``, or ``affine_warp`` for both."""
    if fit == "transform":
        return engine.RIGID if rigid else engine.AFFINE
    return engine.WARP if fit == "warp" else engine.AFFINE_WARP


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        The torch device: ``auto`` (the first GPU when torch sees one, else
        the CPU), ``cpu``, ``cuda`` or a numbered GPU.
    """

    device: str = AUTO

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
        given = known_options("SPOmiAlign", config, OPTIONS)
        return cls(device=device_option(given.get("device", cls.device)))
