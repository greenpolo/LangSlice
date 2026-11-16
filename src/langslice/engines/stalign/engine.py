"""Call STalign 2.0 in LangSlice's process, for the plugin (``plugin.py``).

STalign 2.0.0 (JEFworks-Lab main 8a84bf1f, GPL-3.0) is vendored as
``langslice_vendor.stalign`` (its pure-torch engine; the optional C++ engine
is not included) and imported when a tool runs, never when the plugin
loads. The plugin prepares every image and frame; this module only calls
STalign and samples its result.

The inputs (``mode`` picks the call):

- ``lddmm``: 2-D ``STalign.LDDMM``. ``I`` (C x h x w source), ``xI0``,
  ``xI1`` (its pixel centres, um), ``J`` (1 x H x W target), ``xJ0``,
  ``xJ1``, ``A0`` (3 x 3, source um -> target um, row/col order),
  ``query`` (N x 2 target um, row/col: where the map is sampled).
- ``plane``: 3-D ``STalign.LDDMM_3D_to_slice`` with ``niter=0``: the plane
  search and its refinement only. ``atlas`` (BrainGlobe name), ``image``
  (``template``, one of its additional references or the borrowed Nissl),
  ``J``, ``xJ0``, ``xJ1``, ``A0`` (4 x 4, atlas um -> target um, target =
  (0, row um, col um)). The volume is read at 50 um (the published atlas
  voxel, ``_atlas_images.py``).

``params`` holds STalign's keyword arguments (its mixture included), with
``device`` and ``threads``. The result: ``source_um`` (N x 2 for 2-D: the
source point STalign's map takes each query point to), ``A``,
``WM``/``WB``/``WA``, ``energy`` (iterations x [E, EM, ER(, EP)], STalign's
own record), ``settings`` (JSON: what STalign chose), ``prefit`` (JSON,
3-D), ``device``, ``seconds`` and ``gpu_wait_s``.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast, overload

import numpy as np

from langslice.engines._atlas_images import averaged_image
from langslice.engines._devices import AUTO, CPU, on_gpu, resolve_device
from langslice.plugin_api import gpu_turn

if TYPE_CHECKING:
    import torch

#: The vendored STalign package.
PACKAGE = "langslice_vendor.stalign"
#: The ``mode`` of a plane search; any other runs the 2-D fit.
PLANE = "plane"
#: Query points sampled at once through STalign's map.
QUERY_CHUNK = 1 << 20

#: A value of STalign's keyword arguments as the plugin sets them.
Option = float | int | str | dict[str, str]
#: A value of a call's result.
Output = np.ndarray | str | float


class _Fit(Protocol):
    """Name the entries of STalign's result this module reads."""

    @overload
    def __getitem__(
        self, key: Literal["A", "v", "WM", "WB", "WA"]
    ) -> torch.Tensor: ...

    @overload
    def __getitem__(self, key: Literal["xv"]) -> list[torch.Tensor]: ...

    @overload
    def __getitem__(
        self, key: Literal["settings", "prefit"]
    ) -> Mapping[str, Option]: ...


class _STalign(Protocol):
    """Name the functions of STalign this module calls (it ships no types)."""

    def LDDMM(  # noqa: N802 - STalign's name
        self,
        source_axes: list[np.ndarray],
        source: np.ndarray,
        target_axes: list[np.ndarray],
        target: np.ndarray,
        /,
        **_options: object,
    ) -> _Fit: ...

    def LDDMM_3D_to_slice(  # noqa: N802 - STalign's name
        self,
        source_axes: list[np.ndarray],
        source: np.ndarray,
        target_axes: list[np.ndarray],
        target: np.ndarray,
        /,
        **_options: object,
    ) -> _Fit: ...

    def build_transform(
        self,
        xv: list[torch.Tensor],
        v: torch.Tensor,
        A: torch.Tensor,  # noqa: N803 - STalign's name
        *,
        direction: str,
        XJ: torch.Tensor,  # noqa: N803 - STalign's name
    ) -> torch.Tensor: ...


class _Figures(Protocol):
    """Name what STalign asks of its end-of-fit figures."""

    def update(self, snap: Mapping[str, np.ndarray]) -> None: ...


#: How STalign makes its figures: its class, called with the fit's problem.
FiguresMaker = Callable[[object], _Figures]


class _FiguresModule(Protocol):
    """Name the module that holds STalign's figures class."""

    _Figures: FiguresMaker


class _Records(threading.local):
    """Hold the energy record of the fit running on this thread."""

    def __init__(self) -> None:
        #: The fit's record, or None on a thread that runs no fit here.
        self.energy: list[np.ndarray] | None = None


_RECORDS = _Records()
#: Serializes installing the stand-in for STalign's figures.
_INSTALL = threading.Lock()


def run(
    arrays: Mapping[str, np.ndarray], params: Mapping[str, Option]
) -> dict[str, Output]:
    """Run one STalign call on the inputs the plugin prepared.

    Parameters
    ----------
    arrays
        The call's inputs by name (see the module text); ``mode`` picks
        the call.
    params
        STalign's keyword arguments, with ``device`` (``auto``, ``cpu`` or
        ``cuda``) and ``threads``.

    Returns
    -------
    dict of str to Output
        The call's results by name (see the module text).
    """
    import torch  # local: slow to import, needed only when a fit runs

    options = dict(params)
    device = resolve_device(str(options.pop("device", AUTO)))
    if not on_gpu(device) or str(arrays["mode"]) == PLANE:
        # The plane search runs no fit iterations: its multi-start
        # pre-alignment is CPU work in float64 wherever the fit would run, so
        # it takes no GPU turn and several sections search at once.
        device = CPU
    # the GPU at float32, the CPU at float64 (STalign's C++ engine)
    dtype = torch.float32 if on_gpu(device) else torch.float64
    threads = int(str(options.pop("threads", 4)))
    precision = str(dtype).replace("torch.", "")
    out: dict[str, Output] = {"device": f"{device} {precision}"}
    started = time.monotonic()
    stalign = _stalign()
    how = _How(device, dtype, threads)
    if str(arrays["mode"]) == PLANE:
        out.update(_plane(stalign, arrays, options, how))
    else:
        out.update(_lddmm(stalign, arrays, options, how))
    out["seconds"] = time.monotonic() - started
    return out


@dataclass(frozen=True)
class _How:
    """Hold where and how a fit runs.

    Parameters
    ----------
    device
        The torch device the fit runs on.
    dtype
        The precision of the fit's tensors.
    threads
        The threads STalign may use.
    """

    device: str
    dtype: torch.dtype
    threads: int


def _stalign() -> _STalign:
    """Import STalign, its end-of-fit figures replaced by the record."""
    module = importlib.import_module(PACKAGE)
    figures = cast(  # STalign ships no types: the module of its figures
        "_FiguresModule", importlib.import_module(f"{PACKAGE}._lddmm_common")
    )
    with _INSTALL:
        current = figures._Figures  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - STalign's figures hook
        if current is not _EnergyRecord:
            _EnergyRecord.figures = current
            figures._Figures = _EnergyRecord  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - STalign's figures hook
    return cast("_STalign", module)  # STalign ships no types: its functions


class _EnergyRecord:
    """Stand in for STalign's end-of-fit figures (``draw='end'``).

    On a thread that runs a fit for LangSlice it draws nothing and keeps
    the energy record the figures would plot; on any other thread it draws
    them as STalign does.

    Parameters
    ----------
    problem
        STalign's description of the fit, which its own figures read.
    """

    #: STalign's own figures class, which another caller's fit draws with.
    figures: ClassVar[FiguresMaker | None] = None

    def __init__(self, problem: object) -> None:
        self._energy = _RECORDS.energy
        self._figures: _Figures | None = None
        if self._energy is None and _EnergyRecord.figures is not None:
            self._figures = _EnergyRecord.figures(problem)

    def update(self, snap: Mapping[str, np.ndarray]) -> None:
        """Keep the record, or draw the figures for another caller.

        Parameters
        ----------
        snap
            STalign's state after the last iteration; ``Esave`` is its
            energy per iteration (a tensor on the fit's device, or an
            array).
        """
        if self._figures is not None:
            self._figures.update(snap)
        elif self._energy is not None:
            self._energy[:] = [_plain(snap["Esave"])]


def _plain(value: object) -> np.ndarray:
    """Copy a tensor or array as a float64 array on the CPU."""
    import torch  # local: slow to import, needed only when a fit runs

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, np.float64)


def _json(value: Mapping[str, Option]) -> str:
    """Write STalign's records as JSON (tensors and arrays as lists)."""
    return json.dumps(value, default=_listed)


def _listed(item: object) -> object:
    """Turn a tensor or array into nested lists, for JSON."""
    # numpy types tolist() as Any: nested lists
    return cast("object", _plain(item).tolist())


def _lddmm(
    stalign: _STalign,
    arrays: Mapping[str, np.ndarray],
    options: Mapping[str, Option],
    how: _How,
) -> dict[str, Output]:
    """Run STalign's 2-D fit and sample its map at the query points."""
    import torch  # local: slow to import, needed only when a fit runs

    start = arrays["A0"]
    energy: list[np.ndarray] = []
    _RECORDS.energy = energy
    out: dict[str, Output] = {}
    try:
        with gpu_turn(how.device) as turn:
            out["gpu_wait_s"] = turn.waited_s
            result = stalign.LDDMM(
                [arrays["xI0"], arrays["xI1"]],
                arrays["I"],
                [arrays["xJ0"], arrays["xJ1"]],
                arrays["J"],
                L=start[:2, :2].copy(),
                T=start[:2, 2].copy(),
                prefit=False,
                draw="end",
                device=how.device,
                dtype=how.dtype,
                num_threads=how.threads,
                **options,
            )
            velocity = result["v"]
            query = np.asarray(arrays["query"], np.float64)
            sampled: list[np.ndarray] = []
            for first in range(0, len(query), QUERY_CHUNK):
                points = torch.as_tensor(
                    query[first : first + QUERY_CHUNK],
                    dtype=velocity.dtype,
                    device=velocity.device,
                )[:, None]
                with torch.no_grad():
                    mapped = stalign.build_transform(
                        result["xv"],
                        velocity,
                        result["A"],
                        direction="b",
                        XJ=points,
                    )
                sampled.append(_plain(mapped)[:, 0])
    finally:
        _RECORDS.energy = None
    out["source_um"] = np.zeros((0, 2))
    if sampled:
        out["source_um"] = np.concatenate(sampled, 0)
    out["A"] = _plain(result["A"])
    out["WM"] = _plain(result["WM"])
    out["WB"] = _plain(result["WB"])
    out["WA"] = _plain(result["WA"])
    out["energy"] = energy[0] if energy else np.zeros((0, 3))
    out["settings"] = _json(result["settings"])
    return out


def _plane(
    stalign: _STalign,
    arrays: Mapping[str, np.ndarray],
    options: Mapping[str, Option],
    how: _How,
) -> dict[str, Output]:
    """Run STalign's 3-D plane search and its refinement, no warp."""
    atlas = averaged_image(str(arrays["atlas"]), str(arrays["image"]))
    image = atlas.image[None] / np.mean(np.abs(atlas.image))
    image = np.concatenate((image, (image - image.mean()) ** 2))
    out: dict[str, Output] = {}
    with gpu_turn(how.device) as turn:
        out["gpu_wait_s"] = turn.waited_s
        result = stalign.LDDMM_3D_to_slice(
            atlas.axes,
            image,
            [arrays["xJ0"], arrays["xJ1"]],
            arrays["J"],
            A=arrays["A0"],
            niter=0,
            prefit=True,
            draw=False,
            device=how.device,
            dtype=how.dtype,
            num_threads=how.threads,
            **options,
        )
    prefit = {
        key: value for key, value in result["prefit"].items() if key != "A"
    }
    out["A"] = _plain(result["A"])
    out["prefit"] = _json(prefit)
    out["settings"] = _json(result["settings"])
    return out
