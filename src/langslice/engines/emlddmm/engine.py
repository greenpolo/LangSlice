"""Call emlddmm in LangSlice's process, for the plugin (``plugin.py``).

emlddmm (Apache-2.0) is vendored as ``langslice_vendor.emlddmm.emlddmm``:
twardlab/emlddmm @ 9c4dc80 plus our fork's e426ece, which repairs
``out_of_plane=False``, and 6ff62b0, which adds ``lock_slice_axis``; its
header says what LangSlice changed to import it here. It is imported when a
tool runs, never when the plugin loads. The plugin prepares every image and
the stack's frame; this module only builds emlddmm's call, runs it and reads
its result at the points the plugin asks for.

The inputs: ``J`` (slots x rows x cols: the sections on the stack grid, in
[0, 1]), ``W0`` (the same shape: 1 where a section holds data, 0 on empty
slots, outside the image and on marked damage), ``zs``, ``gy``, ``gx`` (the
slot, row and column coordinates, um, the stack frame), ``R`` (3 x 3, rows:
the stack's slot axis, row axis, column axis in atlas um) and ``c0`` (the
atlas point at the stack's origin, um), ``slot`` (the slot index of each
section), ``q`` (N x 2: stack (row, col) um where the map is read) with
``q_n`` (the number of points of each section).

``params``: ``atlas``, ``image`` (``template``, one of the atlas's additional
references, or the borrowed Nissl), ``device``, ``dtype``, ``mode``
(``stack``: the slots' planes locked; ``plane``: the plane free, no warp),
``cfg`` (emlddmm's keyword arguments per scale, lists as the example's
config file), ``sorting`` (``match_sd``, ``background_sd``, ``artifact_sd``,
``background_mean``, ``artifact_mean``), ``up_vector`` (``plane``), ``warp``
(read the 3-D warp too), ``mu`` (``muB``, and ``muA`` on a white slide).

The result: ``A`` (4 x 4), ``A0``, ``A2d`` (slots x 3 x 3), ``WM`` / ``WB`` /
``WA`` (the final sorting, half precision), ``Acut`` (sections x rows x
cols: the atlas image under every stack pixel of each section, through the
whole fit), ``X`` / ``L`` (N x 3: the atlas um at the points ``q``, with and
without the warp), ``energy`` (iterations x [scale, iteration, total,
mismatch, regularization]: emlddmm's own record) with ``energy_captured``
(1 when it has one row per iteration run), ``settings`` (JSON: what emlddmm
chose), ``log`` (what emlddmm printed), ``device``, ``peak_gpu_gb``,
``seconds`` and ``gpu_wait_s``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np

from langslice.engines._atlas_images import averaged_image
from langslice.engines._devices import on_gpu, resolve_device
from langslice.plugin_api import gpu_turn

if TYPE_CHECKING:
    import torch

MODULE = "langslice_vendor.emlddmm.emlddmm"
PLANE = "plane"
SINGLE = "float32"
SORTING = {
    "match_sd": "sigmaM",
    "background_sd": "sigmaB",
    "artifact_sd": "sigmaA",
    "background_mean": "muB",
    "artifact_mean": "muA",
}
CALLER_DEPTH = 4
FIT_FUNCTION = "emlddmm"

Setting = (
    str | bool | float | Sequence[float] | Mapping[str, float] | Mapping[str, Sequence[object]]
)
Output = np.ndarray | np.int32 | str | float

_LOG = logging.getLogger(__name__)


class _Fit(Protocol):
    """Name the entries of an emlddmm result this module reads."""

    @overload
    def __getitem__(self, key: Literal["A", "A2d", "v", "WM", "WB", "WA"]) -> torch.Tensor: ...

    @overload
    def __getitem__(self, key: Literal["xv"]) -> list[torch.Tensor]: ...

    @overload
    def __getitem__(
        self, key: Literal["muA", "muB", "sigmaM", "sigmaB", "sigmaA"]
    ) -> torch.Tensor | float: ...


class _Emlddmm(Protocol):
    """Name the parts of emlddmm this module uses (it ships no types)."""

    interp: Callable[..., torch.Tensor]
    print: Callable[..., None]

    def emlddmm_multiscale(self, **_config: object) -> list[_Fit]: ...

    def v_to_phii(self, xv: list[torch.Tensor], v: torch.Tensor) -> torch.Tensor: ...


class _Calls(threading.local):
    """Hold what the fit running on this thread keeps."""

    def __init__(self) -> None:
        self.saved: list[list[list[float]]] | None = None
        self.printed: list[str] | None = None


_CALLS = _Calls()
_INSTALL = threading.Lock()


@dataclass(frozen=True)
class _Stack:
    """Hold the stack's frame as emlddmm fits it.

    Parameters
    ----------
    start
        The stack's starting affine (4 x 4, atlas um about the volume's
        centre to stack um).
    centre
        The atlas volume's centre, um.
    axes
        The atlas image's axes about its centre, um.
    image
        The atlas image over its mean, with a leading channel axis.
    """

    start: np.ndarray
    centre: np.ndarray
    axes: list[np.ndarray]
    image: np.ndarray


def run(arrays: Mapping[str, np.ndarray], params: Mapping[str, Setting]) -> dict[str, Output]:
    """Run one emlddmm fit on the stack the plugin prepared.

    Parameters
    ----------
    arrays
        The stack and the points to read (see the module text).
    params
        The fit's settings (see the module text).

    Returns
    -------
    dict of str to Output
        The fit's results by name (see the module text).
    """
    import torch

    emlddmm = _emlddmm()
    started = time.monotonic()
    device = resolve_device(str(params["device"]))
    single = params["dtype"] == SINGLE
    dtype = torch.float32 if single else torch.float64
    precision = str(dtype).replace("torch.", "")
    out: dict[str, Output] = {"device": f"{device} {precision}"}
    stack = _stack(arrays, params)
    cfg = _per_scale(params)
    plane = params["mode"] == PLANE
    grid = [np.asarray(arrays[key], np.float64) for key in ("zs", "gy", "gx")]
    config: dict[str, object] = {
        "I": stack.image,
        "xI": [stack.axes],
        "J": np.asarray(arrays["J"], np.float64)[None],
        "xJ": [grid],
        "W0": np.asarray(arrays["W0"], np.float64),
        "device": device,
        "dtype": dtype,
        **cast("Mapping[str, object]", params["mu"]),
        **cfg,
        "lock_slice_axis": [not plane],
        "slice_matching": [True],
        "out_of_plane": [False],
        "A": stack.start,
        "n_draw": 0,
        "full_outputs": True,
    }
    if plane:
        up = cast("Sequence[float]", params["up_vector"])
        config["up_vector"] = [[float(v) for v in up]]
    saved: list[list[list[float]]] = []
    printed: list[str] = []
    _CALLS.saved, _CALLS.printed = saved, printed
    try:
        with gpu_turn(device) as turn:
            out["gpu_wait_s"] = turn.waited_s
            if on_gpu(device):
                torch.cuda.reset_peak_memory_stats()
            result = emlddmm.emlddmm_multiscale(**config)[-1]
            out["peak_gpu_gb"] = 0.0
            if on_gpu(device):
                out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    finally:
        _CALLS.saved, _CALLS.printed = None, None
    out.update(_read_map(emlddmm, result, arrays, stack, params))
    energy = _energy(saved)
    out["energy"] = energy
    steps = sum(int(n) for n in cast("list[int]", cfg["n_iter"]))
    out["energy_captured"] = np.int32(len(energy) == steps)
    out["settings"] = _json(
        {
            "muA": result["muA"],
            "muB": result["muB"],
            "sigmaM": result["sigmaM"],
            "sigmaB": result["sigmaB"],
            "sigmaA": result["sigmaA"],
            "cfg": cfg,
        }
    )
    out["log"] = "".join(printed)
    out["seconds"] = time.monotonic() - started
    return out


def _stack(arrays: Mapping[str, np.ndarray], params: Mapping[str, Setting]) -> _Stack:
    """Read the atlas image and place the stack on it."""
    atlas = averaged_image(str(params["atlas"]), str(params["image"]))
    image = atlas.image[None].astype(np.float64)
    image /= max(float(np.mean(np.abs(image))), 1e-12)
    centre = _numbers(atlas.centre)
    axes = [axis - c for axis, c in zip(atlas.axes, centre, strict=True)]
    rotation = np.asarray(arrays["R"], np.float64)
    origin = np.asarray(arrays["c0"], np.float64)
    start = np.eye(4)
    start[:3, :3] = rotation
    start[:3, 3] = -rotation @ (origin - atlas.centre)
    return _Stack(start, atlas.centre, axes, image)


def _per_scale(params: Mapping[str, Setting]) -> dict[str, list[object]]:
    """Build emlddmm's keyword arguments per scale, sorting included."""
    given = cast("Mapping[str, Sequence[object]]", params["cfg"])
    cfg = {key: list(value) for key, value in given.items()}
    sorting = cast("Mapping[str, float]", params["sorting"])
    for key, name in SORTING.items():
        if key in sorting:
            cfg[name] = [[float(sorting[key])]]
    return cfg


def _read_map(
    emlddmm: _Emlddmm,
    result: _Fit,
    arrays: Mapping[str, np.ndarray],
    stack: _Stack,
    params: Mapping[str, Setting],
) -> dict[str, Output]:
    """Read the fit at the stack's pixels and at the plugin's points."""
    import torch

    cpu = torch.float64
    affine = result["A"].detach().cpu().to(dtype=cpu)
    locked = params["mode"] != PLANE
    if locked and not torch.allclose(affine[0], torch.as_tensor(stack.start)[0]):
        raise RuntimeError("the locked slice axis moved")
    turns = result["A2d"].detach().cpu().to(dtype=cpu)
    xv = [torch.as_tensor(x).detach().cpu().to(dtype=cpu) for x in result["xv"]]
    velocity = torch.as_tensor(result["v"]).detach().cpu().to(dtype=cpu)
    with torch.no_grad():
        inverse = emlddmm.v_to_phii(xv, velocity)
        identity = torch.stack(torch.meshgrid(xv, indexing="ij"))
    reader = _Reader(
        emlddmm,
        _numbers(np.asarray(arrays["zs"], np.float64)),
        turns,
        affine.inverse(),
        (xv, inverse - identity),
    )
    warp = bool(params["warp"])
    slots = _integers(arrays["slot"])
    out: dict[str, Output] = {"Acut": _atlas_cut(emlddmm, reader, arrays, stack, warp)}
    queries = np.asarray(arrays["q"], np.float64)
    counts = _integers(arrays["q_n"])
    shift = torch.as_tensor(stack.centre, dtype=cpu)
    warped: list[np.ndarray] = [np.zeros((0, 3), np.float32)]
    linear: list[np.ndarray] = [np.zeros((0, 3), np.float32)]
    first = 0
    for slot, n in zip(slots, counts, strict=True):
        part = queries[first : first + n]
        first += n
        if n:
            warped.append((reader.atlas_um(part, slot, warp) + shift).numpy().astype(np.float32))
            linear.append((reader.atlas_um(part, slot, False) + shift).numpy().astype(np.float32))
    out["X"] = np.concatenate(warped, 0)
    out["L"] = np.concatenate(linear, 0)
    out["A"], out["A0"], out["A2d"] = (
        affine.numpy(),
        stack.start,
        turns.numpy(),
    )
    out["WM"] = _half(result["WM"])
    out["WB"] = _half(result["WB"])
    out["WA"] = _half(result["WA"])
    return out


def _half(weights: torch.Tensor) -> np.ndarray:
    """Copy a weight map to the CPU at half precision."""
    return weights.detach().cpu().numpy().astype(np.float16)


def _atlas_cut(
    emlddmm: _Emlddmm,
    reader: _Reader,
    arrays: Mapping[str, np.ndarray],
    stack: _Stack,
    warp: bool,
) -> np.ndarray:
    """Sample the atlas image under every stack pixel of each section."""
    import torch

    cpu = torch.float64
    gy, gx = (np.asarray(arrays[key], np.float64) for key in ("gy", "gx"))
    slots = _integers(arrays["slot"])
    rows, cols = np.meshgrid(gy, gx, indexing="ij")
    pixels = np.stack([rows.ravel(), cols.ravel()], -1)
    atlas = torch.as_tensor(stack.image, dtype=cpu)
    axes = [torch.as_tensor(x, dtype=cpu) for x in stack.axes]
    cut = np.zeros((len(slots), len(gy), len(gx)), np.float32)
    for i, slot in enumerate(slots):
        points = reader.atlas_um(pixels, slot, warp)
        sampled = emlddmm.interp(axes, atlas, points.T.reshape(3, 1, 1, -1))
        cut[i] = sampled.reshape(len(gy), len(gx)).numpy()
    return cut


class _Reader:
    """Carry stack points to atlas um as the fit's own loop samples them.

    A target point goes through its slot's in-plane turn and shift
    (``A2d``), the stack's affine ``A``, then the 3-D warp.

    Parameters
    ----------
    emlddmm
        The vendored module, whose ``interp`` samples the warp.
    zs
        Each slot's coordinate along the slicing axis, um.
    turns
        Each slot's in-plane affine ``A2d`` (slots x 3 x 3).
    inverse_affine
        The inverse of the stack's affine ``A`` (4 x 4).
    warp
        The velocity grid's axes and the warp's displacement on them.
    """

    def __init__(
        self,
        emlddmm: _Emlddmm,
        zs: list[float],
        turns: torch.Tensor,
        inverse_affine: torch.Tensor,
        warp: tuple[list[torch.Tensor], torch.Tensor],
    ) -> None:
        self._emlddmm = emlddmm
        self._zs = zs
        self._turns = turns
        self._inverse = inverse_affine
        self._warp = warp

    def atlas_um(self, points: np.ndarray, slot: int, with_warp: bool) -> torch.Tensor:
        """Map stack (row, col) um on one slot to atlas um.

        Parameters
        ----------
        points
            The points, N x 2, (row, col) um in the stack frame.
        slot
            The slot the points lie on.
        with_warp
            Whether the 3-D warp is applied after the affine.

        Returns
        -------
        torch.Tensor
            N x 3 atlas um about the volume's centre, float64 on the CPU.
        """
        import torch

        cpu = torch.float64
        n = len(points)
        stacked = torch.stack(
            [
                torch.full((n,), self._zs[slot], dtype=cpu),
                torch.as_tensor(points[:, 0], dtype=cpu),
                torch.as_tensor(points[:, 1], dtype=cpu),
            ]
        )
        turn = self._turns[slot].inverse()
        stacked[1:] = turn[:2, :2] @ stacked[1:] + turn[:2, 2:3]
        atlas = self._inverse[:3, :3] @ stacked + self._inverse[:3, 3:4]
        if with_warp:
            axes, displacement = self._warp
            with torch.no_grad():
                moved = self._emlddmm.interp(axes, displacement, atlas.reshape(3, 1, 1, -1))
            atlas = moved.reshape(3, -1) + atlas
        return atlas.T


def _emlddmm() -> _Emlddmm:
    """Import the vendored module, its ``interp`` and ``print`` hooked."""
    module = cast("_Emlddmm", importlib.import_module(MODULE))
    with _INSTALL:
        if not isinstance(module.interp, _EnergyHook):
            module.interp = _EnergyHook(module.interp)
            module.print = _print
    return module


class _EnergyHook:
    """Keep emlddmm's energy record without editing it.

    ``Esave`` is a local list of ``emlddmm.emlddmm`` (total, mismatch and
    regularization per iteration) that only its figures read; ``interp``
    is a module function it calls every iteration, so this stand-in finds
    the calling ``emlddmm`` frame and keeps a reference to that list, on
    the thread's own record.

    Parameters
    ----------
    interp
        emlddmm's own ``interp``, which every call goes on to.
    """

    def __init__(self, interp: Callable[..., torch.Tensor]) -> None:
        self._interp = interp

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Note the caller's ``Esave``, then interpolate as emlddmm does.

        Parameters
        ----------
        *args
            emlddmm's positional arguments.
        **kwargs
            emlddmm's keyword arguments.

        Returns
        -------
        torch.Tensor
            What emlddmm's ``interp`` returns.
        """
        saved = _CALLS.saved
        if saved is not None:
            here = inspect.currentframe()
            caller = None if here is None else here.f_back
            energy = _caller_energy(caller)
            if energy is not None and not any(energy is kept for kept in saved):
                saved.append(energy)
        return self._interp(*args, **kwargs)


def _caller_energy(frame: FrameType | None) -> list[list[float]] | None:
    """Find the ``Esave`` list of the ``emlddmm`` frame that called."""
    for _ in range(CALLER_DEPTH):
        if frame is None:
            return None
        if frame.f_code.co_name == FIT_FUNCTION:
            energy = frame.f_locals.get("Esave")
            if isinstance(energy, list):
                return cast("list[list[float]]", energy)
            return None
        frame = frame.f_back
    return None


def _print(*args: object, sep: str = " ", end: str = "\n", **_options: object) -> None:
    """Keep what emlddmm prints with the fit that printed it.

    A thread with no fit here logs it instead.
    """
    line = sep.join(str(arg) for arg in args) + end
    printed = _CALLS.printed
    if printed is None:
        _LOG.info("emlddmm: %s", line.rstrip())
    else:
        printed.append(line)


def _energy(saved: list[list[list[float]]]) -> np.ndarray:
    """Lay emlddmm's records out as rows, one per iteration.

    Each row is [scale, iteration, total, mismatch, regularization].
    """
    rows: list[list[float]] = []
    for scale, energy in enumerate(saved):
        for i, row in enumerate(energy):
            rows.append([scale, i, *[float(v) for v in row[:3]]])
    return np.asarray(rows, float).reshape(-1, 5)


def _json(value: Mapping[str, Setting | torch.Tensor]) -> str:
    """Write emlddmm's records as JSON (tensors and arrays as lists)."""
    return json.dumps(value, default=_plain)


def _plain(value: object) -> object:
    """Turn a tensor or array into nested lists."""
    import torch

    array = np.asarray(value)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().double().numpy()
    return cast("object", array.tolist())


def _numbers(values: np.ndarray) -> list[float]:
    """Read a 1-D array as Python floats."""
    return cast("list[float]", np.asarray(values, np.float64).tolist())


def _integers(values: np.ndarray) -> list[int]:
    """Read a 1-D array as Python integers."""
    return cast("list[int]", np.asarray(values, int).tolist())
