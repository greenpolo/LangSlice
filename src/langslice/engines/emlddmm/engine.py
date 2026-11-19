"""Call emlddmm in LangSlice's process, for the plugin (``plugin.py``).

emlddmm (Apache-2.0) is vendored as ``langslice._vendor.emlddmm.emlddmm``:
twardlab/emlddmm @ 9c4dc80 plus our fork's e426ece, which repairs
``out_of_plane=False``, and 6ff62b0, which adds ``lock_slice_axis``; its
header says what LangSlice changed to import it here (an energy callback and a
print sink among them). It is imported when a tool runs, never when the plugin
loads. The plugin prepares every image and the stack's frame; this module only
builds emlddmm's call, runs it and reads its result at the points the plugin
asks for.

:func:`run` takes a :class:`FitRequest` and gives a :class:`FitResult`. The
request holds the stack (:class:`StackImages`: the sections on the stack
grid, in [0, 1], and the weight, 1 where a section holds data, 0 on empty
slots, outside the image and on marked damage), the stack's frame
(:class:`StackFrame`), the atlas image (``atlas`` and ``atlas_image``: a
BrainGlobe atlas and its ``template``, one of its additional references or the
borrowed Nissl), the per-scale schedule (:class:`Schedule`), the sorting, and
the points where each section's map is read. The mode is ``stack`` (the slots'
planes locked) or ``plane`` (the plane free, no warp).
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np

from langslice.engines._atlas_average import averaged_image
from langslice.engines._devices import on_gpu, resolve_device
from langslice.engines._sorting import Sorting
from langslice.plugin_api import gpu_turn

if TYPE_CHECKING:
    import torch

MODULE = "langslice._vendor.emlddmm.emlddmm"
GB = 1e9
Mode = Literal["stack", "plane"]
ENERGY_COLUMNS = 5
Array = np.ndarray


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

    def emlddmm_multiscale(self, **_config: object) -> list[_Fit]: ...

    def v_to_phii(self, xv: list[torch.Tensor], v: torch.Tensor) -> torch.Tensor: ...

    def printing_to(self, lines: list[str]) -> AbstractContextManager[list[str]]: ...


@dataclass(frozen=True)
class Schedule:
    """Hold emlddmm's keyword arguments per scale (the coarse, middle and fine).

    Parameters
    ----------
    iterations
        The steps at each scale.
    kernel_um
        The width of the kernel that smooths the warp's velocity (``a``), um.
    velocity_size_um
        The velocity field's grid spacing (``dv``), um.
    warp_start
        The iteration the warp starts at (``v_start``); a number past the
        steps leaves it off.
    warp_step, affine_step, slice_step
        The gradient-descent steps of the warp (``ev``), the stack's affine
        (``eA``) and each slice's in-plane affine (``eA2d``).
    warp_sigma
        The warp's freedom (``sigmaR``).
    down_atlas, down_stack
        How much each scale downsamples the atlas and the stack, per axis.
    local_contrast
        The block of the contrast fit, in pixels per axis.
    """

    iterations: tuple[int, ...]
    kernel_um: float
    velocity_size_um: float
    warp_start: int
    warp_step: float
    affine_step: float
    slice_step: float
    warp_sigma: float
    down_atlas: tuple[tuple[int, int, int], ...] = ((4, 4, 4), (2, 2, 2), (1, 1, 1))
    down_stack: tuple[tuple[int, int, int], ...] = ((1, 4, 4), (1, 2, 2), (1, 1, 1))
    local_contrast: tuple[int, int, int] = (1, 16, 16)

    def keywords(self) -> dict[str, list[object]]:
        """Return emlddmm's per-scale keyword arguments, lists as its example config."""
        return {
            "downI": [list(scale) for scale in self.down_atlas],
            "downJ": [list(scale) for scale in self.down_stack],
            "n_iter": [int(steps) for steps in self.iterations],
            "a": [float(self.kernel_um)],
            "dv": [self.velocity_size_um],
            "v_start": [self.warp_start],
            "ev": [self.warp_step],
            "eA": [self.affine_step],
            "eA2d": [self.slice_step],
            "local_contrast": [list(self.local_contrast)],
            "sigmaR": [float(self.warp_sigma)],
        }


@dataclass(frozen=True)
class StackImages:
    """Hold the sections on the stack grid.

    Parameters
    ----------
    images
        The sections, ``(slots, rows, cols)``, in [0, 1].
    weights
        The weight of each pixel, the same shape.
    depths_um
        Each slot's coordinate along the slicing axis, um.
    rows_um, cols_um
        The grid's row and column coordinates, um (the stack frame).
    """

    images: np.ndarray
    weights: np.ndarray
    depths_um: Array
    rows_um: Array
    cols_um: Array


@dataclass(frozen=True)
class StackFrame:
    """Hold where the stack stands in the atlas.

    Parameters
    ----------
    rotation
        ``(3, 3)``, rows: the stack's slot axis, row axis, column axis in
        atlas um.
    origin_um
        The atlas point at the stack's origin, um.
    slots
        The slot index of each section.
    """

    rotation: Array
    origin_um: Array
    slots: tuple[int, ...]


@dataclass(frozen=True)
class FitRequest:
    """Hold one emlddmm fit.

    Parameters
    ----------
    atlas, atlas_image
        The BrainGlobe atlas and the image of it fitted to.
    stack
        The sections on the stack grid.
    frame
        The stack's place in the atlas.
    schedule
        emlddmm's per-scale arguments.
    sorting
        The values set for the pixel classes (the rest are emlddmm's own).
    white_slide
        Whether the slide is white: the background mean is 1 and the artifact
        mean 0; else the background mean is 0.
    mode
        ``stack`` (the slots' planes locked) or ``plane`` (the plane free).
    up_vector
        The plane's up direction, for ``plane``.
    read_warp
        Whether the 3-D warp is read as well as the affine.
    points
        Where each section's map is read, one ``(n, 2)`` array of stack
        (row, col) um per section.
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU.
    single_precision
        Whether the fit runs at float32 rather than float64.
    """

    atlas: str
    atlas_image: str
    stack: StackImages
    frame: StackFrame
    schedule: Schedule
    sorting: Sorting
    white_slide: bool
    mode: Mode
    read_warp: bool
    points: tuple[Array, ...]
    device: str
    single_precision: bool
    up_vector: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FitResult:
    """Hold what an emlddmm fit gives back.

    Parameters
    ----------
    affine, start_affine
        The fitted and the starting affine of the stack (4 x 4).
    slice_affines
        Each slot's in-plane affine (slots x 3 x 3).
    match_weights, background_weights, artifact_weights
        The final sorting, ``(slots, rows, cols)``, half precision.
    atlas_cut
        The atlas image under every stack pixel of each section, through the
        whole fit, ``(sections, rows, cols)``.
    warped_um, linear_um
        Per section, the atlas um at its points with and without the warp,
        ``(n, 3)``.
    energy
        emlddmm's own record, ``(iterations, 5)``: scale, iteration, total,
        mismatch, regularization.
    energy_captured
        Whether the record has one row per iteration run.
    chosen
        What emlddmm chose (its means and spreads) and the schedule it ran.
    log
        What emlddmm printed.
    device
        The device and precision the fit ran at.
    peak_gpu_gb
        The most GPU memory the fit held.
    seconds
        How long the call took.
    gpu_wait_s
        How long it waited for the GPU queue.
    """

    affine: Array
    start_affine: Array
    slice_affines: Array
    match_weights: np.ndarray
    background_weights: np.ndarray
    artifact_weights: np.ndarray
    atlas_cut: np.ndarray
    warped_um: tuple[np.ndarray, ...]
    linear_um: tuple[np.ndarray, ...]
    energy: Array
    energy_captured: bool
    chosen: Mapping[str, object]
    log: str
    device: str
    peak_gpu_gb: float
    seconds: float
    gpu_wait_s: float


@dataclass(frozen=True)
class _AtlasStack:
    """Hold the atlas image and the stack's starting affine.

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

    start: Array
    centre: Array
    axes: list[Array]
    image: Array


@dataclass
class _Record:
    """Keep the energy rows a fit reports, one per iteration."""

    rows: list[list[float]] = field(default_factory=list)
    scale: int = -1

    def __call__(self, iteration: int, row: Sequence[float]) -> None:
        """Add one iteration's row; iteration 0 starts the next scale."""
        if iteration == 0:
            self.scale += 1
        self.rows.append([float(self.scale), float(iteration), *row[:3]])

    def array(self) -> Array:
        """Lay the rows out as ``(iterations, 5)``."""
        return np.asarray(self.rows, float).reshape(-1, ENERGY_COLUMNS)


def run(request: FitRequest) -> FitResult:
    """Run one emlddmm fit on the stack the plugin prepared.

    Parameters
    ----------
    request
        The stack, the schedule and the points to read.

    Returns
    -------
    FitResult
        The fit's results.
    """
    import torch

    emlddmm = _emlddmm()
    started = time.monotonic()
    device = resolve_device(request.device)
    dtype = torch.float32 if request.single_precision else torch.float64
    precision = str(dtype).replace("torch.", "")
    atlas = _atlas_stack(request)
    schedule = request.schedule.keywords()
    plane = request.mode == "plane"
    stack = request.stack
    grid = [stack.depths_um, stack.rows_um, stack.cols_um]
    means: dict[str, object] = {"muB": [[float(request.white_slide)]]}
    if request.white_slide:
        means["muA"] = [[0.0]]
    sorted_keywords: dict[str, object] = {
        name: [[value]] for name, value in request.sorting.keywords().items()
    }
    config: dict[str, object] = {
        "I": atlas.image,
        "xI": [atlas.axes],
        "J": np.asarray(stack.images, np.float64)[None],
        "xJ": [grid],
        "W0": np.asarray(stack.weights, np.float64),
        "device": device,
        "dtype": dtype,
        **means,
        **schedule,
        **sorted_keywords,
        "lock_slice_axis": [not plane],
        "slice_matching": [True],
        "out_of_plane": [False],
        "A": atlas.start,
        "n_draw": 0,
        "full_outputs": True,
    }
    if request.up_vector is not None:
        config["up_vector"] = [[float(v) for v in request.up_vector]]
    energy = _Record()
    config["energy_callback"] = energy
    printed: list[str] = []
    peak_gpu_gb = 0.0
    with gpu_turn(device) as turn, emlddmm.printing_to(printed):
        if on_gpu(device):
            torch.cuda.reset_peak_memory_stats()
        fit = emlddmm.emlddmm_multiscale(**config)[-1]
        if on_gpu(device):
            peak_gpu_gb = torch.cuda.max_memory_allocated() / GB
    steps = sum(request.schedule.iterations)
    return _read_fit(
        emlddmm,
        fit,
        request,
        atlas,
        Readout(
            energy=energy.array(),
            energy_captured=len(energy.rows) == steps,
            chosen_schedule=schedule,
            log="".join(printed),
            device=f"{device} {precision}",
            peak_gpu_gb=peak_gpu_gb,
            seconds=time.monotonic() - started,
            gpu_wait_s=turn.waited_s,
        ),
    )


@dataclass(frozen=True)
class Readout:
    """Hold what a fit reports about itself, beside its maps.

    Parameters
    ----------
    energy
        emlddmm's energy record, ``(iterations, 5)``.
    energy_captured
        Whether it has one row per iteration run.
    chosen_schedule
        The per-scale arguments emlddmm ran with.
    log
        What emlddmm printed.
    device
        The device and precision.
    peak_gpu_gb
        The most GPU memory the fit held.
    seconds
        How long the call took.
    gpu_wait_s
        How long it waited for the GPU queue.
    """

    energy: Array
    energy_captured: bool
    chosen_schedule: Mapping[str, object]
    log: str
    device: str
    peak_gpu_gb: float
    seconds: float
    gpu_wait_s: float


def _atlas_stack(request: FitRequest) -> _AtlasStack:
    """Read the atlas image and place the stack on it."""
    atlas = averaged_image(request.atlas, request.atlas_image)
    image = atlas.image[None].astype(np.float64)
    image /= max(float(np.mean(np.abs(image))), 1e-12)
    centre = np.asarray(atlas.centre, np.float64)
    axes = [axis - c for axis, c in zip(atlas.axes, _numbers(centre), strict=True)]
    rotation = np.asarray(request.frame.rotation, np.float64)
    origin = np.asarray(request.frame.origin_um, np.float64)
    start = np.eye(4)
    start[:3, :3] = rotation
    start[:3, 3] = -rotation @ (origin - centre)
    return _AtlasStack(start, centre, axes, image)


def _read_fit(
    emlddmm: _Emlddmm,
    fit: _Fit,
    request: FitRequest,
    atlas: _AtlasStack,
    readout: Readout,
) -> FitResult:
    """Read the fit at the stack's pixels and at the plugin's points."""
    import torch

    double = torch.float64
    affine = fit["A"].detach().cpu().to(dtype=double)
    if request.mode == "stack" and not torch.allclose(affine[0], torch.as_tensor(atlas.start)[0]):
        raise RuntimeError("the locked slice axis moved")
    turns = fit["A2d"].detach().cpu().to(dtype=double)
    xv = [torch.as_tensor(x).detach().cpu().to(dtype=double) for x in fit["xv"]]
    velocity = torch.as_tensor(fit["v"]).detach().cpu().to(dtype=double)
    with torch.no_grad():
        inverse = emlddmm.v_to_phii(xv, velocity)
        identity = torch.stack(torch.meshgrid(xv, indexing="ij"))
    reader = _Reader(
        emlddmm,
        _numbers(np.asarray(request.stack.depths_um, np.float64)),
        turns,
        affine.inverse(),
        (xv, inverse - identity),
    )
    shift = torch.as_tensor(atlas.centre, dtype=double)
    warped: list[np.ndarray] = []
    linear: list[np.ndarray] = []
    for slot, points in zip(request.frame.slots, request.points, strict=True):
        if len(points):
            moved = reader.atlas_um(points, slot, request.read_warp) + shift
            warped.append(moved.numpy().astype(np.float32))
            straight = reader.atlas_um(points, slot, False) + shift
            linear.append(straight.numpy().astype(np.float32))
        else:
            warped.append(np.zeros((0, 3), np.float32))
            linear.append(np.zeros((0, 3), np.float32))
    return FitResult(
        affine=affine.numpy(),
        start_affine=atlas.start,
        slice_affines=turns.numpy(),
        match_weights=_half(fit["WM"]),
        background_weights=_half(fit["WB"]),
        artifact_weights=_half(fit["WA"]),
        atlas_cut=_atlas_cut(emlddmm, reader, request, atlas),
        warped_um=tuple(warped),
        linear_um=tuple(linear),
        energy=readout.energy,
        energy_captured=readout.energy_captured,
        chosen=_chosen(fit, readout.chosen_schedule),
        log=readout.log,
        device=readout.device,
        peak_gpu_gb=readout.peak_gpu_gb,
        seconds=readout.seconds,
        gpu_wait_s=readout.gpu_wait_s,
    )


def _chosen(fit: _Fit, schedule: Mapping[str, object]) -> dict[str, object]:
    """Write what emlddmm chose as plain values."""
    return {
        "muA": _plain(fit["muA"]),
        "muB": _plain(fit["muB"]),
        "sigmaM": _plain(fit["sigmaM"]),
        "sigmaB": _plain(fit["sigmaB"]),
        "sigmaA": _plain(fit["sigmaA"]),
        "cfg": dict(schedule),
    }


def _half(weights: torch.Tensor) -> np.ndarray:
    """Copy a weight map to the CPU at half precision."""
    return weights.detach().cpu().numpy().astype(np.float16)


def _atlas_cut(
    emlddmm: _Emlddmm, reader: _Reader, request: FitRequest, atlas: _AtlasStack
) -> np.ndarray:
    """Sample the atlas image under every stack pixel of each section."""
    import torch

    double = torch.float64
    stack = request.stack
    rows, cols = np.meshgrid(stack.rows_um, stack.cols_um, indexing="ij")
    pixels = np.stack([rows.ravel(), cols.ravel()], -1)
    image = torch.as_tensor(atlas.image, dtype=double)
    axes = [torch.as_tensor(x, dtype=double) for x in atlas.axes]
    cut = np.zeros((len(request.frame.slots), len(stack.rows_um), len(stack.cols_um)), np.float32)
    for i, slot in enumerate(request.frame.slots):
        points = reader.atlas_um(pixels, slot, request.read_warp)
        sampled = emlddmm.interp(axes, image, points.T.reshape(3, 1, 1, -1))
        cut[i] = sampled.reshape(len(stack.rows_um), len(stack.cols_um)).numpy()
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

    def atlas_um(self, points: Array, slot: int, with_warp: bool) -> torch.Tensor:
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

        double = torch.float64
        n = len(points)
        stacked = torch.stack(
            [
                torch.full((n,), self._zs[slot], dtype=double),
                torch.as_tensor(points[:, 0], dtype=double),
                torch.as_tensor(points[:, 1], dtype=double),
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
    """Import the vendored module."""
    return cast("_Emlddmm", importlib.import_module(MODULE))


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
