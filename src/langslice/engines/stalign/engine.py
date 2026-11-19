"""Call STalign 2.0 in LangSlice's process, for the plugin (``plugin.py``).

STalign 2.0.0 (JEFworks-Lab main 8a84bf1f, GPL-3.0) is vendored as
``langslice._vendor.stalign`` (its pure-torch engine; the optional C++ engine
is not included) and imported when a tool runs, never when the plugin
loads. The plugin prepares every image and frame; this module only calls
STalign and samples its result.

Two calls, each a frozen request and a frozen result:

- :func:`run_lddmm` (:class:`LddmmRequest`): 2-D ``STalign.LDDMM``. The source
  (C x h x w) sits on its pixel centres, the target (1 x H x W) on its own, a
  start affine takes source um to target um (row/col order), and the
  result holds, for every query point (target um, row/col), the source point
  STalign's map takes it to.
- :func:`run_plane_search` (:class:`PlaneRequest`): 3-D
  ``STalign.LDDMM_3D_to_slice`` with ``niter=0``: the plane search and its
  refinement only, on the BrainGlobe atlas image the request names, read at
  50 um (the published atlas voxel, ``_atlas_average.py``), with the start
  affine taking atlas um to target um, target = (0, row um, col um).

Both run on the CPU for the plane search; the 2-D fit runs on the GPU at
float32 when the settings choose one, else on the CPU at float64.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np

from langslice.engines._atlas_average import averaged_image
from langslice.engines._devices import CPU, on_gpu, resolve_device
from langslice.engines._sorting import Sorting
from langslice.plugin_api import gpu_turn

if TYPE_CHECKING:
    import torch

PACKAGE = "langslice._vendor.stalign"
QUERY_CHUNK = 1 << 20
BACKWARD = "b"


class _Fit(Protocol):
    """Name the entries of STalign's result this module reads."""

    @overload
    def __getitem__(self, key: Literal["A", "v", "WM", "WB", "WA", "Esave"]) -> torch.Tensor: ...

    @overload
    def __getitem__(self, key: Literal["xv"]) -> list[torch.Tensor]: ...

    @overload
    def __getitem__(self, key: Literal["settings", "prefit"]) -> Mapping[str, object]: ...


class _STalign(Protocol):
    """Name the functions of STalign this module calls (it ships no types)."""

    def LDDMM(
        self,
        source_axes: list[np.ndarray],
        source: np.ndarray,
        target_axes: list[np.ndarray],
        target: np.ndarray,
        /,
        **_options: object,
    ) -> _Fit: ...

    def LDDMM_3D_to_slice(
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
        A: torch.Tensor,
        *,
        direction: str,
        XJ: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class Machine:
    """Hold where and how a fit runs.

    Parameters
    ----------
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    threads
        The threads STalign may use.
    """

    device: str
    threads: int


@dataclass(frozen=True)
class LddmmSettings:
    """Hold STalign's keyword arguments for the 2-D fit.

    Parameters
    ----------
    kernel_um
        The width of the kernel that smooths the velocity (``a``), um.
    displacement_um
        The displacement scale (``sigma_disp``), um.
    iterations
        Gradient-descent steps.
    time_steps
        The time steps the velocity is integrated over (``nt``).
    refit_transform
        Whether the in-plane transform is fitted too; False holds it at the
        start affine (``epL`` and ``epT`` zero).
    sorting
        The values set for the pixel classes' spreads and brightnesses.
    mixture_classes
        The classes of STalign's mixture (``MAB``, ``MB`` or ``M``), or None
        for its published sorting.
    """

    kernel_um: float
    displacement_um: float
    iterations: int
    refit_transform: bool
    sorting: Sorting = field(default_factory=Sorting)
    mixture_classes: str | None = None
    time_steps: int = 4

    def keywords(self) -> dict[str, object]:
        """Return the keyword arguments STalign's ``LDDMM`` takes."""
        keywords: dict[str, object] = {
            **self.sorting.keywords(),
            "a": self.kernel_um,
            "sigma_disp": self.displacement_um,
            "niter": self.iterations,
            "nt": self.time_steps,
        }
        if self.mixture_classes is not None:
            keywords["mixture"] = {"classes": self.mixture_classes}
        if not self.refit_transform:
            keywords["epL"] = 0.0
            keywords["epT"] = 0.0
        return keywords


@dataclass(frozen=True)
class LddmmRequest:
    """Hold one 2-D fit.

    Parameters
    ----------
    source
        The atlas image, ``(channels, h, w)``.
    source_rows_um, source_cols_um
        Its pixel centres, um.
    target
        The section image, ``(1, H, W)``.
    target_rows_um, target_cols_um
        Its pixel centres, um.
    start
        The start affine (3 x 3), source um to target um, row/col order.
    query_um
        Where the map is read, ``(N, 2)`` target um, row/col.
    settings
        STalign's keyword arguments.
    machine
        Where the fit runs.
    """

    source: np.ndarray
    source_rows_um: np.ndarray
    source_cols_um: np.ndarray
    target: np.ndarray
    target_rows_um: np.ndarray
    target_cols_um: np.ndarray
    start: np.ndarray
    query_um: np.ndarray
    settings: LddmmSettings
    machine: Machine


@dataclass(frozen=True)
class LddmmResult:
    """Hold what a 2-D fit gives back.

    Parameters
    ----------
    source_um
        The source point STalign's map takes each query point to, ``(N, 2)``.
    affine
        The fitted affine (3 x 3).
    match_weights, background_weights, artifact_weights
        STalign's final sorting of the target's pixels.
    energy
        Its record per iteration: total, mismatch, smoothness.
    chosen
        The settings STalign chose (its own record, plain values).
    device
        The device and precision the fit ran at.
    seconds
        How long the call took.
    gpu_wait_s
        How long it waited for the GPU queue.
    """

    source_um: np.ndarray
    affine: np.ndarray
    match_weights: np.ndarray
    background_weights: np.ndarray
    artifact_weights: np.ndarray
    energy: np.ndarray
    chosen: Mapping[str, object]
    device: str
    seconds: float
    gpu_wait_s: float


@dataclass(frozen=True)
class PlaneSettings:
    """Hold STalign's keyword arguments for the plane search.

    Parameters
    ----------
    time_steps
        The time steps the velocity is integrated over (``nt``).
    sigma_match, sigma_background, sigma_artifact
        The three classes' spreads.
    mu_artifact, mu_background
        The artifact and background classes' brightnesses.
    """

    time_steps: int = 4
    sigma_match: float = 2.0
    sigma_background: float = 2.0
    sigma_artifact: float = 2.0
    mu_artifact: float = 3.0
    mu_background: float = 0.0


@dataclass(frozen=True)
class PlaneRequest:
    """Hold one plane search.

    Parameters
    ----------
    atlas
        The BrainGlobe atlas name.
    atlas_image
        The atlas image searched with (``template``, one of its additional
        references or the borrowed Nissl).
    target
        The section image, ``(1, H, W)``.
    target_rows_um, target_cols_um
        Its pixel centres, um.
    start
        The start affine (4 x 4), atlas um to target um, target = (0, row
        um, col um).
    settings
        STalign's keyword arguments.
    machine
        The threads the search may use (it runs on the CPU).
    """

    atlas: str
    atlas_image: str
    target: np.ndarray
    target_rows_um: np.ndarray
    target_cols_um: np.ndarray
    start: np.ndarray
    settings: PlaneSettings
    machine: Machine


@dataclass(frozen=True)
class PlaneResult:
    """Hold what a plane search gives back.

    Parameters
    ----------
    affine
        The fitted affine (4 x 4), atlas um to target um.
    starts
        Every start's tilts (degrees) and shift (um).
    best
        The winning start's index; None without one.
    prefit
        STalign's whole pre-fit record, less the affine (plain values).
    chosen
        The settings STalign chose (plain values).
    device
        The device and precision the search ran at.
    seconds
        How long the call took.
    """

    affine: np.ndarray
    starts: list[list[float]]
    best: int | None
    prefit: Mapping[str, object]
    chosen: Mapping[str, object]
    device: str
    seconds: float


@dataclass(frozen=True)
class _Precision:
    """Hold the device a call runs on and the precision of its tensors."""

    device: str
    dtype: torch.dtype

    @classmethod
    def for_fit(cls, option: str) -> _Precision:
        """Choose the device and precision of a fit on *option*."""
        import torch

        device = resolve_device(option)
        return cls(device, torch.float32 if on_gpu(device) else torch.float64)

    @classmethod
    def on_cpu(cls) -> _Precision:
        """Choose the CPU at float64."""
        import torch

        return cls(CPU, torch.float64)

    def named(self) -> str:
        """Return the device and precision as the reply says them."""
        return f"{self.device} {str(self.dtype).replace('torch.', '')}"


def run_lddmm(request: LddmmRequest) -> LddmmResult:
    """Run STalign's 2-D fit and sample its map at the query points.

    Parameters
    ----------
    request
        The images, the start and the settings.

    Returns
    -------
    LddmmResult
        The map at the query points, the fitted affine, the final sorting
        and the energy record.
    """
    import torch

    precision = _Precision.for_fit(request.machine.device)
    started = time.monotonic()
    stalign = _stalign()
    start = request.start
    with gpu_turn(precision.device) as turn:
        fit = stalign.LDDMM(
            [request.source_rows_um, request.source_cols_um],
            request.source,
            [request.target_rows_um, request.target_cols_um],
            request.target,
            L=start[:2, :2].copy(),
            T=start[:2, 2].copy(),
            prefit=False,
            draw=False,
            device=precision.device,
            dtype=precision.dtype,
            num_threads=request.machine.threads,
            **request.settings.keywords(),
        )
        velocity = fit["v"]
        query = np.asarray(request.query_um, np.float64)
        sampled = [np.zeros((0, 2))]
        for first in range(0, len(query), QUERY_CHUNK):
            points = torch.as_tensor(
                query[first : first + QUERY_CHUNK], dtype=velocity.dtype, device=velocity.device
            )[:, None]
            with torch.no_grad():
                mapped = stalign.build_transform(
                    fit["xv"], velocity, fit["A"], direction=BACKWARD, XJ=points
                )
            sampled.append(_plain(mapped)[:, 0])
    return LddmmResult(
        source_um=np.concatenate(sampled, 0),
        affine=_plain(fit["A"]),
        match_weights=_plain(fit["WM"]),
        background_weights=_plain(fit["WB"]),
        artifact_weights=_plain(fit["WA"]),
        energy=_plain(fit["Esave"]) if request.settings.iterations > 0 else np.zeros((0, 3)),
        chosen=_plain_tree(fit["settings"]),
        device=precision.named(),
        seconds=time.monotonic() - started,
        gpu_wait_s=turn.waited_s,
    )


def run_plane_search(request: PlaneRequest) -> PlaneResult:
    """Run STalign's 3-D plane search and its refinement, no warp.

    Parameters
    ----------
    request
        The atlas image, the section image, the start and the settings.

    Returns
    -------
    PlaneResult
        The fitted affine, the search's starts and its record.
    """
    precision = _Precision.on_cpu()
    started = time.monotonic()
    stalign = _stalign()
    atlas = averaged_image(request.atlas, request.atlas_image)
    image = atlas.image[None] / np.mean(np.abs(atlas.image))
    image = np.concatenate((image, (image - image.mean()) ** 2))
    settings = request.settings
    fit = stalign.LDDMM_3D_to_slice(
        atlas.axes,
        image,
        [request.target_rows_um, request.target_cols_um],
        request.target,
        A=request.start,
        niter=0,
        prefit=True,
        draw=False,
        device=precision.device,
        dtype=precision.dtype,
        num_threads=request.machine.threads,
        nt=settings.time_steps,
        sigmaM=settings.sigma_match,
        sigmaB=settings.sigma_background,
        sigmaA=settings.sigma_artifact,
        muA=settings.mu_artifact,
        muB=settings.mu_background,
    )
    prefit = _plain_tree(fit["prefit"])
    starts = cast("list[list[float]]", prefit.get("starts") or [])
    best = cast("int | None", prefit.get("best"))
    return PlaneResult(
        affine=_plain(fit["A"]),
        starts=starts,
        best=None if best is None else int(best),
        prefit={key: value for key, value in prefit.items() if key != "A"},
        chosen=_plain_tree(fit["settings"]),
        device=precision.named(),
        seconds=time.monotonic() - started,
    )


def _stalign() -> _STalign:
    """Import the vendored STalign."""
    return cast("_STalign", importlib.import_module(PACKAGE))


def _plain(value: object) -> np.ndarray:
    """Copy a tensor or array as a float64 array on the CPU."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, np.float64)


def _plain_tree(value: Mapping[str, object]) -> dict[str, object]:
    """Copy STalign's records with tensors and arrays as nested lists."""
    return {str(key): _listed(item) for key, item in value.items()}


def _listed(item: object) -> object:
    """Turn a tensor, array or numpy number into plain values, in nested records too."""
    import torch

    if isinstance(item, Mapping):
        return _plain_tree(cast("Mapping[str, object]", item))
    if isinstance(item, (list, tuple)):
        return [_listed(part) for part in cast("list[object]", item)]
    if isinstance(item, (torch.Tensor, np.ndarray, np.generic)):
        return cast("object", _plain(item).tolist())
    return item
