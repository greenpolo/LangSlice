"""Fit SPOmiAlign's transforms to RoMa's matches, in LangSlice's process.

SPOmiAlign (Wang et al. 2026, Brief Bioinform; wangyiyuyang/SPOmiAlign at
commit c7f8b7a) matches points between two images with RoMa
(``matching.py``) and fits a transform to the matched pairs. Its repository
carries no licence, so none of its code is copied here: each step is
written from what its ``align_and_process_images`` does. The transform the
mode names: ``rigid`` a least-squares rotation and shift (Kabsch),
``affine`` least squares, ``warp`` the displacement field alone,
``affine_warp`` an affine and the field on top of it. The field is
SPOmiAlign's ``bspline`` mode: each pair's displacement on the grid (pairs
that share a pixel averaged), filled in between the pairs by TELEA
inpainting and smoothed by a Gaussian of ``smoothing_px``.

Damage and the section mask are the plugin's: matches whose atlas end lies
in the damage zone (``keep`` False), or whose section end lies where the
section mask is below one half (``section_keep`` False), are dropped before
the ``top_k`` most confident are taken. Coordinates on the way in and out
are canvas pixels (pixel centres); RoMa's grid is corner based.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt

# scipy ships no types for ndimage
from scipy.ndimage import (  # pyright: ignore[reportMissingTypeStubs]
    map_coordinates,  # pyright: ignore[reportUnknownVariableType]
)

from langslice.engines.spomialign.matching import (
    GPU_DEVICE,
    H,
    Matcher,
    W,
    load_matcher,
    ranked_matches,
    resolve_device,
)
from langslice.plugin_api import gpu_turn, out_of_memory

if TYPE_CHECKING:
    import torch

#: RoMa's pixel coordinates are corner based (``W / 2 (x + 1)``): a pixel
#: centre sits half a pixel in.
CORNER = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
#: The engine's modes, as the plugin names them.
MODES = ("rigid", "affine", "warp", "affine_warp")
RIGID, AFFINE, WARP, AFFINE_WARP = MODES
#: TELEA inpainting's radius when the field is filled in, px of the grid.
INPAINT_RADIUS = 3
#: Random canvas points at which the field is checked against SimpleITK's.
N_PROBES = 500
#: A section that still finds the GPU full waits OOM_WAIT_S, twice that,
#: ... and runs again, OOM_RETRIES times.
OOM_RETRIES = 4
OOM_WAIT_S = 5.0

Result = TypeVar("Result")


@dataclass(frozen=True)
class EnginePair:
    """Hold one section's inputs: the two canvases and what may match.

    ``target`` is the atlas image and ``source`` the section, both
    ``(h, w, 3)`` uint8 on one canvas; ``keep`` is ``(h, w)`` bool, False
    in the damage zone; ``section_keep`` is ``(h, w)`` bool, False where
    the section mask is below one half, or None without a mask;
    ``smoothing_px`` is the field's Gaussian sigma, px of the grid.
    """

    target: npt.NDArray[np.uint8]
    source: npt.NDArray[np.uint8]
    keep: npt.NDArray[np.bool_]
    smoothing_px: float
    section_keep: npt.NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class EngineAnswer:
    """Hold what the engine fitted for one section.

    ``matrix`` (2 x 3) takes source canvas px to target canvas px: the
    linear part, the identity for ``warp``; ``residual`` (h x w x 2, the
    warp modes) is the full map minus the linear part at every source
    canvas pixel, target canvas px; ``sitk_max_px`` is the largest
    difference between that field and SimpleITK's own ``TransformPoint``.
    ``used`` and ``dropped_pairs`` are rows atlas x, y, section x, y in
    canvas px and confidence: the pairs fitted, and those passed over to
    reach them. ``error`` says why the section failed, empty when it did
    not; ``oom_retries`` counts the runs again after the GPU ran out of
    memory.
    """

    pairs: int = 0
    dropped: int = 0
    dropped_by_mask: int = 0
    used: npt.NDArray[np.float32] | None = None
    dropped_pairs: npt.NDArray[np.float32] | None = None
    matrix: npt.NDArray[np.float64] | None = None
    residual: npt.NDArray[np.float32] | None = None
    sitk_max_px: float = 0.0
    error: str = ""
    oom_retries: int = 0


@dataclass(frozen=True)
class EngineRun:
    """Hold one call's answers, in the order of its pairs.

    ``gpu_wait_s`` is how long the call waited for the GPU's queue.
    """

    answers: list[EngineAnswer]
    gpu_wait_s: float


def fit_pairs(
    pairs: list[EnginePair], *, mode: str, top_k: int, device_name: str = ""
) -> EngineRun:
    """Match and fit every section's pair with one load of the matcher.

    The whole call, the matcher's loading included, holds the machine's
    GPU queue; the matcher is let go before the queue is.

    Parameters
    ----------
    pairs
        Each section's two canvases and what may match on them.
    mode : {"rigid", "affine", "warp", "affine_warp"}
        The transform each section is fitted with.
    top_k
        How many of the most confident pairs each fit uses.
    device_name
        The torch device to run on; empty means the first GPU when torch
        sees one, else the CPU.

    Returns
    -------
    EngineRun
        One answer per pair; a section that failed carries its error and
        leaves the others.
    """
    import torch  # local: slow to import, needed only when a call runs

    device = resolve_device(device_name)
    queue_device = "cpu"
    if device.type == GPU_DEVICE:
        queue_device = GPU_DEVICE

    with gpu_turn(queue_device) as turn:
        try:
            build = functools.partial(load_matcher, device)
            model, _ = _retried(device, build)
            answers = [
                _fit_one(model, device, pair, mode=mode, top_k=top_k)
                for pair in pairs
            ]
        finally:
            if device.type == GPU_DEVICE:
                torch.cuda.empty_cache()
    return EngineRun(answers=answers, gpu_wait_s=turn.waited_s)


def _retried(
    device: torch.device, call: Callable[[], Result]
) -> tuple[Result, int]:
    """Run *call*, again while a process outside the queue fills the GPU.

    Returns the call's value and how many times it ran again.
    """
    import torch  # local: slow to import, needed only when a call runs

    for attempt in range(OOM_RETRIES + 1):
        try:
            return call(), attempt
        except Exception as error:
            last = attempt == OOM_RETRIES
            if device.type != GPU_DEVICE or last or not out_of_memory(error):
                raise
            torch.cuda.empty_cache()
            time.sleep(OOM_WAIT_S * (attempt + 1))
    raise AssertionError("unreachable")


def _fit_one(
    model: Matcher,
    device: torch.device,
    pair: EnginePair,
    *,
    mode: str,
    top_k: int,
) -> EngineAnswer:
    """Fit one section, its failure kept as its answer's error."""
    fit = functools.partial(
        fit_pair, model, device, pair, mode=mode, top_k=top_k
    )
    try:
        answer, retries = _retried(device, fit)
    except Exception as error:  # noqa: BLE001 - one section must not sink the rest
        return EngineAnswer(error=f"{type(error).__name__}: {error}")
    return dataclasses.replace(answer, oom_retries=retries)


def fit_pair(
    model: Matcher,
    device: torch.device,
    pair: EnginePair,
    *,
    mode: str,
    top_k: int,
) -> EngineAnswer:
    """Match one section's pair and fit the transform *mode* names.

    Parameters
    ----------
    model
        RoMa's matcher, from :func:`load_matcher`.
    device
        The torch device the matcher runs on.
    pair
        The section's two canvases and what may match on them.
    mode : {"rigid", "affine", "warp", "affine_warp"}
        The transform fitted.
    top_k
        How many of the most confident pairs the fit uses.

    Returns
    -------
    EngineAnswer
        The fit, or an error when fewer than three pairs are left.
    """
    h, w = cast("tuple[int, int]", pair.target.shape[:2])
    to_grid = CORNER @ resize_map((w, h), (W, H))
    from_grid = np.linalg.inv(resize_map((w, h), (W, H))) @ np.linalg.inv(
        CORNER
    )
    atlas_grid, section_grid, certainty = ranked_matches(
        model, device, pair.target, pair.source
    )
    chosen = choose_pairs(
        pair, atlas_grid, section_grid, certainty, from_grid, top_k=top_k
    )
    if chosen.n_pairs < 3:
        return chosen.answer(
            error=f"only {chosen.n_pairs} matches outside the marked damage"
        )

    atlas_grid = atlas_grid[chosen.fitted]
    section_grid = section_grid[chosen.fitted]
    linear_raw = linear_fit(atlas_grid, section_grid, mode)
    matrix = (from_grid @ homogeneous(linear_raw) @ to_grid)[:2]
    if mode in (RIGID, AFFINE):
        return chosen.answer(matrix=matrix)

    # SPOmiAlign moves the section's points by the linear part, then fits
    # the field from them to their atlas points
    ones = np.ones((len(section_grid), 1))
    moved = (linear_raw @ np.hstack([section_grid, ones]).T).T
    displacement = smoothed_field(moved, atlas_grid, pair.smoothing_px)
    affine_raw = homogeneous(linear_raw) @ to_grid
    residual, sitk_max_px = field_on_canvas(
        displacement, affine_raw, from_grid, matrix, (w, h)
    )
    return chosen.answer(
        matrix=matrix, residual=residual, sitk_max_px=sitk_max_px
    )


def resize_map(
    size_from: tuple[int, int], size_to: tuple[int, int]
) -> npt.NDArray[np.float64]:
    """Map pixel centres through a plain resize between two sizes.

    Parameters
    ----------
    size_from, size_to : (width, height)
        The image's size before and after the resize, px.

    Returns
    -------
    ndarray, shape (3, 3)
        The homogeneous map from the first image's px to the second's.
    """
    sx, sy = size_to[0] / size_from[0], size_to[1] / size_from[1]
    return np.array(
        [[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]]
    )


def homogeneous(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Extend a 2 x 3 (or 3 x 3) affine to its 3 x 3 homogeneous form.

    Parameters
    ----------
    matrix
        The affine; rows past the second are replaced.

    Returns
    -------
    ndarray, shape (3, 3)
        The affine with the row ``[0, 0, 1]``.
    """
    rows = np.asarray(matrix, np.float64)[:2]
    return np.vstack([rows, [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class ChosenPairs:
    """Hold which ranked matches a fit uses and which it passed over.

    ``fitted`` indexes the ranked matches the fit uses, at most ``top_k``;
    ``used`` and ``passed`` are their rows (atlas x, y, section x, y in
    canvas px, confidence) and those of the matches dropped to reach them;
    ``by_mask`` counts the drops by the section mask alone.
    """

    fitted: npt.NDArray[np.intp]
    used: npt.NDArray[np.float32]
    passed: npt.NDArray[np.float32]
    by_mask: int

    @property
    def n_pairs(self) -> int:
        """Count the pairs the fit uses."""
        return len(self.fitted)

    def answer(
        self,
        *,
        matrix: npt.NDArray[np.float64] | None = None,
        residual: npt.NDArray[np.float32] | None = None,
        sitk_max_px: float = 0.0,
        error: str = "",
    ) -> EngineAnswer:
        """Make the section's answer from these pairs and what was fitted.

        Parameters
        ----------
        matrix
            The linear part, source canvas px to target canvas px.
        residual
            The warp's field minus the linear part, target canvas px.
        sitk_max_px
            The field's largest difference from SimpleITK's own.
        error
            Why the section failed; empty when it did not.

        Returns
        -------
        EngineAnswer
            The answer, the pairs counted and listed.
        """
        return EngineAnswer(
            pairs=self.n_pairs,
            dropped=len(self.passed),
            dropped_by_mask=self.by_mask,
            used=self.used,
            dropped_pairs=self.passed,
            matrix=matrix,
            residual=residual,
            sitk_max_px=sitk_max_px,
            error=error,
        )


def choose_pairs(
    pair: EnginePair,
    atlas_grid: npt.NDArray[np.float64],
    section_grid: npt.NDArray[np.float64],
    certainty: npt.NDArray[np.float64],
    from_grid: npt.NDArray[np.float64],
    *,
    top_k: int,
) -> ChosenPairs:
    """Drop the matches in the damage zone or off the mask, then take top_k.

    Parameters
    ----------
    pair
        The section's canvases, damage zone and section mask.
    atlas_grid, section_grid : ndarray, shape (n, 2)
        The ranked matches' two ends, px of the grid.
    certainty : ndarray, shape (n,)
        Their weighted certainty, most confident first.
    from_grid : ndarray, shape (3, 3)
        The map from grid px to canvas px.
    top_k
        How many of the most confident remaining matches are taken.

    Returns
    -------
    ChosenPairs
        The matches taken and those passed over to reach them.
    """
    h, w = cast("tuple[int, int]", pair.keep.shape)
    atlas_at = atlas_grid @ from_grid[:2, :2].T + from_grid[:2, 2]
    section_at = section_grid @ from_grid[:2, :2].T + from_grid[:2, 2]
    outside_damage = pair.keep[_rows(atlas_at, h), _cols(atlas_at, w)]
    inside_mask = np.ones_like(outside_damage)
    if pair.section_keep is not None:
        inside_mask = pair.section_keep[
            _rows(section_at, h), _cols(section_at, w)
        ]

    # the ranks scanned to fill top_k, and those passed over among them
    kept = outside_damage & inside_mask
    reach = int(np.searchsorted(np.cumsum(kept), top_k)) + 1
    passed = ~kept[:reach]
    by_mask = int(
        np.count_nonzero(outside_damage[:reach] & ~inside_mask[:reach])
    )
    fitted = np.flatnonzero(kept)[:top_k]
    table = np.column_stack((atlas_at, section_at, certainty)).astype(
        np.float32
    )
    return ChosenPairs(
        fitted=fitted,
        used=table[fitted],
        passed=table[np.flatnonzero(passed)],
        by_mask=by_mask,
    )


def _rows(points: npt.NDArray[np.float64], h: int) -> npt.NDArray[np.intp]:
    return np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)


def _cols(points: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.intp]:
    return np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)


def linear_fit(
    atlas_grid: npt.NDArray[np.float64],
    section_grid: npt.NDArray[np.float64],
    mode: str,
) -> npt.NDArray[np.float64]:
    """Fit the linear part *mode* names, section to atlas, on the grid.

    Parameters
    ----------
    atlas_grid, section_grid : ndarray, shape (n, 2)
        The pairs' two ends, px of the grid.
    mode : {"rigid", "affine", "warp", "affine_warp"}
        ``rigid``: a least-squares rotation and shift; ``warp``: the
        identity (SPOmiAlign's ``bspline`` mode has no linear step);
        otherwise an affine by least squares.

    Returns
    -------
    ndarray, shape (2, 3)
        The map from section grid px to atlas grid px.
    """
    if mode == RIGID:
        return rigid_fit(atlas_grid, section_grid)
    if mode == WARP:
        return np.eye(3)[:2]
    ones = np.ones((len(section_grid), 1))
    design = np.hstack([section_grid, ones])
    solution = np.linalg.lstsq(design, atlas_grid, rcond=None)[0]
    return np.asarray(solution, np.float64).T


def rigid_fit(
    fixed: npt.NDArray[np.float64], moving: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Fit the rotation and shift that best take *moving* onto *fixed*.

    Parameters
    ----------
    fixed, moving : ndarray, shape (n, 2)
        Corresponding points, one row per pair, in one pixel frame.

    Returns
    -------
    ndarray, shape (2, 3)
        ``[R | t]`` minimizing the squared distances (the Kabsch
        algorithm, a reflection ruled out), rounded through float32 as
        SPOmiAlign returns it.
    """
    fixed_centre = cast("npt.NDArray[np.float64]", fixed.mean(axis=0))
    moving_centre = cast("npt.NDArray[np.float64]", moving.mean(axis=0))
    covariance = (moving - moving_centre).T @ (fixed - fixed_centre)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    shift = fixed_centre - rotation @ moving_centre
    matrix = np.hstack([rotation, shift.reshape(2, 1)]).astype(np.float32)
    return matrix.astype(np.float64)


def smoothed_field(
    fixed: npt.NDArray[np.float64],
    moving: npt.NDArray[np.float64],
    smoothing_px: float,
) -> npt.NDArray[np.float64]:
    """Make SPOmiAlign's smoothed match field on the grid.

    Parameters
    ----------
    fixed, moving : ndarray, shape (n, 2)
        Each pair's two ends, px of the grid: the field at the pixel nearest
        a pair's *fixed* end is its displacement to its *moving* end.
    smoothing_px
        The Gaussian's sigma, px of the grid.

    Returns
    -------
    ndarray, shape (H, W, 2)
        The displacement (x, y) at every grid pixel, zero without a pair.

    Notes
    -----
    Pairs that land on one pixel are averaged; the pixels without a pair
    are filled by TELEA inpainting (radius :data:`INPAINT_RADIUS`) and the
    field blurred by a Gaussian whose kernel spans six sigmas.
    """
    # each pixel's running mean of its pairs, rounded to float32 at every
    # step as SPOmiAlign keeps it in a float32 image
    means: dict[tuple[int, int], tuple[float, float, int]] = {}
    moving_rows = cast("list[list[float]]", moving.tolist())
    fixed_rows = cast("list[list[float]]", fixed.tolist())
    for (mx, my), (fx, fy) in zip(moving_rows, fixed_rows, strict=True):
        x, y = round(fx), round(fy)
        if not (0 <= x < W and 0 <= y < H):
            continue
        ddx, ddy = mx - fx, my - fy
        previous = means.get((y, x))
        n = 0
        if previous is not None:
            mean_x, mean_y, n = previous
            ddx = (mean_x * n + ddx) / (n + 1)
            ddy = (mean_y * n + ddy) / (n + 1)
        means[(y, x)] = (float(np.float32(ddx)), float(np.float32(ddy)), n + 1)
    if not means:
        return np.zeros((H, W, 2))

    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
    known = np.zeros((H, W), dtype=bool)
    rows = [y for y, _ in means]
    cols = [x for _, x in means]
    dx[rows, cols] = [mean[0] for mean in means.values()]
    dy[rows, cols] = [mean[1] for mean in means.values()]
    known[rows, cols] = True

    # fill in between the pairs, then smooth
    unknown = (~known).astype(np.uint8) * 255
    size = int(max(3, (smoothing_px * 6) // 2 * 2 + 1))
    channels: list[npt.NDArray[np.float32]] = []
    for start in (dx, dy):
        filled = cv2.inpaint(start, unknown, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        blurred = cv2.GaussianBlur(filled, (size, size), smoothing_px)
        channels.append(np.asarray(blurred, dtype=np.float32))
    return np.stack(channels, axis=-1).astype(np.float64)


def field_on_canvas(
    displacement: npt.NDArray[np.float64],
    affine_raw: npt.NDArray[np.float64],
    from_grid: npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
    size: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], float]:
    """Read the grid's field at every canvas pixel, less the linear part.

    Parameters
    ----------
    displacement : ndarray, shape (H, W, 2)
        The field on the grid, from :func:`smoothed_field`.
    affine_raw : ndarray, shape (3, 3)
        Canvas px to the linearly moved grid px.
    from_grid : ndarray, shape (3, 3)
        Grid px to canvas px.
    matrix : ndarray, shape (2, 3)
        The linear part, canvas px to canvas px.
    size : (w, h)
        The canvas, px.

    Returns
    -------
    residual : ndarray, shape (h, w, 2)
        The full map minus the linear part, target canvas px.
    sitk_max_px : float
        The largest difference at :data:`N_PROBES` random points between
        this reading and SimpleITK's own displacement-field transform.
    """
    import SimpleITK as sitk  # local: slow to import, needed for a warp

    w, h = size
    image = sitk.GetImageFromArray(displacement, isVector=True)
    # SimpleITK's stubs leave a vector's element type unknown
    image.SetSpacing(  # pyright: ignore[reportUnknownMemberType]
        [1.0, 1.0]
    )
    transform = sitk.DisplacementFieldTransform(image)

    rng = np.random.default_rng(0)
    probe = np.column_stack(
        (rng.uniform(0, w - 1, N_PROBES), rng.uniform(0, h - 1, N_PROBES))
    )
    grid_points = probe @ affine_raw[:2, :2].T + affine_raw[:2, 2]
    points = cast("list[list[float]]", grid_points.tolist())
    # SimpleITK's stubs leave a point's element type unknown
    moved = cast(
        "list[tuple[float, float]]",
        [
            transform.TransformPoint(  # pyright: ignore[reportUnknownMemberType]
                (x, y)
            )
            for x, y in points
        ],
    )
    own = np.array(moved, dtype=np.float64)
    own = own @ from_grid[:2, :2].T + from_grid[:2, 2]
    read = _read_field(displacement, affine_raw, from_grid, probe)
    sitk_max_px = float(cast(float, np.max(np.abs(own - read))))

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    canvas = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64)
    linear = canvas @ matrix[:, :2].T + matrix[:, 2]
    full = _read_field(displacement, affine_raw, from_grid, canvas)
    residual = (full - linear).reshape(h, w, 2).astype(np.float32)
    return residual, sitk_max_px


def _read_field(
    displacement: npt.NDArray[np.float64],
    affine_raw: npt.NDArray[np.float64],
    from_grid: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map canvas points through the linear part and the field, as ITK does.

    ITK's displacement-field transform interpolates linearly (edge-clamped)
    inside the field's buffer, continuous index in [-0.5, size - 0.5], and
    displaces nothing outside it.
    """
    grid = points @ affine_raw[:2, :2].T + affine_raw[:2, 2]
    inside = (
        (grid[:, 0] >= -0.5)
        & (grid[:, 0] <= W - 0.5)
        & (grid[:, 1] >= -0.5)
        & (grid[:, 1] <= H - 0.5)
    )
    moved = grid.copy()
    for c in range(2):
        # scipy's stubs leave the output's type open
        sampled = cast(
            "npt.NDArray[np.float64]",
            map_coordinates(
                displacement[..., c],
                [grid[:, 1], grid[:, 0]],
                order=1,
                mode="nearest",
            ),
        )
        moved[:, c] += np.where(inside, sampled, 0.0)
    return moved @ from_grid[:2, :2].T + from_grid[:2, 2]
