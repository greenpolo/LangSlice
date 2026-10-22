# pyright: reportMissingImports=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
"""SPOmiAlign's matching and fitting for the LangSlice plugin, run in its own environment.

Run as ``python engine.py IN.npz OUT.npz`` with the ``rigid_spomialign`` interpreter
(``plugin.py`` does this). SPOmiAlign (Wang et al. 2026, Brief Bioinform; no licence in
repo: local research use only) and its bundled RoMa are imported from
``comparators/src/spomialign`` (``SPOMIALIGN_SRC`` overrides). The steps are those of
``SPOmiAlign/roma.py::align_and_process_images`` as ``comparators/rigid/spomialign_runner.py``
audits them (``_local/comparators/fig3_baselines_20261016/runs/spomialign/FAITHFULNESS.md``):
both images resized to SPOmiAlign's 1152 x 864 grid, RoMa outdoor matches, certainty
weighted by the source's FFT edge map, 3 x 3 grid NMS, the 5000 most confident matches;
then the transform the mode names, fitted with SPOmiAlign's own functions. The match budget
(``matches``, the paper's K, "typically 5000") and the warp's smoothing (``smoothing_px``,
``fit_bspline_transform``'s Gaussian sigma on the 1152 x 864 grid, 15 by default) are the
call's; the edge-weight constants stay as the authors fixed them.

Damage is the plugin's, as the native fits handle it: ``plugin.py`` paints the marked regions
as background in the atlas image, and here matches whose atlas end lands in the damage zone
(the marked regions plus the native margin; ``keep`` False) are dropped before the 5000 are
taken.

IN, for each section ``i`` (``n`` sections): ``target_i`` (h x w x 3 uint8, the atlas
image), ``source_i`` (the same size: the section on the same canvas), ``keep_i`` (h x w
bool, False in the damage zone); ``mode`` (rigid | affine | warp | affine_warp),
``matches``, ``smoothing_px``.
OUT: ``matrix_i`` (2 x 3, source canvas px -> target canvas px: the linear part; the
identity for ``warp``), ``residual_i`` (h x w x 2 for the warp modes: the full map minus the
linear part at every source canvas pixel, target canvas px), ``pairs_i``, ``dropped_i``,
``sitk_max_px_i`` (the vectorized field against SimpleITK's own ``TransformPoint``),
``used_i`` / ``dropped_pairs_i`` (rows atlas x, y, section x, y in canvas px, confidence: the
pairs fitted, and those in the damage zone passed over to reach them),
``error_i`` (empty, or why the section failed).
"""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("SPOMIALIGN_SRC")
           or "~/Software/SliceBench/comparators/src/spomialign/SPOmiAlign").expanduser()
sys.path.insert(0, str(SRC / "software" / "Roma"))
sys.path.insert(0, str(SRC))

import roma as SP  # noqa: E402  (SPOmiAlign/roma.py)
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

#: SPOmiAlign's registration grid (roma.py ``H, W = 864, 1152``; paper: "1152 x 864")
H, W = 864, 1152
#: RoMa pixel coordinates are corner based (``W / 2 (x + 1)``): pixel centre + 0.5
#: The GPU's one queue, shared with FireANTs, STalign and emlddmm (their lock file).
GPU_LOCK = Path(os.environ.get("LANGSLICE_GPU_LOCK")
                or Path.home() / ".cache" / "langslice-stalign" / "gpu.lock")
#: A section that still finds the GPU full waits OOM_WAIT_S, twice that, ... and runs again.
OOM_RETRIES = 4
OOM_WAIT_S = 5.0


def out_of_memory(exc: BaseException) -> bool:
    """The GPU ran out of memory (torch's error, or what cuDNN and cuBLAS raise
    when their allocation fails)."""
    text = str(exc)
    return (type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text
            or "CUDNN_STATUS_INTERNAL_ERROR" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text)


class gpu_queue:
    """The whole call, the matcher's loading included, in the GPU's queue (a
    lock on GPU_LOCK); nothing on the CPU."""

    def __init__(self, device):
        self.device = str(device)
        self.handle = None
        self.waited_s = 0.0

    def __enter__(self):
        if self.device.startswith("cuda"):
            GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(GPU_LOCK, "w")
            asked = time.monotonic()
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            self.waited_s = time.monotonic() - asked
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


#: Times the GPU was out of memory and a call ran again (RETRIES["n"]).
RETRIES = {"n": 0}


def retried(device, call):
    """*call*, run again when a process outside the queue has taken the GPU's memory."""
    for attempt in range(OOM_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            if (not str(device).startswith("cuda") or attempt == OOM_RETRIES
                    or not out_of_memory(exc)):
                raise
            torch.cuda.empty_cache()
            RETRIES["n"] += 1
            time.sleep(OOM_WAIT_S * (attempt + 1))
    raise AssertionError("unreachable")


CORNER = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
MODES = ("rigid", "affine", "warp", "affine_warp")


def resize_map(w_from: int, h_from: int, w_to: int, h_to: int) -> np.ndarray:
    """Pixel-centre map of a plain resize, image of size (w_from, h_from) -> (w_to, h_to)."""
    sx, sy = w_to / w_from, h_to / h_from
    return np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])


def homog(m: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(m, np.float64)[:2], [0.0, 0.0, 1.0]])


def matches(model, device, target: np.ndarray, source: np.ndarray):
    """SPOmiAlign's ranked matches, every one (RoMa grid, corner convention):
    ``(kpts1 target, kpts2 source, weighted certainty)``, most confident first."""
    with tempfile.TemporaryDirectory() as tmp:
        img1, img2 = Path(tmp) / "atlas.png", Path(tmp) / "section.png"
        Image.fromarray(target).save(img1)
        Image.fromarray(source).save(img2)
        warp, certainty = model.match(str(img1), str(img2), device=device)
        edge = SP.compute_edge_weight(str(img2), H, W, device)
    edge_full = torch.cat((torch.zeros_like(edge), edge), dim=1)
    if certainty.dim() == 3:
        certainty = certainty.squeeze(0)
    weighted = certainty * (edge_full * 0.9 + 0.1)
    weighted[:, :W] = 0
    filtered = SP.apply_grid_nms(weighted, kernel_size=3)
    k1, k2, cert = model.get_sorted_matches(warp, filtered, H, W)
    return (k1.cpu().numpy().astype(np.float64), k2.cpu().numpy().astype(np.float64),
            cert.float().cpu().numpy().astype(np.float64))


def run(model, device, target: np.ndarray, source: np.ndarray, keep: np.ndarray,
        mode: str, top_k: int, smoothing_px: float,
        section_keep: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h, w = target.shape[:2]
    to_grid = CORNER @ resize_map(w, h, W, H)  # canvas px -> RoMa grid
    from_grid = np.linalg.inv(resize_map(w, h, W, H)) @ np.linalg.inv(CORNER)
    k1, k2, cert = matches(model, device, target, source)
    # the plugin's step: drop matches whose atlas end is in the damage zone and, with a
    # section mask, those whose section end is where the mask is under one half; then take
    # the top_k most confident of the rest
    at = k1 @ from_grid[:2, :2].T + from_grid[:2, 2]
    col = np.clip(np.rint(at[:, 0]).astype(int), 0, w - 1)
    row = np.clip(np.rint(at[:, 1]).astype(int), 0, h - 1)
    section_at = k2 @ from_grid[:2, :2].T + from_grid[:2, 2]
    outside_damage = keep[row, col]
    if section_keep is None:
        inside_mask = np.ones_like(outside_damage)
    else:
        scol = np.clip(np.rint(section_at[:, 0]).astype(int), 0, w - 1)
        srow = np.clip(np.rint(section_at[:, 1]).astype(int), 0, h - 1)
        inside_mask = section_keep[srow, scol]
    kept = outside_damage & inside_mask
    reach = int(np.searchsorted(np.cumsum(kept), top_k)) + 1  # ranks scanned to fill top_k
    passed = ~kept[:reach]

    def rows(sel: np.ndarray) -> np.ndarray:
        return np.c_[at[sel], section_at[sel], cert[sel]].astype(np.float32)

    out: dict[str, np.ndarray] = {
        "dropped": np.array(int(passed.sum())),
        "dropped_by_mask": np.array(int((outside_damage[:reach] & ~inside_mask[:reach]).sum())),
        "dropped_pairs": rows(np.flatnonzero(passed)),
        "used": rows(np.flatnonzero(kept)[:top_k])}
    k1, k2 = k1[kept][:top_k], k2[kept][:top_k]
    out["pairs"] = np.array(len(k1))
    if len(k1) < 3:
        out["error"] = np.array(f"only {len(k1)} matches outside the marked damage")
        return out
    ones = np.ones((len(k2), 1))
    if mode == "rigid":
        m_raw = np.asarray(SP.fit_rigid_transform(k1, k2), np.float64)[:2]
    elif mode == "warp":  # SPOmiAlign's own 'bspline' mode: no linear step
        m_raw = np.eye(3)[:2]
    else:  # affine and affine+bspline: least squares, source -> target (roma.py)
        m_raw = np.linalg.lstsq(np.hstack([k2, ones]), k1, rcond=None)[0].T
    out["matrix"] = (from_grid @ homog(m_raw) @ to_grid)[:2]
    if mode in ("rigid", "affine"):
        return out
    # fit_bspline_transform(M kpts2, kpts1): tx_csv, as SPOmiAlign moves spot coordinates
    moved = (m_raw @ np.hstack([k2, ones]).T).T
    tx = SP.fit_bspline_transform(moved, k1, H, W, smooth_sigma=smoothing_px)
    disp = (sitk.GetArrayFromImage(tx.GetDisplacementField()).astype(np.float64)
            if tx.GetName() == "DisplacementFieldTransform" else np.zeros((H, W, 2)))
    a_raw = homog(m_raw) @ to_grid  # canvas px -> affine-mapped RoMa grid

    def field(xy: np.ndarray) -> np.ndarray:
        p = xy @ a_raw[:2, :2].T + a_raw[:2, 2]
        # ITK's displacement-field transform: linear interpolation (edge-clamped) inside the
        # field's buffer, continuous index in [-0.5, size - 0.5]; zero outside it
        inside = ((p[:, 0] >= -0.5) & (p[:, 0] <= W - 0.5) & (p[:, 1] >= -0.5)
                  & (p[:, 1] <= H - 0.5))
        q = p.copy()
        for c in range(2):
            q[:, c] += np.where(inside, map_coordinates(disp[..., c], [p[:, 1], p[:, 0]],
                                                        order=1, mode="nearest"), 0.0)
        return q @ from_grid[:2, :2].T + from_grid[:2, 2]

    rng = np.random.default_rng(0)
    probe = np.c_[rng.uniform(0, w - 1, 500), rng.uniform(0, h - 1, 500)]
    pa = probe @ a_raw[:2, :2].T + a_raw[:2, 2]
    own = np.array([tx.TransformPoint(tuple(map(float, p))) for p in pa])
    own = own @ from_grid[:2, :2].T + from_grid[:2, 2]
    out["sitk_max_px"] = np.array(float(np.abs(own - field(probe)).max()))
    ys, xs = np.mgrid[0:h, 0:w]
    grid = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
    linear = grid @ out["matrix"][:, :2].T + out["matrix"][:, 2]
    out["residual"] = (field(grid) - linear).reshape(h, w, 2).astype(np.float32)
    return out


def main() -> int:
    data = np.load(sys.argv[1])
    mode = str(data["mode"])
    if mode not in MODES:
        raise SystemExit(f"mode is one of {MODES}; got {mode!r}")
    device = SP.resolve_torch_device(os.environ.get("SPOMIALIGN_DEVICE") or None)
    with gpu_queue(device) as turn:
        return _match_all(data, mode, device, turn.waited_s)


def _match_all(data, mode: str, device, waited_s: float = 0.0) -> int:
    model = retried(device, lambda: SP.roma_outdoor(device=device, coarse_res=560,
                                                    upsample_res=(H, W)))
    result: dict[str, np.ndarray] = {"gpu_wait_s": np.array(waited_s)}
    for i in range(int(data["n"])):
        before = RETRIES["n"]
        try:
            held = f"section_keep_{i}"
            got = retried(device, lambda i=i, held=held: run(
                model, device, data[f"target_{i}"], data[f"source_{i}"],
                data[f"keep_{i}"].astype(bool), mode, int(data["matches"]),
                float(data[f"smoothing_px_{i}"]),
                data[held].astype(bool) if held in data.files else None))
        except Exception as exc:  # one section failing leaves the others
            got = {"error": np.array(f"{type(exc).__name__}: {exc}")}
        got["oom_retries"] = np.array(RETRIES["n"] - before)
        result.update({f"{key}_{i}": value for key, value in got.items()})
    np.savez(sys.argv[2], **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
