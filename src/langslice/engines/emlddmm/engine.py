# pyright: reportMissingImports=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
"""emlddmm for the LangSlice plugin, run in its own environment (Apache-2.0 stays here).

Run as ``python -I engine.py IN.npz OUT.npz`` with the ``rigid_emlddmm`` interpreter
(``plugin.py`` does this). ``emlddmm.py`` is imported from the fork named in the parameters
(``comparators/src/emlddmm_fix``: twardlab/emlddmm @ 9c4dc80 plus e426ece, which repairs
``out_of_plane=False``, and 6ff62b0, which adds ``lock_slice_axis``); nothing of it is copied or
edited. The plugin prepares every image and the stack's frame; this file only builds emlddmm's
call, runs it and reads its result at the points the plugin asks for.

IN: ``J`` (slots x rows x cols: the sections on the stack grid, in [0, 1]), ``W0``
(the same shape: 1 where a section holds data, 0 on empty slots, outside the image and on marked
damage), ``zs``, ``gy``, ``gx`` (the slot, row and column coordinates, um, the stack frame), ``R``
(3 x 3, rows: the stack's slot axis, row axis, column axis in atlas um) and ``c0`` (the atlas
point at the stack's origin, um), ``slot`` (the slot index of each section), ``q`` (N x 2: stack
(row, col) um where the map is read) with ``q_n`` (the number of points of each section),
``params`` (JSON, below).

``params``: ``emlddmm_src``, ``brainglobe_dir``, ``atlas``, ``image`` (``template``, one of the
atlas's additional references, or the borrowed Nissl), ``device``, ``dtype``, ``mode`` (``stack``:
the slots' planes locked; ``plane``: the plane free, no warp), ``cfg`` (emlddmm's keyword
arguments per scale, lists as the example's config file), ``sorting`` (``match_sd``,
``background_sd``, ``artifact_sd``, ``background_mean``, ``artifact_mean``), ``up_vector``
(``plane``), ``warp`` (read the 3-D warp too), ``mu`` (``muB``, and ``muA`` on a white slide).

OUT: ``A`` (4 x 4), ``A0``, ``A2d`` (slots x 3 x 3), ``WM`` / ``WB`` / ``WA`` (the final sorting,
half precision), ``Acut`` (sections x rows x cols: the atlas image under every stack pixel of
each section, through the whole fit), ``X`` / ``L`` (N x 3: the atlas um at the points ``q``, with
and without the warp), ``energy`` (iterations x [scale, iteration, total, mismatch,
regularization]: emlddmm's own record) with ``energy_captured`` (1 when it has one row per
iteration run), ``settings`` (JSON: what emlddmm chose), ``device``, ``peak_gpu_gb``, ``seconds``;
``error`` when the fit failed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

CACHE = Path.home() / ".cache" / "langslice-emlddmm"
#: The lock file STalign's engine takes too: one engine at a time on the GPU.
GPU_LOCK = Path.home() / ".cache" / "langslice-stalign" / "gpu.lock"
ATLAS_UM = 50.0
#: The Nissl LangSlice lends every Allen CCFv3 atlas (``core/atlas/images.py``): the template of
#: the augmented atlas, which is the CCFv3 grid extended 0.35 mm at the front.
BORROWED_NISSL = "ccfv3augmented_nissl"
AUGMENTED, AUGMENTED_VOXEL_UM, AUGMENTED_AP_OFFSET_UM = "ccfv3augmented_mouse_25um", 25.0, 350.0
#: The ``Esave`` lists of the emlddmm calls (one per scale), found by ``_watch``.
SAVED: list[list] = []
SORTING = {"match_sd": "sigmaM", "background_sd": "sigmaB", "artifact_sd": "sigmaA",
           "background_mean": "muB", "artifact_mean": "muA"}


def _watch(emlddmm) -> None:
    """Keep the energy record without editing the fork. ``Esave`` is a local list of
    ``emlddmm.emlddmm`` (total, mismatch and regularization per iteration) that only its figures
    read; ``interp`` is a module function it calls every iteration, so the wrapper finds the
    calling ``emlddmm`` frame and keeps a reference to that list."""
    real = emlddmm.interp

    def interp(*args, **kwargs):
        frame = sys._getframe(1)
        for _ in range(4):
            if frame is None:
                break
            if frame.f_code.co_name == "emlddmm":
                held = frame.f_locals.get("Esave")
                if isinstance(held, list) and not any(held is seen for seen in SAVED):
                    SAVED.append(held)
                break
            frame = frame.f_back
        return real(*args, **kwargs)

    emlddmm.interp = interp


def _energy() -> np.ndarray:
    rows = []
    for scale, held in enumerate(SAVED):
        for i, row in enumerate(held):
            rows.append([scale, i, *[float(v) for v in row[:3]]])
    return np.asarray(rows, float).reshape(-1, 5)


def device_of(name: str, dtype: str) -> tuple[str, torch.dtype]:
    """``auto``: the GPU when torch sees one, else the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        name = "cuda:0" if name == "cuda" else name
    return name, torch.float32 if dtype == "float32" else torch.float64


def atlas_folder(root: str, atlas: str) -> Path:
    """The BrainGlobe atlas folder ``<root>/<atlas>_v<version>`` (the newest installed). This
    environment has no ``brainglobe_atlasapi``: the atlas files are read as the API reads them."""
    def version(path: Path) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", path.name.rpartition("_v")[2]))

    found = sorted(Path(root).expanduser().glob(f"{atlas}_v*"), key=version)
    if not found:
        raise FileNotFoundError(f"no installed BrainGlobe atlas {atlas!r} under {root}")
    return found[-1]


def _block(data: np.ndarray, f: list[int]) -> np.ndarray:
    sh = [s // k * k for s, k in zip(data.shape, f, strict=True)]
    return data[:sh[0], :sh[1], :sh[2]].reshape(sh[0] // f[0], f[0], sh[1] // f[1], f[1],
                                                sh[2] // f[2], f[2]).mean((1, 3, 5))


def _borrowed_nissl(root: str, folder: Path, res: np.ndarray, f: list[int]) -> np.ndarray:
    """The borrowed Nissl as LangSlice builds it, at the block size *f*: the augmented template
    resampled (linear, voxel centres, its AP offset) onto the atlas grid and zeroed outside the
    brain, then block-averaged, one slab of blocks at a time (a 10 um volume is gigabytes)."""
    import tifffile
    from scipy.ndimage import affine_transform

    source = np.asarray(tifffile.imread(atlas_folder(root, AUGMENTED) / "reference.tiff"),
                        np.float32)
    annotation = tifffile.imread(folder / "annotation.tiff")
    step = res / AUGMENTED_VOXEL_UM
    offset = 0.5 * step - 0.5 + np.array([AUGMENTED_AP_OFFSET_UM / AUGMENTED_VOXEL_UM, 0, 0])
    slabs = []
    for i in range(annotation.shape[0] // f[0]):
        rows = slice(i * f[0], (i + 1) * f[0])
        slab = affine_transform(source, step, offset=offset + [step[0] * i * f[0], 0, 0],
                                output_shape=(f[0], *annotation.shape[1:]), order=1)
        slab[annotation[rows] == 0] = 0
        slabs.append(_block(slab, [f[0], f[1], f[2]]))
    return np.concatenate(slabs, 0)


def volume(root: str, atlas: str, image: str
           ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """The atlas image volume at 50 um (block mean), its axes in atlas um (voxel centres at
    index times resolution), the volume's centre (um) and its resolution."""
    import tifffile

    CACHE.mkdir(parents=True, exist_ok=True)
    folder = atlas_folder(root, atlas)
    held = CACHE / f"{folder.name}__{image}__{int(ATLAS_UM)}um.npz"
    if held.exists():
        z = np.load(held)
        return z["I"], [z["x0"], z["x1"], z["x2"]], z["centre"], z["res"]
    meta = json.loads((folder / "metadata.json").read_text())
    res = np.asarray(meta["resolution"], float)
    f = [max(1, int(round(ATLAS_UM / r))) for r in res]
    if image == BORROWED_NISSL:
        block = _borrowed_nissl(root, folder, res, f)
        shape = np.asarray(meta["shape"], float)
    else:
        path = (folder / "reference.tiff" if image == "template"
                else folder / "additional_references" / f"{image}.tiff")
        data = np.asarray(tifffile.imread(path), np.float32)
        block, shape = _block(data, f), np.asarray(data.shape, float)
    axes = [(np.arange(n) * k + (k - 1) / 2.0) * r
            for n, k, r in zip(block.shape, f, res, strict=True)]
    centre = (shape - 1) / 2.0 * res
    np.savez(held, I=block, x0=axes[0], x1=axes[1], x2=axes[2], centre=centre, res=res)
    return block, axes, centre, res


class gpu_turn:
    """One engine at a time on the GPU (a lock file shared by every process, STalign's too)."""

    def __init__(self, device: str):
        self.device = device

    def __enter__(self):
        self.handle = None
        if self.device.startswith("cuda"):
            GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(GPU_LOCK, "w")
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def _json(value) -> str:
    def plain(o):
        if torch.is_tensor(o):
            return o.detach().cpu().double().numpy().tolist()
        return np.asarray(o).tolist()

    return json.dumps(value, default=plain)


def run(z, out: dict) -> None:
    p = json.loads(str(z["params"]))
    sys.path.insert(0, p["emlddmm_src"])
    import emlddmm

    _watch(emlddmm)
    t0 = time.monotonic()
    device, dtype = device_of(p["device"], p["dtype"])
    out["device"] = f"{device} {str(dtype).replace('torch.', '')}"
    I, axes, centre, res = volume(  # noqa: E741 (emlddmm's name for the atlas image)
        p["brainglobe_dir"], p["atlas"], p["image"])
    I = I[None].astype(np.float64)  # noqa: E741
    I /= max(float(np.mean(np.abs(I))), 1e-12)  # noqa: E741 (the atlas image over its mean)
    xI = [ax - c for ax, c in zip(axes, centre, strict=True)]  # about the volume's centre, as
    # the example's atlas: emlddmm turns each section about the origin
    zs, gy, gx = (np.asarray(z[k], np.float64) for k in ("zs", "gy", "gx"))
    R, c0 = np.asarray(z["R"], np.float64), np.asarray(z["c0"], np.float64)
    A0 = np.eye(4)
    A0[:3, :3] = R
    A0[:3, 3] = -R @ (c0 - centre)
    W0 = np.asarray(z["W0"], np.float64)
    cfg = {key: list(value) for key, value in p["cfg"].items()}
    for key, name in SORTING.items():
        if key in p["sorting"]:
            cfg[name] = [[float(p["sorting"][key])]]
    plane = p["mode"] == "plane"
    config = {"I": I, "xI": [xI], "J": np.asarray(z["J"], np.float64)[None], "xJ": [[zs, gy, gx]],
              "W0": W0, "device": device, "dtype": dtype, **p["mu"], **cfg,
              "lock_slice_axis": [not plane], "slice_matching": [True],
              "out_of_plane": [False], "A": A0, "n_draw": 0, "full_outputs": True}
    if plane:
        config["up_vector"] = [[float(v) for v in p["up_vector"]]]
    with gpu_turn(device):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        result = emlddmm.emlddmm_multiscale(**config)[-1]
        out["peak_gpu_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                              if device.startswith("cuda") else 0.0)
    cpu = {"dtype": torch.float64}
    A = result["A"].detach().cpu().to(**cpu)
    if not plane and not torch.allclose(A[0], torch.as_tensor(A0)[0]):
        raise RuntimeError("the locked slice axis moved")
    A2d = result["A2d"].detach().cpu().to(**cpu)
    xv = [torch.as_tensor(x).detach().cpu().to(**cpu) for x in result["xv"]]
    v = torch.as_tensor(result["v"]).detach().cpu().to(**cpu)
    warp = bool(p["warp"])
    with torch.no_grad():
        phii = emlddmm.v_to_phii(xv, v)
        XV = torch.stack(torch.meshgrid(xv, indexing="ij"))
    Ai = torch.linalg.inv(A)
    shift = torch.as_tensor(centre, **cpu)

    def chain(q, k: int, with_warp: bool) -> torch.Tensor:
        """Stack (row, col) um on slot k -> atlas um (N, 3): the sample the fit's own loop
        takes (target point -> A2d_k^-1 -> A^-1 -> the warp)."""
        n = len(q)
        Y = torch.stack([torch.full((n,), float(zs[k]), **cpu),
                         torch.as_tensor(q[:, 0], **cpu), torch.as_tensor(q[:, 1], **cpu)])
        R2i = torch.linalg.inv(A2d[k])
        Y[1:] = R2i[:2, :2] @ Y[1:] + R2i[:2, 2:3]
        X = Ai[:3, :3] @ Y + Ai[:3, 3:4]
        if with_warp:
            with torch.no_grad():
                X = emlddmm.interp(xv, phii - XV, X.reshape(3, 1, 1, -1)).reshape(3, -1) + X
        return X.T

    slots = np.asarray(z["slot"], int)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], -1)
    Atlas = torch.as_tensor(I, **cpu)
    xA = [torch.as_tensor(x, **cpu) for x in xI]
    Acut = np.zeros((len(slots), len(gy), len(gx)), np.float32)
    for i, k in enumerate(slots):
        X = chain(grid, int(k), warp)
        Acut[i] = emlddmm.interp(xA, Atlas, X.T.reshape(3, 1, 1, -1)).reshape(len(gy), len(gx)
                                                                              ).numpy()
    out["Acut"] = Acut
    q, counts = np.asarray(z["q"], np.float64), np.asarray(z["q_n"], int)
    X_, L_, at = [], [], 0
    for k, n in zip(slots, counts, strict=True):
        part = q[at:at + n]
        at += n
        X_.append(((chain(part, int(k), warp) + shift) if n else torch.zeros((0, 3), **cpu)
                   ).numpy())
        L_.append(((chain(part, int(k), False) + shift) if n else torch.zeros((0, 3), **cpu)
                   ).numpy())
    out["X"] = np.concatenate(X_, 0).astype(np.float32) if X_ else np.zeros((0, 3), np.float32)
    out["L"] = np.concatenate(L_, 0).astype(np.float32) if L_ else np.zeros((0, 3), np.float32)
    out["A"], out["A0"], out["A2d"] = A.numpy(), A0, A2d.numpy()
    for key in ("WM", "WB", "WA"):
        out[key] = result[key].detach().cpu().numpy().astype(np.float16)
    out["energy"] = _energy()  # the wrapper's hold on emlddmm's locals may break: count the rows
    out["energy_captured"] = np.int32(len(out["energy"]) == sum(int(n) for n in cfg["n_iter"]))
    out["settings"] = _json({"muA": result["muA"], "muB": result["muB"],
                             "sigmaM": result["sigmaM"], "sigmaB": result["sigmaB"],
                             "sigmaA": result["sigmaA"], "cfg": cfg})
    out["seconds"] = time.monotonic() - t0


def main(src: str, dst: str) -> int:
    z = np.load(src, allow_pickle=False)
    out: dict = {}
    try:
        run(z, out)
    except Exception as exc:  # the plugin turns it into the call's error
        traceback.print_exc()
        out = {"error": f"{type(exc).__name__}: {exc}"}
    np.savez_compressed(dst, **out)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.exit(main(sys.argv[1], sys.argv[2]))
