# pyright: reportMissingImports=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
"""STalign 2.0 for the LangSlice plugin, run in its own environment (GPL-3.0 stays here).

Run as ``python -I engine.py IN.npz OUT.npz`` with the ``stalign2`` interpreter (``plugin.py``
does this). STalign 2.0.0 (JEFworks-Lab main 8a84bf1f, installed from
``comparators/src/stalign2``) is imported as installed; nothing of it is copied. The plugin
prepares every image and frame; this file only calls STalign and samples its result.

IN (``mode`` picks the call):

- ``lddmm`` / ``sorting``: 2-D ``STalign.LDDMM``. ``I`` (C x h x w source), ``xI0``, ``xI1``
  (its pixel centres, um), ``J`` (1 x H x W target), ``xJ0``, ``xJ1``, ``A0`` (3 x 3, source um
  -> target um, row/col order), ``query`` (N x 2 target um, row/col: where the map is
  sampled), ``params`` (JSON: STalign keyword arguments, mixture included), and optional
  ``pointsI``/``pointsJ``. ``sorting`` holds the source still (no step of any kind) for 56
  iterations, so STalign's own sorting updates run at iterations 50 and 55 at the current
  alignment.
- ``plane``: 3-D ``STalign.LDDMM_3D_to_slice`` with ``niter=0``: the plane search and its
  refinement only. ``atlas`` (BrainGlobe name), ``image`` (``template`` or one of its
  additional references), ``J``, ``xJ0``, ``xJ1``, ``A0`` (4 x 4, atlas um -> target um,
  target = (0, row um, col um)), ``params``. The volume is block-averaged to 50 um (the
  published atlas voxel) and cached under ``~/.cache/langslice-stalign``.

OUT: ``source_um`` (N x 2 for 2-D: the source point STalign's map takes each query point to),
``A``, ``WM``/``WB``/``WA``, ``energy`` (iterations x [E, EM, ER(, EP)], STalign's record),
``settings`` (JSON: what STalign chose), ``prefit`` (JSON, 3-D), ``device``, ``seconds``;
``error`` when the fit failed.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from STalign import STalign, _lddmm_common

CACHE = Path.home() / ".cache" / "langslice-stalign"
ATLAS_UM = 50.0
#: The Nissl image LangSlice offers on an Allen CCFv3 atlas, borrowed from another atlas.
BORROWED_NISSL = "ccfv3augmented_nissl"
ENERGY: list[np.ndarray] = []


class _Record:
    """Stands in for STalign's end-of-fit figures (``draw='end'``): draws nothing and keeps
    the energy record the figures would plot."""

    def __init__(self, problem):
        pass

    def update(self, snap):
        ENERGY[:] = [np.asarray(snap["Esave"], float)]


_lddmm_common._Figures = _Record


def device_of(name: str) -> tuple[str, torch.dtype]:
    """``auto``: the GPU at float32 when torch sees one, else the CPU at float64 (STalign's
    C++ engine)."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        return ("cuda:0" if name == "cuda" else name), torch.float32
    return "cpu", torch.float64


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().double().numpy()
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def _json(value) -> str:
    """JSON of STalign's records (tensors and arrays as lists)."""
    return json.dumps(value, default=lambda o: np.asarray(plain(o)).tolist())


def borrowed_nissl(a) -> np.ndarray:
    """LangSlice's borrowed Nissl on an Allen CCFv3 atlas (``core/atlas/images.py``): the
    ``ccfv3augmented_mouse_25um`` template, whose grid starts 0.35 mm further forward,
    resampled onto *a*'s grid, zero outside its brain."""
    from brainglobe_atlasapi import BrainGlobeAtlas
    from scipy.ndimage import affine_transform

    source = np.asarray(BrainGlobeAtlas("ccfv3augmented_mouse_25um", check_latest=False).template,
                        np.float32)
    step = np.asarray(a.resolution, float) / 25.0
    offset = 0.5 * step - 0.5 + np.array([350.0 / 25.0, 0.0, 0.0])
    annotation = np.asarray(a.annotation)
    data = affine_transform(source, np.diag(step), offset=offset, output_shape=annotation.shape,
                            order=1, output=np.float32)
    data[annotation == 0] = 0
    return data


def volume(atlas: str, image: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """The atlas image volume at 50 um (block mean) and its axes in atlas um."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    CACHE.mkdir(parents=True, exist_ok=True)
    held = CACHE / f"{atlas}__{image}__{int(ATLAS_UM)}um.npz"
    if held.exists():
        z = np.load(held)
        return z["I"], [z["x0"], z["x1"], z["x2"]]
    a = BrainGlobeAtlas(atlas, check_latest=False)
    if image == BORROWED_NISSL:
        data = borrowed_nissl(a)
    else:
        data = a.reference if image == "template" else a.additional_references[image]
    data = np.asarray(data, np.float32)
    res = np.asarray(a.resolution, float)
    f = [max(1, int(round(ATLAS_UM / r))) for r in res]
    sh = [s // k * k for s, k in zip(data.shape, f, strict=True)]
    block = data[:sh[0], :sh[1], :sh[2]].reshape(sh[0] // f[0], f[0], sh[1] // f[1], f[1],
                                                 sh[2] // f[2], f[2]).mean((1, 3, 5))
    axes = [(np.arange(n) * k + (k - 1) / 2.0) * r
            for n, k, r in zip(block.shape, f, res, strict=True)]
    np.savez(held, I=block, x0=axes[0], x1=axes[1], x2=axes[2])
    return block, axes


class gpu_turn:
    """One engine at a time on the GPU (a lock file shared by every process)."""

    def __init__(self, device: str):
        self.device = device
        self.waited_s = 0.0

    def __enter__(self):
        self.handle = None
        if self.device.startswith("cuda"):
            CACHE.mkdir(parents=True, exist_ok=True)
            self.handle = open(CACHE / "gpu.lock", "w")
            asked = time.monotonic()
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            self.waited_s = time.monotonic() - asked
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def run(z, out: dict) -> None:
    mode = str(z["mode"])
    params = json.loads(str(z["params"]))
    device, dtype = device_of(params.pop("device", "auto"))
    threads = int(params.pop("threads", 4))
    out["device"] = f"{device} {str(dtype).replace('torch.', '')}"
    t0 = time.monotonic()
    if mode == "plane":
        I, xI = volume(str(z["atlas"]), str(z["image"]))  # noqa: E741
        I = I[None] / np.mean(np.abs(I))  # noqa: E741 (STalign's name)
        I = np.concatenate((I, (I - I.mean()) ** 2))  # noqa: E741 (STalign's name)
        with gpu_turn(device):
            result = STalign.LDDMM_3D_to_slice(
                xI, I, [z["xJ0"], z["xJ1"]], z["J"], A=z["A0"], niter=0, prefit=True,
                draw=False, device=device, dtype=dtype, num_threads=threads, **params)
        pf = result.get("prefit", {})
        out["A"] = plain(result["A"])
        out["prefit"] = _json({k: v for k, v in pf.items() if k != "A"})
    else:
        kw = dict(params)
        if mode == "sorting":
            kw.update(niter=56, epL=0.0, epT=0.0, epV=0.0)
        if "pointsI" in z.files:
            kw.update(pointsI=z["pointsI"], pointsJ=z["pointsJ"])
        A0 = z["A0"]
        with gpu_turn(device) as turn:
            out["gpu_wait_s"] = turn.waited_s
            result = STalign.LDDMM([z["xI0"], z["xI1"]], z["I"], [z["xJ0"], z["xJ1"]], z["J"],
                                   L=A0[:2, :2].copy(), T=A0[:2, 2].copy(), prefit=False,
                                   draw="end", device=device, dtype=dtype, num_threads=threads,
                                   **kw)
            A_t, v_t = result["A"], result["v"]
            xv = result["xv"]
            query = np.asarray(z["query"], float)
            got = []
            for k in range(0, len(query), 1 << 20):
                Xq = torch.as_tensor(query[k:k + (1 << 20)], dtype=v_t.dtype,
                                     device=v_t.device)[:, None]
                with torch.no_grad():
                    got.append(plain(STalign.build_transform(xv, v_t, A_t, direction="b",
                                                             XJ=Xq))[:, 0])
        out["source_um"] = np.concatenate(got, 0) if got else np.zeros((0, 2))
        out["A"] = plain(A_t)
        for key in ("WM", "WB", "WA"):
            out[key] = plain(result[key])
        out["energy"] = ENERGY[0] if ENERGY else np.zeros((0, 3))
    out["settings"] = _json(dict(result.get("settings", {})))
    out["seconds"] = time.monotonic() - t0


def main(src: str, dst: str) -> int:
    z = np.load(src, allow_pickle=False)
    out: dict = {}
    try:
        run(z, out)
    except Exception as exc:  # the plugin turns it into the section's error
        traceback.print_exc()
        out = {"error": f"{type(exc).__name__}: {exc}"}
    np.savez_compressed(dst, **out)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.exit(main(sys.argv[1], sys.argv[2]))
