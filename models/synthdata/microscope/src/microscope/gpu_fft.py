"""Surgical GPU offload for microsim's PSF convolution.

Why this instead of microsim's built-in `np_backend="cupy"`:
  The cupy backend forces EVERY array op onto cupy, so when microsim wraps the
  result back into an xarray.DataArray, xarray's `as_compatible_data` tries to
  coerce the cupy array via `np.asarray`, which cupy refuses ("Implicit
  conversion to NumPy array is not allowed"). That path is broken in
  microsim 0.0.12 (OS-independent).

What we do instead:
  Keep the pipeline on the numpy backend so xarray only ever sees numpy arrays,
  and replace the ONE expensive, repeated operation -- the 3D FFT convolution in
  `_PSFModality.render` (`microsim/schema/modality/_simple_psf.py`) -- with a
  cupy implementation that is numpy-in / numpy-out. The PSF itself is cached by
  microsim, so the convolution is the only hot path.

Usage:
    from .gpu_fft import enable_cuda_wheels, install_gpu_fftconvolve
    enable_cuda_wheels()          # let cupy-cuda12x find the pip-wheel CUDA libs
    install_gpu_fftconvolve()     # patch NumpyAPI.fftconvolve -> GPU
    # ...then build a normal numpy-backend Simulation and call .digital_image()
"""
from __future__ import annotations

import glob
import os
import sysconfig


def enable_cuda_wheels() -> int:
    """Add the NVIDIA pip-wheel `bin` dirs to PATH so cupy-cuda12x can load
    nvrtc-builtins etc. (loaded via PATH, not just the loader search path)."""
    sp = sysconfig.get_paths()["purelib"]
    bins = glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
        for b in bins:
            try:
                os.add_dll_directory(b)
            except OSError:
                pass
    return len(bins)


# --- instrumentation: count + time the convolutions so we can see the split ---
STATS = {"calls": 0, "seconds": 0.0}


def reset_stats() -> None:
    STATS["calls"] = 0
    STATS["seconds"] = 0.0


_cupy = None
_cu_fftconvolve = None


def _load_cupy():
    global _cupy, _cu_fftconvolve
    if _cupy is None:
        import cupy
        from cupyx.scipy.signal import fftconvolve

        _cupy, _cu_fftconvolve = cupy, fftconvolve
    return _cupy, _cu_fftconvolve


def install_gpu_fftconvolve(verbose: bool = True, sync: bool = True) -> None:
    """Patch microsim's NumpyAPI so its fftconvolve runs on the GPU.

    numpy-in / numpy-out: xarray never sees a cupy array, so the broken
    cupy<->xarray coercion path is never exercised.
    """
    import time

    from microsim.schema.backend import NumpyAPI

    cupy, cu_fftconvolve = _load_cupy()

    def _unwrap(x):
        # xarray.DataArray exposes .dims; unwrap to its raw ndarray. Leave plain
        # ndarrays alone (their .data is a memoryview/MemoryPointer, not data).
        return x.data if hasattr(x, "dims") else x

    def fftconvolve(self, a, b, mode="full"):
        t0 = time.perf_counter()
        a_gpu = cupy.asarray(_unwrap(a))
        b_gpu = cupy.asarray(_unwrap(b))
        out_gpu = cu_fftconvolve(a_gpu, b_gpu, mode=mode)
        result = cupy.asnumpy(out_gpu)  # back to host
        del a_gpu, b_gpu, out_gpu
        if sync:
            cupy.cuda.Stream.null.synchronize()
        STATS["calls"] += 1
        STATS["seconds"] += time.perf_counter() - t0
        return result

    NumpyAPI.fftconvolve = fftconvolve
    if verbose:
        print("[gpu_fft] patched NumpyAPI.fftconvolve -> cupyx.scipy.signal.fftconvolve")


def free_gpu_mem() -> None:
    """Release cupy's memory pool (useful between differently-sized runs)."""
    if _cupy is not None:
        _cupy.get_default_memory_pool().free_all_blocks()
        _cupy.fft.config.clear_plan_cache()
