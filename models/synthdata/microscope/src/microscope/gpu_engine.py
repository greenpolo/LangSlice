"""GPU (or CPU) spatial engine for microsim renders.

Design: microsim is the *physics oracle* (it computes the emission-weighted PSF
kernel and the detector model -- small data). This module runs the *spatial
pipeline* on the big arrays (truth -> convolve -> downsample -> detector noise)
on whichever backend you pass (`numpy` or `cupy`). Big arrays never become
xarray DataArrays, so the broken cupy<->xarray coercion path is never touched.

Reproduces, exactly, microsim's:
  - optical_image  : fftconvolve(truth_counts, summed_weighted_psf, "same")
                     (the emission magnitude is baked into the PSF weights)
  - digital_image  : coarsen-sum to output space, then CCD detector simulate()

Parity is validated against microsim on the *deterministic* optical image
(see parity_cpu.py); the detector noise is reproduced formula-for-formula from
microsim/schema/detectors/_camera.py (CCD path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from microsim.schema.backend import NumpyAPI
from microsim.schema.dimensions import Axis


def _is_cupy(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


def fftconvolve_same(xp: Any, a: Any, b: Any) -> Any:
    """3D FFT convolution, 'same' mode, on the given backend."""
    if _is_cupy(xp):
        from cupyx.scipy.signal import fftconvolve
    else:
        from scipy.signal import fftconvolve
    return fftconvolve(a, b, mode="same")


def coarsen_sum(xp: Any, arr: Any, downscale: int | tuple[int, ...]) -> Any:
    """Sum non-overlapping blocks -> microsim's output_space.rescale (coarsen.sum).

    Truncates any remainder (microsim shapes are chosen to divide evenly).
    """
    if downscale in (1, (1, 1, 1)):
        return arr
    d = (downscale,) * arr.ndim if isinstance(downscale, int) else tuple(downscale)
    crop = tuple(slice(0, (s // dd) * dd) for s, dd in zip(arr.shape, d, strict=False))
    arr = arr[crop]
    new_shape: list[int] = []
    for s, dd in zip(arr.shape, d, strict=False):
        new_shape += [s // dd, dd]
    arr = arr.reshape(new_shape)
    return arr.sum(axis=tuple(range(1, arr.ndim, 2)))


@dataclass
class Optics:
    """The small physics extracted from a microsim Simulation."""

    psf: np.ndarray  # emission-weighted, spectrally-summed PSF (nz, ny, nx)
    downscale: int | tuple[int, ...]
    exposure_ms: float
    detector: Any  # microsim _Camera (we read scalar params off it) or None
    channel_name: str


def build_optics(sim: Any, *, exposure_ms: float | None = None) -> Optics:
    """Pull the emission-weighted PSF + detector model out of a microsim Sim.

    The Simulation only needs correct optics: modality, objective, channel,
    detector, settings (esp. max_psf_radius_aus + spectral bins), and a
    truth_space whose z-extent/scale match the tiles you'll render. Its truth
    array is NOT used here, so a small dummy Y/X is fine for the config sim.
    """
    xp = NumpyAPI.create("numpy")
    em_rates = sim.filtered_emission_rates()  # (C, F, W)
    ch = em_rates.coords[Axis.C].values[0]
    fluor = em_rates.coords[Axis.F].values[0]
    em_spectrum = em_rates.sel({Axis.C: ch, Axis.F: fluor})
    if Axis.F in em_spectrum.dims:  # pragma: no cover
        em_spectrum = em_spectrum.isel({Axis.F: 0})

    # microsim's own routine -> emission-weighted sum of per-wavelength PSFs
    summed = sim.modality._summed_weighted_psf(
        em_spectrum, sim.settings, sim.truth_space, sim.objective_lens, xp
    )
    psf = np.asarray(summed, dtype=np.float32)

    downscale: int | tuple[int, ...] = 1
    if sim.output_space is not None and hasattr(sim.output_space, "downscale"):
        downscale = sim.output_space.downscale

    exp = exposure_ms if exposure_ms is not None else sim.exposure_ms
    return Optics(
        psf=psf,
        downscale=downscale,
        exposure_ms=float(exp),
        detector=sim.detector,
        channel_name=str(getattr(ch, "name", ch)),
    )


def optical_image(truth: Any, optics: Optics, xp: Any) -> Any:
    """Deterministic optical image (photons/s), (Z, Y, X) -- matches microsim."""
    t = xp.asarray(truth, dtype=xp.float32)
    psf = xp.asarray(optics.psf, dtype=xp.float32)
    return fftconvolve_same(xp, t, psf)


def apply_ccd(xp: Any, photons_per_s: Any, det: Any, exposure_ms: float) -> Any:
    """CCD detector model, reproduced from microsim _Camera.simulate (CCD path).

    Reads scalar params off the microsim detector object so it stays in sync.
    Returns uint16 gray values. (binning=1, no EM gain, no serial-register FW --
    matches the ICX285 CCD default used in our renders.)
    """
    exp_s = exposure_ms / 1000.0
    inc = xp.maximum(photons_per_s * exp_s, 0)
    detected = xp.random.poisson(inc).astype(xp.float32)
    dark_e = det.dark_current * exp_s + det.clock_induced_charge
    thermal = xp.random.poisson(dark_e, size=inc.shape).astype(xp.float32)
    total_e = detected + thermal
    total_e = xp.minimum(total_e, det.full_well)
    noise = xp.random.standard_normal(inc.shape).astype(xp.float32)
    voltage = (total_e + det.read_noise * noise) * det.gain
    gray = xp.maximum(xp.round(voltage / det.adc_gain + det.offset), 0)
    gray = xp.minimum(gray, det.max_intensity)
    return gray.astype(xp.uint16)


def render(
    truth: Any,
    optics: Optics,
    xp: Any,
    *,
    exposure_ms: float | None = None,
    seed: int = 0,
    add_noise: bool = True,
    reduce: str | None = "plane",
    plane_frac: float = 0.5,
) -> Any:
    """Full pipeline: convolve -> downscale -> detector -> (optional) 2D reduce.

    `reduce`: "plane" (single focal plane at plane_frac through z), "mip"
    (max over z), or None (return the full 3D digital image).
    """
    if hasattr(xp, "random") and hasattr(xp.random, "seed"):
        xp.random.seed(seed)

    optical = optical_image(truth, optics, xp)  # (Z, Y, X) photons/s
    optical = coarsen_sum(xp, optical, optics.downscale)

    exp = exposure_ms if exposure_ms is not None else optics.exposure_ms
    if add_noise and optics.detector is not None:
        img = apply_ccd(xp, optical, optics.detector, exp)
    else:
        img = optical * (exp / 1000.0)

    if reduce == "plane":
        return img[int(img.shape[0] * plane_frac)]
    if reduce == "mip":
        return img.max(axis=0)
    return img


def to_host(xp: Any, arr: Any) -> np.ndarray:
    """Bring an array back to the CPU as numpy."""
    return arr.get() if _is_cupy(xp) else np.asarray(arr)
