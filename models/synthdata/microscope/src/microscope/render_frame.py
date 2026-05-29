"""Render ONE real BZ-X800 microscope FRAME of synthetic DAPI.

Unlike the old square patches, this matches a real acquisition exactly: the
field of view, aspect ratio (4:3) and pixel size all come from the BZ-X800
FOV-by-magnification table via lab_optics.frame_spec(). Most whole-slice imaging
is 10x or 20x, so those are the presets.

  20x high_res -> 1920x1440 px over 724.69x543.52 um  @ 0.37744 um/px
  10x high_res -> 1920x1440 px over 1449.38x1087.03 um @ 0.75488 um/px

FOCUS knob (how the 50um slab collapses to a 2D image):
  full_focus : max-intensity projection over z  (Keyence Full Focus / EDF -- the
               Keyence default; whole slab in focus)
  plane      : a single focal plane            (one optical section, hazy slab)
  confocal   : optical sectioning via a pinhole then one plane (crisp)

A frame is bigger than one FFT fits, so it's rendered in halo-tiled blocks
(see render_large_patch); the per-type size ratio + chromocenter ("speck")
params from the 10x marker extraction are baked into the template bank.

    python render_frame.py                          # 10x, full_focus, SSp (default)
    python render_frame.py --mag 20 --focus plane
    python render_frame.py --mag 20 --focus confocal --region CA1sp
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .gpu_fft import enable_cuda_wheels

enable_cuda_wheels()

import cupy  # noqa: E402
from microsim import schema as ms  # noqa: E402

from . import density_lookup as dl  # noqa: E402
from . import gpu_engine as ge  # noqa: E402
from . import gpu_truth as gt  # noqa: E402
from .lab_optics import (  # noqa: E402
    LAB_DZ_UM,
    bzx_optical_config,
    frame_spec,
    objective_air,
)
from .render_shape_texture import style_navy  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out" / "gpu"
OUT.mkdir(parents=True, exist_ok=True)
SHAPES = HERE / "assets" / "hippocampus_shapes.npy"

MAX_AU = 40
BINS = 3
CORE = 512                 # core tile size (px); footprint = CORE + 2*halo
VRAM_LIMIT_GB = 18.0
EXPOSURE_MS = 200

FOCUS_REDUCE = {"full_focus": "mip", "plane": "plane", "confocal": "plane"}


def build_modality(focus: str):
    if focus == "confocal":
        return ms.Confocal(pinhole_au=1.0)
    return ms.Widefield()


def config_sim(mag: int, px_um: float, focus: str, nz: int):
    # PSF z-extent is taken from truth_space.shape[0], so it MUST span the full
    # slab or the out-of-focus haze is truncated. Lateral dummy size is small
    # (PSF lateral radius is set by max_psf_radius_aus, not this).
    scale = (LAB_DZ_UM, px_um, px_um)
    dummy = np.zeros((nz, 320, 320), np.float32)
    label = ms.FluorophoreDistribution(
        distribution=ms.FluorophoreDistribution.from_array(dummy).distribution,
        fluorophore=ms.Fluorophore.from_fpbase("DAPI"),
    )
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(nz, 320, 320), scale=scale),
        output_space=ms.DownscaledSpace(downscale=1),
        sample=ms.Sample(labels=[label]),
        modality=build_modality(focus),
        objective_lens=objective_air(mag),
        channels=[bzx_optical_config("DAPI")],
        detector=ms.detectors.lib.ICX285,
        settings=ms.Settings(
            random_seed=0, np_backend="numpy",
            spectral_bins_per_emission_channel=BINS, max_psf_radius_aus=MAX_AU,
        ),
    )


def render_tiled(truth_h, optics, halo, reduce):
    """Convolve the big truth in CORE tiles with a `halo` margin; stitch."""
    nz, ny, nx = truth_h.shape
    out = np.zeros((ny, nx), dtype=np.uint16)
    tid = 0
    for y in range(0, ny, CORE):
        for x in range(0, nx, CORE):
            y1, x1 = min(y + CORE, ny), min(x + CORE, nx)
            ya, yb = max(y - halo, 0), min(y1 + halo, ny)
            xa, xb = max(x - halo, 0), min(x1 + halo, nx)
            sub = cupy.asarray(truth_h[:, ya:yb, xa:xb])
            img = ge.render(sub, optics, cupy, exposure_ms=EXPOSURE_MS,
                            reduce=reduce, seed=tid)
            img_h = ge.to_host(cupy, img)
            cy, cx = y - ya, x - xa
            out[y:y1, x:x1] = img_h[cy:cy + (y1 - y), cx:cx + (x1 - x)]
            del sub, img
            cupy.get_default_memory_pool().free_all_blocks()
            tid += 1
    return out, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag", type=int, default=10, choices=(10, 20))
    ap.add_argument("--readout", default="high_res",
                    choices=("high_res", "standard", "high_sensitivity"))
    ap.add_argument("--focus", default="full_focus", choices=tuple(FOCUS_REDUCE))
    ap.add_argument("--region", default="SSp")
    ap.add_argument("--thickness-um", type=float, default=50.0)
    args = ap.parse_args()

    cupy.get_default_memory_pool().set_limit(size=int(VRAM_LIMIT_GB * 1e9))
    px_um, fov_um, (nx, ny) = frame_spec(args.mag, args.readout)
    nz = max(1, int(round(args.thickness_um / LAB_DZ_UM)))
    scale = (LAB_DZ_UM, px_um, px_um)
    e = dl.region(args.region)
    print(f"{args.mag}x {args.readout}  FOV {fov_um[0]:.0f}x{fov_um[1]:.0f}um  "
          f"{nx}x{ny}px @ {px_um:.5f}um/px  focus={args.focus}")
    print(f"region {e['acronym']} ({e['name']})  {e['total']:.0f} cells/mm3  "
          f"neuron_frac {e['neuron_frac']:.2f}  slab {args.thickness_um:.0f}um")

    optics = ge.build_optics(config_sim(args.mag, px_um, args.focus, nz))
    halo = optics.psf.shape[-1] // 2
    print(f"PSF {tuple(optics.psf.shape)} -> halo {halo}px ({halo*px_um:.0f}um)")

    shapes = gt.load_shape_bank(SHAPES)
    rng = np.random.default_rng(0)
    # clean typing path: neuron_frac from the table (no deprecated size labels);
    # size_ratio + speck params come from gpu_truth/nuclei (10x-grounded).
    bank = gt.make_real_template_bank(
        shapes, rng, scale, n_templates=400, rotate=True,
        neuron_frac=e["neuron_frac"])

    t0 = time.perf_counter()
    truth, n = gt.stamp_nuclei(cupy, (nz, ny, nx), scale, e["total"], bank, rng)
    truth_h = ge.to_host(cupy, truth)
    del truth
    cupy.get_default_memory_pool().free_all_blocks()
    print(f"stamped {n} nuclei ({time.perf_counter()-t0:.1f}s); rendering...")

    t1 = time.perf_counter()
    plane, ntiles = render_tiled(truth_h, optics, halo, FOCUS_REDUCE[args.focus])
    cupy.cuda.Stream.null.synchronize()
    print(f"rendered {ntiles} tiles in {time.perf_counter()-t1:.1f}s")

    tag = f"{args.region}_{args.mag}x_{args.focus}"
    out = OUT / f"frame_{tag}.png"
    style_navy(plane.astype(np.float32), out, pct=(1, 99.6))
    print(f"wrote {out}  ({n} nuclei, {nx}x{ny}px)")


if __name__ == "__main__":
    main()
