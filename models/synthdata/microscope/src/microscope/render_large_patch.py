"""Render a LARGE single coherent DAPI patch via halo-tiled GPU convolution.

A single 1024px patch already peaks ~14GB on the widefield FFT, so a bigger
field can't be convolved in one shot. Instead we:

  1. STAMP the whole truth volume once (one coherent field -> nuclei are placed
     consistently across the whole patch, no per-tile reseeding seams), on GPU,
     then move it to host and free the pool.
  2. RENDER in core tiles, each read with a PSF-radius HALO so every core pixel
     gets the full out-of-focus contribution from its neighbours; convolve the
     padded sub-volume, take the focal plane, crop the halo, stitch the core.

The halo (= half the PSF width, ~161px here) is what makes the seams invisible:
the widefield haze from a nucleus just outside a tile still lands inside it.

    python render_large_patch.py                # SSp, 2048px (~0.77mm)
    python render_large_patch.py SSp 3072
"""
from __future__ import annotations

import sys
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
    LAB_PX_UM,
    bzx_optical_config,
    objective_20x_air,
)
from .render_shape_texture import style_navy  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out" / "gpu"
OUT.mkdir(parents=True, exist_ok=True)
SHAPES = HERE / "assets" / "hippocampus_shapes.npy"
LABELS = HERE / "assets" / "hippocampus_labels.npy"

SCALE = (LAB_DZ_UM, LAB_PX_UM, LAB_PX_UM)
NZ = 100              # 50um slab
MAX_AU = 40
BINS = 3
CORE = 512            # core tile size (px); footprint = CORE + 2*halo (~834)
VRAM_LIMIT_GB = 18.0
EXPOSURE_MS = 200


def config_sim():
    dummy = np.zeros((NZ, 320, 320), np.float32)
    label = ms.FluorophoreDistribution(
        distribution=ms.FluorophoreDistribution.from_array(dummy).distribution,
        fluorophore=ms.Fluorophore.from_fpbase("DAPI"),
    )
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(NZ, 320, 320), scale=SCALE),
        output_space=ms.DownscaledSpace(downscale=1),
        sample=ms.Sample(labels=[label]),
        modality=ms.Widefield(),
        objective_lens=objective_20x_air(),
        channels=[bzx_optical_config("DAPI")],
        detector=ms.detectors.lib.ICX285,
        settings=ms.Settings(
            random_seed=0, np_backend="numpy",
            spectral_bins_per_emission_channel=BINS, max_psf_radius_aus=MAX_AU,
        ),
    )


def render_tiled(truth_h, optics, halo):
    """Convolve the big truth in CORE tiles with a `halo` margin; stitch planes."""
    nz, ny, nx = truth_h.shape
    out = np.zeros((ny, nx), dtype=np.uint16)
    tile_id = 0
    for y in range(0, ny, CORE):
        for x in range(0, nx, CORE):
            y1, x1 = min(y + CORE, ny), min(x + CORE, nx)
            ya, yb = max(y - halo, 0), min(y1 + halo, ny)
            xa, xb = max(x - halo, 0), min(x1 + halo, nx)
            sub = cupy.asarray(truth_h[:, ya:yb, xa:xb])
            img = ge.render(sub, optics, cupy, exposure_ms=EXPOSURE_MS,
                            reduce="plane", seed=tile_id)
            img_h = ge.to_host(cupy, img)
            cy, cx = y - ya, x - xa            # core offset inside padded tile
            out[y:y1, x:x1] = img_h[cy:cy + (y1 - y), cx:cx + (x1 - x)]
            del sub, img
            cupy.get_default_memory_pool().free_all_blocks()
            tile_id += 1
    return out, tile_id


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "SSp"
    nxy = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    cupy.get_default_memory_pool().set_limit(size=int(VRAM_LIMIT_GB * 1e9))

    e = dl.region(key)
    mm = nxy * SCALE[1] / 1000
    print(f"region {e['acronym']} ({e['name']})  {e['total']:.0f} cells/mm3  "
          f"neuron_frac {e['neuron_frac']:.2f}")
    print(f"patch {mm:.2f}x{mm:.2f}mm ({nxy}px, {NZ*SCALE[0]:.0f}um thick), "
          f"core {CORE}px, VRAM cap {VRAM_LIMIT_GB:.0f}GB")

    optics = ge.build_optics(config_sim())
    halo = optics.psf.shape[-1] // 2
    print(f"PSF {tuple(optics.psf.shape)} -> halo {halo}px ({halo*SCALE[1]:.0f}um)")

    shapes = gt.load_shape_bank(SHAPES)
    labels = np.load(LABELS) if LABELS.exists() else None
    rng = np.random.default_rng(0)
    bank = gt.make_real_template_bank(
        shapes, rng, SCALE, n_templates=400, rotate=True,
        labels=labels, neuron_frac=e["neuron_frac"])

    t0 = time.perf_counter()
    truth, n = gt.stamp_nuclei(cupy, (NZ, nxy, nxy), SCALE, e["total"], bank, rng)
    truth_h = ge.to_host(cupy, truth)
    del truth
    cupy.get_default_memory_pool().free_all_blocks()
    print(f"stamped {n} nuclei ({time.perf_counter()-t0:.1f}s); rendering tiles...")

    t1 = time.perf_counter()
    plane, ntiles = render_tiled(truth_h, optics, halo)
    cupy.cuda.Stream.null.synchronize()
    print(f"rendered {ntiles} tiles in {time.perf_counter()-t1:.1f}s "
          f"(total {time.perf_counter()-t0:.1f}s)")

    out = OUT / f"large_patch_{key}_{nxy}.png"
    style_navy(plane.astype(np.float32), out, pct=(1, 99.6))
    print(f"wrote {out}  ({n} nuclei over {mm:.2f}x{mm:.2f}mm)")


if __name__ == "__main__":
    main()
