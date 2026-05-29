"""Render the improved DAPI nuclei with more of microsim's physics.

- 20x-style crisp view (downscale 1 -> 0.0705 um/px, confocal) to SHOW the new
  nucleus shape + chromatin texture.
- 10x-style widefield view (downscale 2 -> 0.141 um/px) matching the hazy ref.
- multi-bin spectral PSF (spectral_bins_per_emission_channel) = real chromatic blur.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from microsim import schema as ms
from microsim.schema.optical_config import lib as oc_lib

from .nuclei import build_truth

OUT = Path(__file__).parent / "out" / "shape"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = (0.5, 0.0705, 0.0705)
SHAPE = (90, 760, 760)        # 45um thick, ~54um FOV


def style_navy(img2d, path=None, pct=(0.5, 99.7), grad_ang=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = img2d.astype(np.float32)
    if grad_ang is not None:
        gy = np.linspace(-1, 1, img.shape[0])[:, None]
        gx = np.linspace(-1, 1, img.shape[1])[None, :]
        img = img * np.clip(1.0 - 0.26 * (np.cos(grad_ang) * gy
                            + np.sin(grad_ang) * gx), 0.55, 1.2)
    lo, hi = np.percentile(img, pct)
    n = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 1.25
    rgb = np.clip(np.stack([0.17 * n ** 2.5, 0.23 * n ** 2.0, 0.07 + 0.93 * n], -1), 0, 1)
    if path is not None:
        plt.imsave(path, rgb)
    return rgb


def make_sim(truth, dapi, downscale, modality, seed, bins=5):
    label = ms.FluorophoreDistribution(
        distribution=ms.FluorophoreDistribution.from_array(truth).distribution,
        fluorophore=dapi)
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=SHAPE, scale=SCALE),
        output_space=ms.DownscaledSpace(downscale=downscale),
        sample=ms.Sample(labels=[label]),
        modality=modality,
        objective_lens=ms.ObjectiveLens(numerical_aperture=1.4),
        channels=[oc_lib.DAPI],
        detector=ms.detectors.lib.ICX285,
        settings=ms.Settings(random_seed=seed, spectral_bins_per_emission_channel=bins),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Building improved nuclei truth...")
    t0 = time.time()
    truth, n, counts = build_truth(rng, SHAPE, SCALE, density_per_mm3=1.15e6)
    print(f"  {n} nuclei {counts} in {time.time()-t0:.0f}s")

    dapi = ms.Fluorophore.from_fpbase("DAPI")

    # 20x crisp confocal -> shows shape + chromatin
    print("Rendering 20x confocal (shape/texture showcase)...")
    t0 = time.time()
    conf = np.asarray(make_sim(truth, dapi, 1, ms.Confocal(pinhole_au=1.0),
                               args.seed).digital_image(exposure_ms=500))[0]
    zf = int(conf.shape[0] * 0.45)
    plane20 = conf[zf]
    style_navy(plane20, OUT / "dapi_20x_confocal.png")
    # zoom crop (center quarter) to reveal per-nucleus chromatin
    H, W = plane20.shape
    crop = plane20[H // 3:H // 3 + H // 4, W // 3:W // 3 + W // 4]
    style_navy(crop, OUT / "dapi_20x_crop.png", pct=(1, 99.8))
    print(f"  confocal in {time.time()-t0:.0f}s")

    # 10x widefield -> hazy ref2-style
    print("Rendering 10x widefield...")
    t0 = time.time()
    wf = np.asarray(make_sim(truth, dapi, 2, ms.Widefield(),
                             args.seed).digital_image(exposure_ms=180))[0]
    zf = int(wf.shape[0] * 0.22)
    style_navy(wf[zf], OUT / "dapi_10x_widefield.png", grad_ang=rng.uniform(0, 6.28))
    print(f"  widefield in {time.time()-t0:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
