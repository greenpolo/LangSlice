"""QC: white-matter architecture (interfascicular oligo rows) vs uniform stamping.

CPU-only (no cupy / no GPU touch -- the lab GPU is busy serving). Builds a small
white-matter patch (region cc = corpus callosum, neuron_frac 0) two ways and
renders both through the SAME real BZ-X800 widefield DAPI optics:

    A) WM rows  -- white_matter.stamp_wm   (fiber-aligned interfascicular rows)
    B) uniform  -- gpu_truth.stamp_nuclei  (current behaviour)

plus a crisp truth-MIP of (A) to show the underlying placement before optics.
Output is blue-only (style_navy). Compare panel A against the real exemplars in
models/langslice-gemma-4/data/augmentation/GT_Textures/DAPI/DAPI_whitematter_*.png

    .venv/Scripts/python.exe demo_wm_arch.py
    .venv/Scripts/python.exe demo_wm_arch.py --nxy 480 --nz 60 --base-deg 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from microsim import schema as ms

from . import density_lookup as dl
from . import gpu_engine as ge
from . import gpu_truth as gt
from . import white_matter as wm
from .lab_optics import LAB_DZ_UM, bzx_optical_config, frame_spec, objective_air
from .render_shape_texture import style_navy

HERE = Path(__file__).parent
OUT = HERE / "out" / "wm"
OUT.mkdir(parents=True, exist_ok=True)
SHAPES = HERE / "assets" / "hippocampus_shapes.npy"

MAG = 20
BINS = 3
EXPOSURE_MS = 200


def config_sim(px_um: float, nz: int, max_au: int, psf_lat: int = 192):
    # psf_lat sets the PSF lateral footprint (microsim pins it to the config
    # truth-space lateral dim, not max_au); smaller = faster CPU FFT + tighter
    # out-of-focus halo (fine at the zoom-IN scale we QC here).
    scale = (LAB_DZ_UM, px_um, px_um)
    dummy = np.zeros((nz, psf_lat, psf_lat), np.float32)
    label = ms.FluorophoreDistribution(
        distribution=ms.FluorophoreDistribution.from_array(dummy).distribution,
        fluorophore=ms.Fluorophore.from_fpbase("DAPI"),
    )
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(nz, psf_lat, psf_lat), scale=scale),
        output_space=ms.DownscaledSpace(downscale=1),
        sample=ms.Sample(labels=[label]),
        modality=ms.Widefield(),
        objective_lens=objective_air(MAG),
        channels=[bzx_optical_config("DAPI")],
        detector=ms.detectors.lib.ICX285,
        settings=ms.Settings(
            random_seed=0, np_backend="numpy",
            spectral_bins_per_emission_channel=BINS, max_psf_radius_aus=max_au,
        ),
    )


def _panel(img2d, title):
    return style_navy(img2d.astype(np.float32), path=None, pct=(1, 99.6)), title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="cc")
    ap.add_argument("--nxy", type=int, default=560)
    ap.add_argument("--nz", type=int, default=100, help="z planes (x0.5um); 100 = real 50um slab")
    ap.add_argument("--max-au", type=int, default=40)
    ap.add_argument("--psf-lat", type=int, default=192, help="PSF lateral footprint px (speed)")
    ap.add_argument("--density-mult", type=float, default=5.0,
                    help="WM cell-density multiplier on the Ero table value (Ero undercounts WM)")
    ap.add_argument("--px-um", type=float, default=0.46, help="render pixel size um/px (default ~ real QuPath images)")
    ap.add_argument("--row-pitch", type=float, default=9.0, help="axon (oligo-chain) spacing across fibers (um)")
    ap.add_argument("--bead-pitch", type=float, default=6.0, help="oligo spacing along the axon (um, ~touching @ 6.5um nucleus)")
    ap.add_argument("--scatter-frac", type=float, default=0.2, help="fraction of cells that are random astro+microglia")
    ap.add_argument("--base-deg", type=float, default=0.0, help="dominant fiber azimuth (0=horizontal)")
    ap.add_argument("--wobble", type=float, default=0.0, help="fiber curvature std deg (0=straight; for real curved tracts)")
    ap.add_argument("--row-jitter", type=float, default=1.5, help="per-axon perpendicular jitter (um)")
    ap.add_argument("--bead-jitter", type=float, default=1.5, help="per-oligo jitter (um)")
    ap.add_argument("--bg", type=float, default=0.10, help="dark background floor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    px_um = args.px_um if args.px_um > 0 else frame_spec(MAG, "high_res")[0]
    scale = (LAB_DZ_UM, px_um, px_um)
    shape = (args.nz, args.nxy, args.nxy)
    fov_um = args.nxy * px_um
    e = dl.region(args.region)
    density = e["total"] * args.density_mult
    print(f"region {e['acronym']} ({e['name']})  Ero {e['total']:.0f} x{args.density_mult:g} "
          f"= {density:.0f} cells/mm3  neuron_frac {e['neuron_frac']:.2f}")
    print(f"patch {args.nxy}x{args.nxy}px @ {px_um:.5f}um/px = {fov_um:.0f}x{fov_um:.0f}um, "
          f"{args.nz} z ({args.nz*LAB_DZ_UM:.0f}um slab)  row_pitch {args.row_pitch}um bead {args.bead_pitch}um")

    optics = ge.build_optics(config_sim(px_um, args.nz, args.max_au, args.psf_lat))
    print(f"PSF {tuple(optics.psf.shape)}")

    shapes = gt.load_shape_bank(SHAPES)
    rng = np.random.default_rng(args.seed)
    # two populations: small round oligodendrocytes (form the chains) + a scatter
    # bank of larger-dim astrocytes and irregular microglia (interspersed).
    oligo_bank = wm.oligo_template_bank(shapes, rng, scale, n_templates=300)  # size from table
    scatter_bank = wm.wm_scatter_bank(shapes, rng, scale, n_templates=200)

    # white matter: oligo chains + astro/microglia scatter, rendered two ways
    t0 = time.perf_counter()
    truth_wm, n_wm, _ = wm.stamp_wm(
        np, shape, scale, density, oligo_bank, scatter_bank, rng,
        row_pitch_um=args.row_pitch, bead_pitch_um=args.bead_pitch,
        row_jitter_um=args.row_jitter, bead_jitter_um=args.bead_jitter,
        scatter_frac=args.scatter_frac,
        base_deg=args.base_deg, wobble_deg=args.wobble, bg_count=args.bg)
    mip_truth = truth_wm.max(0).copy()
    img_full = ge.render(truth_wm, optics, np, exposure_ms=EXPOSURE_MS, reduce="mip", seed=1)
    img_plane = ge.render(truth_wm, optics, np, exposure_ms=EXPOSURE_MS, reduce="plane", seed=1)
    del truth_wm
    style_navy(img_full.astype(np.float32), OUT / f"wm_fullfocus_{args.region}_{args.nxy}px.png", pct=(1, 99.6))
    style_navy(img_plane.astype(np.float32), OUT / f"wm_plane_{args.region}_{args.nxy}px.png", pct=(1, 99.6))
    style_navy(mip_truth.astype(np.float32), OUT / f"wm_truth_{args.region}_{args.nxy}px.png", pct=(1, 99.6))
    print(f"WM: {n_wm} cells, rendered in {time.perf_counter()-t0:.1f}s")

    # compose labeled side-by-side (blue-only)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        _panel(mip_truth, f"placement (truth MIP) - {n_wm} cells"),
        _panel(img_plane, "BZ-X 20x widefield - single plane"),
        _panel(img_full, "BZ-X 20x widefield - full-focus (MIP)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    for ax, (rgb, title) in zip(axes, panels, strict=False):
        ax.imshow(rgb, interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"White-matter DAPI - {e['acronym']}  |  round oligo chains + astro/microglia scatter  |  "
        f"{density:.0f} cells/mm3, {args.nz*LAB_DZ_UM:.0f}um slab, 20x",
        color="white", fontsize=11)
    fig.tight_layout()
    out = OUT / f"wm_arch_{args.region}_{args.nxy}px.png"
    fig.savefig(out, dpi=130, facecolor="black")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
