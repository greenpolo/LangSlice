"""QC/demo: research-grounded WM rows (down-left), real BZ-X800, full + single-plane.

Renders the NEW wm_rows generator (segmented, wavy, Gamma-renewal, astro-spacer
interfascicular rows -- fimbria-grounded) through the real BZ-X800 widefield DAPI
optics, and emits BOTH focus modes from ONE convolution:

  * full-focus  : max-intensity projection over the z-slab (Keyence Full Focus / EDF)
  * single-plane: one optical section through the middle of the slab (non-full-focus)

Whole-frame convolution (no tiling -- the box has 128 GB DRAM). Blue-only output
(style_navy; DAPI is never white). Fiber azimuth base_deg=135 = down & to the left.

    .venv/Scripts/python.exe -m microscope.demo_wm_rows --mag 20 --readout high_res
    ... --mag 20 --readout high_res --thickness-um 40 --base-deg 135
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
from . import wm_rows as wmr
from .lab_optics import LAB_DZ_UM, bzx_optical_config, frame_spec, objective_air
from .render_shape_texture import style_navy

HERE = Path(__file__).parent
OUT = HERE / "out" / "wm"
OUT.mkdir(parents=True, exist_ok=True)
SHAPES = HERE / "assets" / "hippocampus_shapes.npy"

BINS = 3
EXPOSURE_MS = 200


def config_sim(mag: int, px_um: float, nz: int, max_au: int = 40, psf_lat: int = 160):
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
        objective_lens=objective_air(mag),
        channels=[bzx_optical_config("DAPI")],
        detector=ms.detectors.lib.ICX285,
        settings=ms.Settings(
            random_seed=0, np_backend="numpy",
            spectral_bins_per_emission_channel=BINS, max_psf_radius_aus=max_au,
        ),
    )


def render_both_focus(truth, optics, *, exposure_ms=EXPOSURE_MS, seed=1):
    """One whole-frame convolution -> (full_focus MIP, single mid-plane). Seam-free."""
    optical = ge.optical_image(truth, optics, np)            # (Z, Y, X) photons/s
    optical = ge.coarsen_sum(np, optical, optics.downscale)
    np.random.seed(seed)
    noisy = ge.apply_ccd(np, optical, optics.detector, exposure_ms)  # (Z, Y, X) uint16
    return noisy.max(axis=0), noisy[noisy.shape[0] // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="cc")
    ap.add_argument("--mag", type=int, default=20)
    ap.add_argument("--readout", default="high_res", help="high_res (default) | standard | high_sensitivity")
    ap.add_argument("--thickness-um", type=float, default=40.0)
    ap.add_argument("--base-deg", type=float, default=135.0, help="fiber azimuth; 135 = down-left")
    ap.add_argument("--density-mult", type=float, default=2.0)
    ap.add_argument("--wobble-sd-deg", type=float, default=6.0)
    ap.add_argument("--dip-sd-deg", type=float, default=0.0, help="out-of-plane tilt SD; 0=flat/traceable, ~20=realistic mess")
    ap.add_argument("--inplane-disp-deg", type=float, default=10.0, help="per-chain in-plane heading dispersion (fanning)")
    ap.add_argument("--chain-len-um", type=float, default=180.0, help="mean chain arc length (Exponential)")
    ap.add_argument("--mu-within", type=float, default=8.0)
    ap.add_argument("--seg-mean", type=float, default=8.0)
    ap.add_argument("--with-old", action="store_true", help="also render the old stamp_wm full-focus")
    ap.add_argument("--psf-lat", type=int, default=160)
    ap.add_argument("--max-au", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    px_um, fov_um, (nx, ny) = frame_spec(args.mag, args.readout)
    nz = max(1, int(round(args.thickness_um / LAB_DZ_UM)))
    scale = (LAB_DZ_UM, px_um, px_um)
    shape = (nz, ny, nx)
    e = dl.region(args.region)
    density = e["total"] * args.density_mult
    print(f"{args.mag}x {args.readout}: {nx}x{ny}px @ {px_um:.5f}um/px, "
          f"FOV {fov_um[0]:.0f}x{fov_um[1]:.0f}um, slab {args.thickness_um:.0f}um ({nz}z)")
    print(f"region {e['acronym']} ({e['name']}) Ero {e['total']:.0f} x{args.density_mult:g} "
          f"= {density:.0f} cells/mm3; base_deg {args.base_deg} (down-left)")

    optics = ge.build_optics(config_sim(args.mag, px_um, nz, args.max_au, args.psf_lat))
    print(f"PSF {tuple(optics.psf.shape)}")

    shapes = gt.load_shape_bank(SHAPES)
    rng = np.random.default_rng(args.seed)
    oligo_bank = wm.oligo_template_bank(shapes, rng, scale, n_templates=240)
    astro_bank = wm.wm_scatter_bank(shapes, rng, scale, n_templates=160, astro_frac=1.0)
    micro_bank = wm.wm_scatter_bank(shapes, rng, scale, n_templates=160, astro_frac=0.0)

    t0 = time.perf_counter()
    truth, n_new, counts = wmr.stamp_wm_rows(
        np, shape, scale, density, oligo_bank, astro_bank, micro_bank, rng,
        base_deg=args.base_deg, mu_within_um=args.mu_within, seg_mean=args.seg_mean,
        wobble_sd_deg=args.wobble_sd_deg, dip_sd_deg=args.dip_sd_deg,
        inplane_disp_deg=args.inplane_disp_deg, chain_len_mean_um=args.chain_len_um)
    print(f"stamped {n_new} cells {counts} in {time.perf_counter()-t0:.1f}s; convolving {nx}x{ny}x{nz}...")
    mip_truth = truth.max(0).copy()
    t1 = time.perf_counter()
    img_full, img_plane = render_both_focus(truth, optics)
    print(f"rendered full+plane in {time.perf_counter()-t1:.1f}s")
    del truth

    tag = f"{args.region}_{args.mag}x_{args.readout}_deg{int(args.base_deg)}"
    p_full = OUT / f"wm_NEW_fullfocus_{tag}.png"
    p_plane = OUT / f"wm_NEW_plane_{tag}.png"
    p_truth = OUT / f"wm_NEW_truth_{tag}.png"
    style_navy(img_full.astype(np.float32), p_full, pct=(1, 99.6))
    style_navy(img_plane.astype(np.float32), p_plane, pct=(1, 99.6))
    style_navy(mip_truth.astype(np.float32), p_truth, pct=(1, 99.6))

    panels = [
        (style_navy(mip_truth.astype(np.float32), path=None, pct=(1, 99.6)),
         f"placement (truth MIP) - {n_new} cells {counts}"),
        (style_navy(img_full.astype(np.float32), path=None, pct=(1, 99.6)),
         f"BZ-X {args.mag}x widefield FULL FOCUS (MIP over {args.thickness_um:.0f}um)"),
        (style_navy(img_plane.astype(np.float32), path=None, pct=(1, 99.6)),
         f"BZ-X {args.mag}x widefield SINGLE PLANE (non-full-focus)"),
    ]
    if args.with_old:
        scatter_bank = wm.wm_scatter_bank(shapes, rng, scale, n_templates=160)
        truth_old, n_old, _ = wm.stamp_wm(
            np, shape, scale, density, oligo_bank, scatter_bank, rng,
            row_pitch_um=9.0, bead_pitch_um=6.0, base_deg=args.base_deg)
        old_full, _ = render_both_focus(truth_old, optics)
        del truth_old
        panels.append((style_navy(old_full.astype(np.float32), path=None, pct=(1, 99.6)),
                       f"OLD stamp_wm full-focus - {n_old} cells"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(6.6 * ncol, 5.4), facecolor="black")
    if ncol == 1:
        axes = [axes]
    for ax, (rgb, title) in zip(axes, panels, strict=False):
        ax.imshow(rgb, interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"White matter DOWN-LEFT - {e['acronym']} {density:.0f} cells/mm3 - "
        f"{args.mag}x {args.readout} ({px_um:.3f} um/px), {args.thickness_um:.0f}um slab",
        color="white", fontsize=12)
    fig.tight_layout()
    p_cmp = OUT / f"wm_focus_compare_{tag}.png"
    fig.savefig(p_cmp, dpi=130, facecolor="black")
    for p in (p_truth, p_full, p_plane, p_cmp):
        print("WROTE", p)


if __name__ == "__main__":
    main()
