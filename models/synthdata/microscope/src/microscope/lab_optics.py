"""Real optics for the Walsh-lab Keyence BZ-X800, 20x dry objective.

Objective: Nikon CFI Plan Apo lambda 20x (air / dry), NA 0.75, designed for a
#1.5 coverslip (170 um), NO oil. Sample is a DAPI-stained brain slice under a
coverslip. From the BZ-X800 FOV table (high-resolution mode):
  20x -> 0.37744 um/px lateral, FOV 724.69 x 543.52 um.

NaN gotcha (solved): ObjectiveLens.numerical_aperture has alias 'na' and
populate_by_name is OFF, so passing `numerical_aperture=` is IGNORED (NA stays
the 1.4 default) while the `_vroot` validator reads the `numerical_aperture`
key. We therefore pass BOTH `na=` (populates the field) and
`numerical_aperture=` (satisfies the validator). With NA actually 1.4 on an air
lens (ni=1.0), arcsin(1.4) -> NaN; with NA 0.75 it's fine.
"""
from __future__ import annotations

import warnings
from functools import cache

from microsim.schema.lens import ObjectiveLens

LAB_PX_UM = 0.37744               # 20x high-resolution lateral pixel size
LAB_FOV_UM = (724.69, 543.52)     # (X, Y) field of view, microns
LAB_DZ_UM = 0.5                   # z-step we simulate the slab at

# ---------------------------------------------------------------------------
# BZ-X800 magnification presets (from the official FOV-by-magnification table).
# A real acquisition is ONE sensor frame: FOV (um) is fixed per objective; the
# readout mode sets the pixel size (and thus the pixel dimensions of the frame).
# High-resolution = native sensor 1920x1440; Standard = 2x bin (960x720);
# High-sensitivity = 3x bin (640x480). Most whole-slice imaging is 10x or 20x.
BZX_FOV_UM = {                    # (X, Y) field of view in microns, per mag
    10: (1449.38, 1087.03),
    20: (724.69, 543.52),
}
BZX_PX_UM_HIRES = {              # high-resolution (native) lateral um/px, per mag
    10: 0.75488,
    20: 0.37744,
}
BZX_OBJ_NA = {10: 0.45, 20: 0.75}            # CFI Plan Apo lambda dry NA
BZX_OBJ_WD_UM = {10: 4000.0, 20: 2100.0}     # working distance (um), from table
READOUT_BIN = {"high_res": 1, "standard": 2, "high_sensitivity": 3}


def frame_spec(mag: int = 20, readout: str = "high_res"):
    """Resolve a real BZ-X frame: returns (px_um, fov_um, (nx, ny)).

    px_um = native um/px * readout bin; fov is fixed by the objective; the pixel
    dimensions follow from fov / px_um (e.g. 20x high_res -> 1920x1440 over
    724.69x543.52 um; 10x high_res -> 1920x1440 over 1449.38x1087.03 um).
    """
    if mag not in BZX_FOV_UM:
        raise ValueError(f"mag {mag} not in {sorted(BZX_FOV_UM)}")
    if readout not in READOUT_BIN:
        raise ValueError(f"readout {readout!r} not in {list(READOUT_BIN)}")
    px = BZX_PX_UM_HIRES[mag] * READOUT_BIN[readout]
    fov = BZX_FOV_UM[mag]
    nx, ny = int(round(fov[0] / px)), int(round(fov[1] / px))
    return px, fov, (nx, ny)

# FPbase "Keyence" microscope -> its real BZ-X filter cubes (Chroma ET sets +
# metal-halide source). NOTE this is the SPECTRAL channel only (ex/em filters,
# light source); the objective/coverslip/pixel size live in objective_20x_air()
# + the Detector, which FPbase does not carry.
BZX_FPBASE_SCOPE = "Khq2vpdFMCMdWVcYrizRT9"
BZX_CONFIG_NAMES = {
    "DAPI": "BZ-X Filter DAPI",
    "GFP": "BZ-X Filter GFP",
    "TRITC": "BZ-X Filter TRITC",
    "CY5": "BZ-X Filter Cy5",
}


@cache
def bzx_optical_config(channel: str = "DAPI"):
    """Pull the real BZ-X800 filter cube for `channel` from FPbase.

    DAPI -> Chroma ET360/40x (ex) + ET460/50m (em) + Metal Halide source.
    Falls back to microsim's built-in generic config if FPbase is unreachable
    (offline / API down) so a render never hangs or fails on the network. The
    result is cached for the process. FPbase also caches to disk after the first
    successful fetch, so warmed runs are offline-safe.
    """
    from microsim import schema as ms

    key = channel.upper()
    cfg_name = BZX_CONFIG_NAMES[key]
    try:
        return ms.OpticalConfig.from_fpbase(f"{BZX_FPBASE_SCOPE}::{cfg_name}")
    except Exception as exc:  # network down, API change, etc.
        from microsim.schema.optical_config import lib as oc_lib
        warnings.warn(
            f"FPbase fetch of {cfg_name!r} failed ({type(exc).__name__}: {exc}); "
            f"falling back to microsim's generic {key} config.", stacklevel=2
        )
        return getattr(oc_lib, key)


def objective_air(mag: int = 20, specimen_ri: float = 1.47) -> ObjectiveLens:
    """CFI Plan Apo lambda dry/air objective for `mag` (10 or 20).

    `specimen_ri` is the mounting-medium RI (ns): ~1.33 aqueous, ~1.40
    Fluoromount, ~1.47 ProLong-class hardset. It only weakly affects the single
    focal-plane PSF used in the depth-invariant convolution; it would matter
    more for depth-variant spherical aberration (not modeled by a single PSF).

    NA/WD per the BZ-X800 table: 10x -> NA 0.45, WD 4.0mm; 20x -> NA 0.75,
    WD 2.1mm. We pass BOTH `na=` and `numerical_aperture=` (the alias gotcha).
    """
    na = BZX_OBJ_NA[mag]
    return ObjectiveLens(
        na=na,                           # alias -> populates the field
        numerical_aperture=na,           # -> satisfies microsim's validator
        immersion_medium_ri=1.0,         # ni  (air)
        immersion_medium_ri_spec=1.0,    # ni0 (air, design)
        coverslip_ri=1.515,              # ng  (#1.5 coverslip)
        coverslip_ri_spec=1.515,         # ng0
        coverslip_thickness_um=170.0,    # tg  (#1.5 = 0.17 mm)
        coverslip_thickness_spec_um=170.0,
        specimen_ri=specimen_ri,         # ns  (mounting medium)
        working_distance_um=BZX_OBJ_WD_UM[mag],
        magnification=mag,
    )


def objective_20x_air(specimen_ri: float = 1.47) -> ObjectiveLens:
    """Back-compat alias for the 20x objective; see objective_air()."""
    return objective_air(20, specimen_ri=specimen_ri)
