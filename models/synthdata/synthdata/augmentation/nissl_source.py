"""``ara_nissl`` as a per-pixel cell-density source, aligned to the BrainGlobe annotation.

The Allen CCFv3 *average template* (``allen_mouse_25um.reference``) is a smooth,
low-frequency anatomical average that does not resolve per-region cell density.
The Allen *Nissl* volume (``ara_nissl``) sits on the identical 25 µm CCF voxel
grid and carries real cytoarchitecture — cortical laminae, hippocampal
pyramidal / dentate granule layers, fiber tracts. This module loads
``ara_nissl``, reorients it into the BrainGlobe atlas space (so it aligns
voxel-for-voxel with the annotation we slice for ground truth), and serves
coronal slices as a ``[0, 1]`` density source.

Label-safe: ``ara_nissl`` only feeds density / intensity modulation; region
boundaries always come from the (untouched) annotation.

Source: Allen Institute ``ara_nissl_25.nrrd`` (Allen Terms of Use, noncommercial
research — the same license as the ``allen_mouse_25um`` atlas we already use).

The reorientation from native Allen axes to BrainGlobe axes is a pure
permutation + flips (both are 25 µm isotropic on the same grid). It is derived
empirically by matching the native ``average_template`` against the in-memory
``atlas.reference`` (the same data, already reoriented by BrainGlobe), then
applied to ``ara_nissl``. The derived transform + aligned volume are cached.
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ALLEN_BASE = (
    "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"
)
_FILES = {
    "average_template_25.nrrd": f"{_ALLEN_BASE}/average_template/average_template_25.nrrd",
    "ara_nissl_25.nrrd": f"{_ALLEN_BASE}/ara_nissl/ara_nissl_25.nrrd",
}

# In-process cache of the aligned volume, keyed by atlas name.
_volume_cache: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    """Resolve ``<repo>/_local/synth_data/nissl_density`` (shared across worktrees)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "_local").is_dir():
            d = parent / "_local" / "synth_data" / "nissl_density"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = here.parent / "_nissl_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_nrrd(path: Path) -> np.ndarray:
    try:
        import nrrd

        data, _ = nrrd.read(str(path))
        return np.asarray(data)
    except ImportError:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _ensure_native(fname: str) -> Path:
    """Return a local path to a native Allen NRRD, copying/downloading as needed."""
    p = _cache_dir() / fname
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    # Reuse an already-downloaded copy from the temp QC scratch dir if present.
    tmp = Path(tempfile.gettempdir()) / "synthvar" / fname
    if tmp.exists() and tmp.stat().st_size > 1_000_000:
        shutil.copy2(tmp, p)
        return p
    logger.info("downloading %s", fname)
    urllib.request.urlretrieve(_FILES[fname], p)
    return p


def _derive_orientation(
    native_template: np.ndarray, atlas_reference: np.ndarray
) -> tuple[tuple[int, ...], tuple[bool, ...], float]:
    """Find the (perm, flips) mapping native Allen axes → BrainGlobe axes.

    Axis lengths of the Allen 25 µm grid (528/320/456) are all distinct, so the
    permutation is uniquely fixed by matching shapes; flips are chosen by the
    best correlation against the (already-reoriented) ``atlas_reference``.
    """
    nshape = tuple(int(s) for s in native_template.shape)
    tshape = tuple(int(s) for s in atlas_reference.shape)
    perm = tuple(nshape.index(length) for length in tshape)
    nt = np.transpose(native_template, perm)

    ref_ds = atlas_reference[::8, ::8, ::8].astype(np.float32).ravel()
    best_corr, best_flips = -2.0, (False, False, False)
    for flips in itertools.product([False, True], repeat=3):
        cand = nt
        for ax, flip in enumerate(flips):
            if flip:
                cand = np.flip(cand, ax)
        cand_ds = cand[::8, ::8, ::8].astype(np.float32).ravel()
        if cand_ds.shape != ref_ds.shape:
            continue
        corr = float(np.corrcoef(cand_ds, ref_ds)[0, 1])
        if corr > best_corr:
            best_corr, best_flips = corr, flips
    return perm, best_flips, best_corr


def _reorient(vol: np.ndarray, perm: tuple[int, ...], flips: tuple[bool, ...]) -> np.ndarray:
    out = np.transpose(vol, perm)
    for ax, flip in enumerate(flips):
        if flip:
            out = np.flip(out, ax)
    return np.ascontiguousarray(out)


def load_nissl_volume(atlas: object) -> np.ndarray:
    """Return ``ara_nissl`` reoriented into BrainGlobe space, float32 in [0, 1].

    Same shape/orientation as ``atlas.annotation``. Built + cached on first call.
    """
    key = str(getattr(atlas, "atlas_name", "allen_mouse_25um"))
    if key in _volume_cache:
        return _volume_cache[key]

    cache = _cache_dir()
    npy = cache / f"ara_nissl_bg_{key}.npy"
    if npy.exists():
        vol = np.load(npy)
        _volume_cache[key] = vol
        return vol

    reference = np.asarray(atlas.reference)  # type: ignore[attr-defined]
    native_template = _read_nrrd(_ensure_native("average_template_25.nrrd"))
    native_nissl = _read_nrrd(_ensure_native("ara_nissl_25.nrrd"))

    perm, flips, corr = _derive_orientation(native_template, reference)
    if corr < 0.5:
        raise RuntimeError(
            f"ara_nissl orientation match is weak (corr={corr:.3f}); refusing to "
            "cache a possibly-misaligned volume."
        )

    nissl = _reorient(native_nissl, perm, flips).astype(np.float32)
    if nissl.shape != reference.shape:
        raise RuntimeError(f"aligned nissl {nissl.shape} != reference {reference.shape}")

    # Robust normalization to [0, 1] (99.5th pct of tissue, clip outliers).
    nonzero = nissl[nissl > 0]
    scale = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
    nissl = np.clip(nissl / (scale or 1.0), 0.0, 1.0).astype(np.float32)

    np.save(npy, nissl)
    (cache / f"transform_{key}.json").write_text(
        json.dumps({"perm": list(perm), "flips": [bool(f) for f in flips], "corr": corr})
    )
    logger.info("built aligned ara_nissl for %s (perm=%s flips=%s corr=%.3f)", key, perm, flips, corr)
    _volume_cache[key] = nissl
    return nissl


def nissl_slice(atlas: object, idx: int, axis: int) -> np.ndarray:
    """Coronal ``ara_nissl`` slice (HW float32 in [0, 1]) aligned to the annotation.

    ``idx`` / ``axis`` must be the same values used to slice ``atlas.annotation``.
    """
    vol = load_nissl_volume(atlas)
    return np.ascontiguousarray(np.take(vol, idx, axis=axis)).astype(np.float32)
