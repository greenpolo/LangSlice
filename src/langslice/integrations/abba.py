"""LangSlice as an ABBA registration plugin.

Registers LangSlice's generative-image nonlinear registration into a running
abba-python session via ABBA's ``ExternalRegistrationPlugin`` socket
(``SimpleRegistrationPlugin`` + ``SimpleRegistrationWrapper``). ABBA hands the
plugin two images resampled onto a common pixel grid; the plugin returns an
``InvertibleRealTransform`` mapping fixed pixels to moving pixels, and ABBA
appends it to the slice's registration stack like any native elastix/BigWarp
step (undoable, serialized into the state file).

The fixed image requested for the registration is ABBA's atlas *coordinate*
channels — per-pixel (AP, DV, ML) atlas position in mm — so the adapter never
assumes an axis convention, offset, or slicing angle: it samples the
BrainGlobe atlas volumes at exactly the coordinates ABBA reports and builds
the colored region map + grayscale reference on ABBA's own grid. The moving
image is the histology channel. Image-gen + Elastix then produce the
fixed→moving pixel transform, returned as a thin-plate spline over a landmark
grid sampled from the dense Elastix deformation field.

Usage (inside an abba-python session)::

    from abba_python.abba import Abba
    from langslice.integrations.abba import (
        enable_langslice_registration,
        register_selected_slices,
    )

    abba = Abba("Adult Mouse Brain - Allen Brain Atlas V3p1")
    enable_langslice_registration(abba, atlas_name="allen_mouse_10um")
    # ... import + position slices, select them, then:
    register_selected_slices(abba)

Reopening a saved state requires ``enable_langslice_registration`` to have
been called first in that session, otherwise ABBA drops the LangSlice step
(slice and position survive).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

REGISTRATION_NAME = "LangSlice-Nonlinear"

# Channel layout of ABBA's resliced atlas sources (legacy Allen CCFv3p1 java
# atlas): 0 Nissl, 1 template, 2 ?, 3 AP-mm, 4 DV-mm, 5 ML-mm, 6 left/right,
# 7 label ids. Verified empirically in _local/abba_spike/spike3*.
# ponytail: indices hardcoded for the legacy Allen atlas; make discoverable
# per-atlas when a second atlas is actually used in ABBA.
COORD_CHANNELS = (3, 4, 5)
DEFAULT_MOVING_CHANNEL = 0

_MAX_ATTEMPTS = 3


@dataclass
class LangSliceAbbaConfig:
    atlas_name: str = "allen_mouse_10um"
    provider: str = "google"
    model: str | None = None
    route: str | None = None
    voxel_size_um: float = 40.0
    landmark_grid: int = 14


_config = LangSliceAbbaConfig()


# ---------------------------------------------------------------------------
# Pure-numpy core (no JVM) — unit-testable
# ---------------------------------------------------------------------------


def render_atlas_at_coords(
    atlas: Any, coords_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a BrainGlobe atlas at per-pixel (AP, DV, ML) mm coordinates.

    Returns ``(colored_rgb uint8, reference_gray uint8, annotation_ids)`` on
    the coordinate grid. Out-of-volume pixels are background (id 0, black).
    """
    from langslice.atlas.recolor import color_lut

    res_mm = np.asarray(atlas.resolution, dtype=np.float64) / 1000.0
    shape = np.asarray(atlas.annotation.shape)

    idx = np.floor(coords_mm / res_mm[None, None, :]).astype(np.int64)
    oob = np.any((idx < 0) | (idx >= shape[None, None, :]), axis=2)
    idx_c = np.clip(idx, 0, shape - 1)

    ids = np.asarray(atlas.annotation[idx_c[..., 0], idx_c[..., 1], idx_c[..., 2]])
    ids = ids.astype(np.int64)
    ids[oob] = 0

    ref = np.asarray(
        atlas.reference[idx_c[..., 0], idx_c[..., 1], idx_c[..., 2]], dtype=np.float64
    )
    ref[oob] = 0.0
    peak = float(ref.max())
    reference_gray = (255.0 * ref / peak).astype(np.uint8) if peak > 0 else ref.astype(np.uint8)

    lut = color_lut(atlas)
    colored = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for uid in np.unique(ids):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        colored[ids == uid_int] = lut.get(uid_int, (128, 128, 128))

    return colored, reference_gray, ids


def classify_to_ids(rgb: np.ndarray, ids_present: np.ndarray, lut: dict) -> np.ndarray:
    """Nearest-color classification of an RGB image against the LUT colors of
    the region ids present on this section. Near-black pixels are background."""
    palette_ids = [int(u) for u in np.unique(ids_present) if int(u) != 0 and int(u) in lut]
    if not palette_ids:
        return np.zeros(rgb.shape[:2], dtype=np.int64)
    palette = np.array([lut[u] for u in palette_ids], dtype=np.float32)
    pix = rgb.reshape(-1, 3).astype(np.float32)
    background = np.max(pix, axis=1) < 20.0
    nearest = np.argmin(
        ((pix[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    out = np.asarray(palette_ids, dtype=np.int64)[nearest]
    out[background] = 0
    return out.reshape(rgb.shape[:2])


def landmarks_from_field(
    field: np.ndarray, mask: np.ndarray, grid: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed→moving landmark pairs from a dense (H, W, 2) displacement
    field on a regular grid restricted to *mask* (atlas foreground)."""
    h, w = mask.shape
    ys = np.linspace(0, h - 1, grid).round().astype(int)
    xs = np.linspace(0, w - 1, grid).round().astype(int)
    src, tgt = [], []
    for y in ys:
        for x in xs:
            if not mask[y, x]:
                continue
            dx, dy = float(field[y, x, 0]), float(field[y, x, 1])
            src.append((float(x), float(y)))
            tgt.append((float(x) + dx, float(y) + dy))
    if len(src) < 4:
        raise RuntimeError(
            f"Too few landmarks inside tissue ({len(src)}); registration field unusable"
        )
    return np.asarray(src), np.asarray(tgt)


def normalize_to_rgb(img: np.ndarray) -> Image.Image:
    """Percentile-normalize a raw single-channel image to an 8-bit RGB PIL."""
    data = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(data, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    gray = (255.0 * scaled).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


def _dump_debug_images(attempt: int, **images: np.ndarray) -> None:
    """Write attempt images under LANGSLICE_VLM_DEBUG_DIR/abba, if set."""
    root = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    if not root:
        return
    out = os.path.join(root, "abba", f"attempt_{attempt}")
    os.makedirs(out, exist_ok=True)
    for name, data in images.items():
        Image.fromarray(np.asarray(data)).save(os.path.join(out, f"{name}.png"))


def compute_registration_landmarks(
    coords_mm: np.ndarray,
    histology: np.ndarray,
    config: LangSliceAbbaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Full LangSlice nonlinear pass on ABBA-provided rasters.

    Returns (src, tgt) landmark arrays in fixed-grid pixels such that the
    mapping src→tgt carries a fixed (atlas) pixel to its moving (histology)
    pixel. Raises on Elastix error codes after exhausting attempts.
    """
    from langslice.atlas.core import load_atlas
    from langslice.atlas.recolor import color_lut
    from langslice.nonlinear.image_gen_helpers import (
        _compute_deformation_field,
        _elastix_report,
        _register_rgb_pair,
        _segmentation_prompt_for_plane,
        _warp_atlas_rgb,
    )
    from langslice.nonlinear.providers import (
        SegmentationGenerationRequest,
        generate_warped_segmentation_image,
    )

    atlas = load_atlas(config.atlas_name)
    colored, reference_gray, ids = render_atlas_at_coords(atlas, coords_mm)
    h, w = ids.shape
    slice_image = normalize_to_rgb(histology)

    lut = color_lut(atlas)
    prompt = _segmentation_prompt_for_plane("coronal")

    last_codes: list[Any] = []
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        generated = generate_warped_segmentation_image(
            SegmentationGenerationRequest(
                colored_regions=Image.fromarray(colored, mode="RGB"),
                reference_slice=Image.fromarray(reference_gray, mode="L"),
                slice_image=slice_image,
                prompt=prompt,
                provider=config.provider,
                model=config.model,
                route=config.route,
            )
        )
        model_rgb = np.asarray(
            generated.image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        )
        # fixed = atlas render, moving = model output (histology-shaped):
        # the resulting transform maps fixed pixels -> moving pixels, which is
        # exactly the direction ABBA's plugin contract requires.
        transform, _elapsed = _register_rgb_pair(colored, model_rgb)
        field = _compute_deformation_field(transform, model_rgb[:, :, 0])
        if field is None:
            last_codes = [{"code": "NO_DEFORMATION_FIELD"}]
            continue

        warped_model = _warp_atlas_rgb(model_rgb, transform)
        _dump_debug_images(
            attempt,
            colored=colored,
            reference=reference_gray,
            histology=np.asarray(slice_image),
            model_output=model_rgb,
            warped_model=warped_model,
        )
        # Both sides classified through the same palette: Allen family colors
        # are shared across sibling regions, so raw annotation ids would read
        # as REGION_MISSING for every collapsed sibling.
        report = _elastix_report(
            atlas_classified=classify_to_ids(colored, ids, lut),
            warped_classified=classify_to_ids(warped_model, ids, lut),
            structures=getattr(atlas, "structures", None),
            deformation_field=field,
        )
        codes = report.get("codes", [])
        if not codes:
            return landmarks_from_field(field, ids != 0, config.landmark_grid)
        last_codes = codes
        logger.warning(
            "LangSlice-ABBA attempt %d/%d refused by Elastix report: %s",
            attempt,
            _MAX_ATTEMPTS,
            codes,
        )

    raise RuntimeError(f"Registration refused after {_MAX_ATTEMPTS} attempts: {last_codes}")


# ---------------------------------------------------------------------------
# Java-facing side (requires a running abba-python JVM)
# ---------------------------------------------------------------------------


def _imageplus_to_numpy(imp: Any, ij_class: Any) -> np.ndarray:
    """ImagePlus -> numpy. ponytail: tiff round-trip via IJ.saveAsTiff +
    tifffile — bulletproof across pixel types; replace with a direct pixel
    bridge only if these tiny rasters ever become a measured bottleneck."""
    import tifffile  # pyright: ignore[reportMissingImports]

    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        ij_class.saveAsTiff(imp, path)
        return tifffile.imread(path)
    finally:
        os.unlink(path)


def _build_java_tps(src: np.ndarray, tgt: np.ndarray) -> Any:
    """Landmarks -> invertible 3D-wrapped thin-plate-spline RealTransform."""
    from jpype import JArray, JDouble  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    ThinplateSplineTransform = jimport("net.imglib2.realtransform.ThinplateSplineTransform")
    WrappedIterative = jimport(
        "net.imglib2.realtransform.inverse.WrappedIterativeInvertibleRealTransform"
    )
    # The invertible variant (from bigwarp) — plain Wrapped2DTransformAs3D is
    # not an InvertibleRealTransform, and the plugin contract requires one.
    Wrapped2DAs3D = jimport("net.imglib2.realtransform.InvertibleWrapped2DTransformAs3D")

    def as_java(points: np.ndarray) -> Any:
        return JArray(JArray(JDouble))(
            [points[:, 0].tolist(), points[:, 1].tolist()]
        )

    tps = ThinplateSplineTransform(as_java(src), as_java(tgt))
    # Verify apply() direction on the first landmark; TPS interpolates its
    # landmarks exactly, so the correct orientation reproduces tgt.
    probe_in = JArray(JDouble)([float(src[0, 0]), float(src[0, 1])])
    probe_out = JArray(JDouble)([0.0, 0.0])
    tps.apply(probe_in, probe_out)
    err = np.hypot(float(probe_out[0]) - tgt[0, 0], float(probe_out[1]) - tgt[0, 1])
    if err > 1e-3:
        tps = ThinplateSplineTransform(as_java(tgt), as_java(src))
    return Wrapped2DAs3D(WrappedIterative(tps))


def enable_langslice_registration(abba: Any, **overrides: Any) -> str:
    """Register the LangSlice nonlinear plugin type into a running session.

    Keyword overrides update :class:`LangSliceAbbaConfig` fields (atlas_name,
    provider, model, route, voxel_size_um, landmark_grid). Returns the
    registration type name. Must also be called before ``state_load`` of a
    project containing LangSlice registrations.
    """
    global _config
    for key, value in overrides.items():
        if not hasattr(_config, key):
            raise TypeError(f"Unknown LangSlice ABBA option: {key}")
        setattr(_config, key, value)

    from jpype import JImplements, JOverride  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    IJ = jimport("ij.IJ")
    MultiSlicePositioner = jimport("ch.epfl.biop.atlas.aligner.MultiSlicePositioner")
    SimpleRegistrationWrapper = jimport(
        "ch.epfl.biop.registration.plugin.SimpleRegistrationWrapper"
    )
    Supplier = jimport("java.util.function.Supplier")

    @JImplements("ch.epfl.biop.registration.plugin.SimpleRegistrationPlugin")
    class LangSlicePlugin:
        @JOverride
        def getVoxelSizeInMicron(self):
            return float(_config.voxel_size_um)

        @JOverride
        def setRegistrationParameters(self, parameters):
            pass  # the wrapper keeps ROI params to itself; nothing reaches us

        @JOverride
        def register(self, fixed, moving, fixedMask, movingMask):
            # Exceptions crossing back into Java surface only as an opaque
            # PyExceptionProxy — log the Python traceback before re-raising.
            try:
                coords = _imageplus_to_numpy(fixed, IJ)  # (3, H, W): AP, DV, ML mm
                histology = _imageplus_to_numpy(moving, IJ)
                if histology.ndim == 3:
                    histology = histology[0]
                coords_mm = np.stack([coords[0], coords[1], coords[2]], axis=-1)
                src, tgt = compute_registration_landmarks(coords_mm, histology, _config)
                logger.info("LangSlice-ABBA registration: %d landmarks", len(src))
                return _build_java_tps(src, tgt)
            except Exception:
                logger.exception("LangSlice-ABBA registration failed")
                raise

    @JImplements(Supplier)
    class LangSliceSupplier:
        @JOverride
        def get(self):
            return SimpleRegistrationWrapper(REGISTRATION_NAME, LangSlicePlugin())

    MultiSlicePositioner.registerRegistrationPlugin(REGISTRATION_NAME, LangSliceSupplier())
    logger.info("Registered ABBA registration plugin: %s", REGISTRATION_NAME)
    return REGISTRATION_NAME


def register_selected_slices(
    abba: Any, moving_channel: int = DEFAULT_MOVING_CHANNEL
) -> None:
    """Apply LangSlice nonlinear registration to the selected slices."""
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    HashMap = jimport("java.util.HashMap")
    SourcesChannelsSelect = jimport(
        "ch.epfl.biop.sourceandconverter.processor.SourcesChannelsSelect"
    )
    abba.mp.registerSelectedSlices(
        REGISTRATION_NAME,
        SourcesChannelsSelect(list(COORD_CHANNELS)),
        SourcesChannelsSelect(moving_channel),
        HashMap(),
    )
    abba.wait_for_end_of_tasks()


GUI_COMMAND_NAME = "LangSlice Nonlinear Registration"
GUI_DEPENDENCY = "ch.epfl.biop:pyimagej-scijava-command:0.2.1"


def install_gui(abba: Any) -> str:
    """Add LangSlice to ABBA's ``Register >`` menu.

    Builds a small dialog (moving channel) via ``PyCommandBuilder`` and
    registers it through ABBA's registration-plugin UI registry, which the
    BDV view reads when it is built — so call this (and
    :func:`enable_langslice_registration`) before ``abba.show_bdv_ui()``.
    """
    from jpype import JImplements, JOverride  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    try:
        PyCommandBuilder = jimport("org.scijava.command.PyCommandBuilder")
    except Exception as exc:
        raise RuntimeError(
            f"PyCommandBuilder is not on the JVM classpath. Append {GUI_DEPENDENCY!r} "
            "to scyjava.config.endpoints before constructing Abba (run_gui_session "
            "does this automatically)."
        ) from exc
    MultiSlicePositioner = jimport("ch.epfl.biop.atlas.aligner.MultiSlicePositioner")
    SourcesChannelsSelect = jimport(
        "ch.epfl.biop.sourceandconverter.processor.SourcesChannelsSelect"
    )
    HashMap = jimport("java.util.HashMap")
    Integer = jimport("java.lang.Integer")
    Function = jimport("java.util.function.Function")

    @JImplements(Function)
    class RunLangSliceRegistration:
        @JOverride
        def apply(self, inputs):
            mp = inputs.get("mp")
            channel = int(inputs.get("moving_channel"))
            mp.registerSelectedSlices(
                REGISTRATION_NAME,
                SourcesChannelsSelect(list(COORD_CHANNELS)),
                SourcesChannelsSelect(channel),
                HashMap(),
            )
            return HashMap()

    (
        PyCommandBuilder()
        .name(GUI_COMMAND_NAME)
        .input("mp", MultiSlicePositioner.class_)
        .input("moving_channel", Integer.class_)
        .menuPath(
            "Plugins>BIOP>Atlas>Multi Image To Atlas>Align>" + GUI_COMMAND_NAME
        )
        .function(RunLangSliceRegistration())
        .create(abba.ij.context())
    )
    MultiSlicePositioner.registerRegistrationPluginUI(REGISTRATION_NAME, GUI_COMMAND_NAME)
    logger.info("Installed ABBA GUI entry: Register > %s", GUI_COMMAND_NAME)
    return GUI_COMMAND_NAME


def run_gui_session(
    abba_atlas: str = "Adult Mouse Brain - Allen Brain Atlas V3p1", **overrides: Any
) -> None:
    """Launch an ABBA GUI session with LangSlice registration installed.

    Blocks until the ImageJ/ABBA session's JVM shuts down. Keyword overrides
    are forwarded to :func:`enable_langslice_registration`.
    """
    import time

    import jpype  # pyright: ignore[reportMissingImports]
    import scyjava.config  # pyright: ignore[reportMissingImports]
    from abba_python.abba import Abba  # pyright: ignore[reportMissingImports]

    # PyCommandBuilder (used by install_gui) ships in pyimagej-scijava-command,
    # which ABBA declares only test-scope — add it to the JVM ourselves.
    scyjava.config.endpoints.append(GUI_DEPENDENCY)

    abba = Abba(abba_atlas)
    enable_langslice_registration(abba, **overrides)
    install_gui(abba)
    abba.show_bdv_ui()
    while jpype.isJVMStarted():
        time.sleep(1)
