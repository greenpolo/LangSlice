"""Native BigWarp registrations from calibrated pairs or sampled Elastix maps.

The stored spline is the complete mapping, including its affine component.
ABBA resamples using a fixed-to-moving TPS, exactly as BigWarp does.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _sample_elastix_landmarks(
    spline: dict[str, Any], *, tolerance_mm: float = 0.005,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the exact pullback as ABBA's native sampled TPS.

    Check independent off-grid locations and sampled Jacobians before returning
    anything the mirror can apply. The serialized Elastix maps stay authoritative;
    these correspondences are an export representation, never agent landmarks.
    """
    from langslice.landmark_warp import _ThinPlateKernel, fit_spline

    bounds = np.asarray(spline.get("domain_mm"), dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("Elastix export requires its saved physical domain_mm")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Export tolerance must be positive and finite")
    exact = fit_spline(spline)

    def grid(n: int, offset: float = 0.) -> np.ndarray:
        fractions = np.linspace(0, 1, n) if offset == 0 else (np.arange(n-1)+offset)/(n-1)
        coordinates = bounds[0] + fractions[:, None] * (bounds[1]-bounds[0])
        x, y = np.meshgrid(coordinates[:, 0], coordinates[:, 1])
        return np.column_stack((x.ravel(), y.ravel()))

    screening = grid(65)
    epsilon = float(np.min(bounds[1]-bounds[0])) * 1e-5
    derivatives = []
    for axis in np.eye(2) * epsilon:
        derivatives.append((exact.inverse(screening+axis)-exact.inverse(screening-axis))/(2*epsilon))
    exact_determinant = np.linalg.det(np.stack(derivatives, axis=2))
    if not np.isfinite(exact_determinant).all() or np.any(exact_determinant <= 1e-6):
        raise ValueError("Elastix export pullback folds or collapses on the sampled domain")

    last_error = float("inf")
    for n in (9, 17, 25, 33):
        target = grid(n)
        source = exact.inverse(target)
        approximation = _ThinPlateKernel(target, source, max_points=1089)
        # Two differently offset grids detect interpolation errors between knots;
        # an additional dense grid includes the domain boundary.
        probes = np.vstack((grid(2*n-1), grid(n, .37), grid(n, .71)))
        error = np.linalg.norm(approximation.forward(probes)-exact.inverse(probes), axis=1)
        last_error = float(np.max(error))
        determinant = np.linalg.det(approximation.jacobian(probes))
        if not np.isfinite(determinant).all() or np.any(determinant <= 1e-6):
            continue
        if last_error <= tolerance_mm:
            report = {"grid_size": n, "points": len(target),
                      "max_error_mm": last_error, "tolerance_mm": tolerance_mm,
                      "min_sampled_jacobian": float(determinant.min()),
                      "validation_points": len(probes)}
            if diagnostics is not None:
                diagnostics.update(report)
            logger.info("ABBA Elastix export approximation: %s", report)
            return source, target
    raise ValueError(
        "Elastix export could not meet sampled TPS accuracy/fold checks "
        f"within 1089 points (last maximum error {last_error:.6g} mm)"
    )


def spline_world_landmarks(
    spline: dict[str, Any], *, size: tuple[int, int], pixel_size_um: float,
    rotation_deg: int = 0, diagnostics: dict[str, Any] | None = None,
    tolerance_mm: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized oriented-section landmarks into centred ABBA mm."""
    dimensions = np.asarray(size, dtype=float)
    if dimensions.shape != (2,) or not np.all(np.isfinite(dimensions) & (dimensions > 0)):
        raise ValueError("Snapshot dimensions must be positive and finite")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0 or rotation_deg % 90:
        raise ValueError("Spline requires calibrated pixels and quarter-turn orientation")
    if rotation_deg % 180:
        dimensions = dimensions[::-1]
    extent = dimensions * pixel_size_um / 1000.0
    recorded = np.asarray(spline.get("extent_mm"), dtype=float)
    if recorded.shape != (2,) or not np.allclose(recorded, extent, rtol=1e-5, atol=1e-7):
        raise ValueError("Spline physical extent does not match the calibrated ABBA snapshot")
    if spline.get("backend") == "elastix":
        source_mm, target_mm = _sample_elastix_landmarks(
            spline, diagnostics=diagnostics, tolerance_mm=tolerance_mm,
        )
        return source_mm - extent / 2, target_mm - extent / 2
    if spline.get("backend", "tps") != "tps":
        raise ValueError("Unknown landmark warp backend; cannot export as a legacy TPS")
    source = np.asarray(spline.get("source"), dtype=float)
    target = np.asarray(spline.get("target"), dtype=float)
    if (source.ndim != 2 or source.shape[1] != 2 or len(source) < 3
            or target.shape != source.shape or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))):
        raise ValueError("Spline needs at least three finite paired XY landmarks")
    for points in (source, target):
        if (len(np.unique(points, axis=0)) != len(points)
                or np.linalg.matrix_rank(np.column_stack([points, np.ones(len(points))])) < 3):
            raise ValueError("Spline landmarks must be distinct and non-collinear")
    return (source - .5) * extent, (target - .5) * extent


def prepare_spline_registration(abba: Any, source_mm: np.ndarray, target_mm: np.ndarray) -> Any:
    """Prepare a serializable, completed native BigWarp step without opening UI.

    Building and round-tripping it happens before removing any previous step.
    ``setTransform`` marks the native plugin complete; RegisterSliceAction then
    appends it directly instead of calling its interactive ``register`` method.
    """
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    from langslice.integrations.abba import _build_java_tps

    BigWarp = jimport(
        "ch.epfl.biop.registration.sourceandconverter.bigwarp.SacBigWarp2DRegistration"
    )
    ctx = abba.ij.context()
    service = ctx.getService(jimport("org.scijava.plugin.PluginService").class_)
    registration = service.getPlugin(BigWarp.class_).createInstance()
    registration.setScijavaContext(ctx)
    # Legacy TPS pairs reproduce the Python pullback exactly. Elastix pairs
    # approximate its exact pullback within the checked export tolerance.
    registration.setRealTransform(_build_java_tps(target_mm, source_mm))
    registration.setTransform(registration.getTransform())
    registration.setRegistrationParameters(jimport("java.util.HashMap")())
    registration.setRegistrationName("LangSlice landmarks")
    if not registration.isRegistrationDone():
        raise RuntimeError("ABBA could not prepare the landmark spline registration")
    return registration
