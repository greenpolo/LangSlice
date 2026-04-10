"""
Evaluation: Dense deformation field extraction using ITK-Elastix and SimpleITK Demons.

Use case: Two flat-colored brain region maps (~1200x864, RGB), same color scheme,
slightly different geometry. Need dense displacement field between them.

Install first:
    pip install itk-elastix   # ~150MB, includes itk core
    # OR just:
    pip install SimpleITK     # ~50MB, already sufficient for Demons

Run:
    python test_dense_registration.py
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Option A: ITK-Elastix  (B-spline deformable registration)
# ---------------------------------------------------------------------------

def register_elastix(fixed_path: str, moving_path: str) -> dict:
    """
    B-spline registration via itk-elastix.
    Returns dict with displacement_field (H,W,2 numpy), elapsed time, warped image.
    """
    import itk

    # Read as float32 scalar (grayscale).
    # Elastix works on scalar images; for same-color maps we convert to gray.
    fixed_rgb = itk.imread(fixed_path)
    moving_rgb = itk.imread(moving_path)

    # Convert RGB -> grayscale via luminance (or just average for flat-color maps)
    fixed_np = itk.array_from_image(fixed_rgb).astype(np.float32)
    moving_np = itk.array_from_image(moving_rgb).astype(np.float32)

    if fixed_np.ndim == 3:  # (H, W, C)
        fixed_gray = np.mean(fixed_np, axis=-1)
        moving_gray = np.mean(moving_np, axis=-1)
    else:
        fixed_gray = fixed_np
        moving_gray = moving_np

    # Create ITK images from numpy
    fixed_itk = itk.image_from_array(fixed_gray.astype(np.float32))
    moving_itk = itk.image_from_array(moving_gray.astype(np.float32))

    # Copy spacing/origin from original if available, else defaults (1.0)
    # For our case pixel spacing = 1.0 is fine

    # Build parameter maps: rigid (for initial alignment) + B-spline (deformable)
    parameter_object = itk.ParameterObject.New()

    # Rigid pre-alignment (fast, catches any global shift)
    rigid_map = parameter_object.GetDefaultParameterMap("rigid")
    rigid_map["MaximumNumberOfIterations"] = ["256"]
    rigid_map["Metric"] = ["AdvancedMeanSquares"]  # ideal for same-modality
    rigid_map["NumberOfResolutions"] = ["3"]
    parameter_object.AddParameterMap(rigid_map)

    # B-spline deformable
    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_map["Metric"] = ["AdvancedMeanSquares"]  # SSD-type, perfect for same-color
    bspline_map["MaximumNumberOfIterations"] = ["512"]
    bspline_map["NumberOfResolutions"] = ["4"]
    bspline_map["FinalGridSpacingInPhysicalUnits"] = ["16"]  # fine grid
    bspline_map["NumberOfSpatialSamples"] = ["4096"]
    parameter_object.AddParameterMap(bspline_map)

    t0 = time.perf_counter()

    # Run registration
    result_image, result_transform = itk.elastix_registration_method(
        fixed_itk, moving_itk,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    # Extract dense deformation field via transformix
    deformation_field = itk.transformix_deformation_field(
        moving_itk, result_transform
    )

    elapsed = time.perf_counter() - t0

    # Convert to numpy: shape (H, W, 2) — displacement vectors in pixels
    field_np = itk.array_from_image(deformation_field)
    result_np = itk.array_from_image(result_image)

    print(f"[Elastix B-spline] Time: {elapsed:.2f}s")
    print(f"  Displacement field shape: {field_np.shape}")
    print(f"  Max displacement: {np.max(np.abs(field_np)):.2f} px")

    return {
        "displacement_field": field_np,
        "warped_image": result_np,
        "elapsed": elapsed,
        "method": "elastix_bspline",
    }


# ---------------------------------------------------------------------------
# Option B: SimpleITK Demons (direct displacement field)
# ---------------------------------------------------------------------------

def register_demons(fixed_path: str, moving_path: str,
                    variant: str = "diffeomorphic") -> dict:
    """
    Demons registration via SimpleITK.

    variant: "diffeomorphic", "fast_symmetric", "symmetric", "classic"

    Returns dict with displacement_field (H,W,2 numpy), elapsed time, warped image.
    """
    import SimpleITK as sitk

    fixed_rgb = sitk.ReadImage(fixed_path)
    moving_rgb = sitk.ReadImage(moving_path)

    # Demons filters require scalar float images
    if fixed_rgb.GetNumberOfComponentsPerPixel() > 1:
        # Convert RGB to grayscale (vector magnitude is one approach;
        # simple average works well for flat-color maps)
        fixed_np = sitk.GetArrayFromImage(fixed_rgb).astype(np.float32)
        moving_np = sitk.GetArrayFromImage(moving_rgb).astype(np.float32)
        fixed_gray_np = np.mean(fixed_np, axis=-1)
        moving_gray_np = np.mean(moving_np, axis=-1)
        fixed = sitk.GetImageFromArray(fixed_gray_np)
        moving = sitk.GetImageFromArray(moving_gray_np)
        fixed.CopyInformation(sitk.GetImageFromArray(fixed_gray_np))
    else:
        fixed = sitk.Cast(fixed_rgb, sitk.sitkFloat32)
        moving = sitk.Cast(moving_rgb, sitk.sitkFloat32)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Optional: histogram matching (helps Demons converge, though our images
    # already share the same color scheme so effect is minimal)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # Choose Demons variant
    if variant == "diffeomorphic":
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        demons.SetStandardDeviations(1.5)
    elif variant == "fast_symmetric":
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        demons.SetStandardDeviations(1.5)
    elif variant == "symmetric":
        demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        demons.SetStandardDeviations(1.5)
    else:  # classic
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        demons.SetStandardDeviations(1.5)

    demons.SetSmoothDisplacementField(True)

    t0 = time.perf_counter()

    # Single-scale run (fast for small images)
    displacement_field = demons.Execute(fixed, moving)

    elapsed = time.perf_counter() - t0

    # Convert displacement field to numpy
    field_np = sitk.GetArrayFromImage(displacement_field)  # (H, W, 2)

    # Warp the moving image using the displacement field
    outTx = sitk.DisplacementFieldTransform(displacement_field)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    warped = resampler.Execute(moving)
    warped_np = sitk.GetArrayFromImage(warped)

    print(f"[Demons ({variant})] Time: {elapsed:.2f}s")
    print(f"  Iterations: {demons.GetElapsedIterations()}")
    print(f"  RMS change: {demons.GetRMSChange():.6f}")
    print(f"  Displacement field shape: {field_np.shape}")
    print(f"  Max displacement: {np.max(np.abs(field_np)):.2f} px")

    return {
        "displacement_field": field_np,
        "warped_image": warped_np,
        "elapsed": elapsed,
        "method": f"demons_{variant}",
        "iterations": demons.GetElapsedIterations(),
        "rms": demons.GetRMSChange(),
    }


def register_demons_multiscale(fixed_path: str, moving_path: str) -> dict:
    """
    Multi-scale Diffeomorphic Demons — better for larger deformations.
    Uses a coarse-to-fine pyramid.
    """
    import SimpleITK as sitk

    fixed_rgb = sitk.ReadImage(fixed_path)
    moving_rgb = sitk.ReadImage(moving_path)

    # Convert to scalar float
    if fixed_rgb.GetNumberOfComponentsPerPixel() > 1:
        fixed_np = sitk.GetArrayFromImage(fixed_rgb).astype(np.float32)
        moving_np = sitk.GetArrayFromImage(moving_rgb).astype(np.float32)
        fixed = sitk.GetImageFromArray(np.mean(fixed_np, axis=-1))
        moving = sitk.GetImageFromArray(np.mean(moving_np, axis=-1))
    else:
        fixed = sitk.Cast(fixed_rgb, sitk.sitkFloat32)
        moving = sitk.Cast(moving_rgb, sitk.sitkFloat32)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Histogram matching
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # Multi-scale implementation using shrink + smooth pyramid
    shrink_factors = [4, 2, 1]
    smooth_sigmas = [4.0, 2.0, 0.0]
    iterations_per_level = [200, 100, 50]

    t0 = time.perf_counter()

    displacement_field = None

    for level, (sf, sigma, n_iter) in enumerate(
        zip(shrink_factors, smooth_sigmas, iterations_per_level)
    ):
        # Downsample
        if sf > 1:
            f_img = sitk.Shrink(fixed, [sf, sf])
            m_img = sitk.Shrink(moving, [sf, sf])
        else:
            f_img = fixed
            m_img = moving

        if sigma > 0:
            f_img = sitk.SmoothingRecursiveGaussian(f_img, sigma)
            m_img = sitk.SmoothingRecursiveGaussian(m_img, sigma)

        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(n_iter)
        demons.SetStandardDeviations(1.5)
        demons.SetSmoothDisplacementField(True)

        if displacement_field is not None:
            # Resample displacement field to current resolution
            displacement_field = sitk.Resample(displacement_field, f_img)
            displacement_field = demons.Execute(f_img, m_img, displacement_field)
        else:
            displacement_field = demons.Execute(f_img, m_img)

        print(f"  Level {level} (shrink={sf}): {demons.GetElapsedIterations()} iters, "
              f"RMS={demons.GetRMSChange():.6f}")

    # Resample to full resolution if needed
    if displacement_field.GetSize() != fixed.GetSize():
        displacement_field = sitk.Resample(displacement_field, fixed)

    elapsed = time.perf_counter() - t0

    field_np = sitk.GetArrayFromImage(displacement_field)

    outTx = sitk.DisplacementFieldTransform(displacement_field)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    warped = resampler.Execute(moving)
    warped_np = sitk.GetArrayFromImage(warped)

    print(f"[Demons Multiscale Diffeomorphic] Time: {elapsed:.2f}s")
    print(f"  Displacement field shape: {field_np.shape}")
    print(f"  Max displacement: {np.max(np.abs(field_np)):.2f} px")

    return {
        "displacement_field": field_np,
        "warped_image": warped_np,
        "elapsed": elapsed,
        "method": "demons_multiscale_diffeomorphic",
    }


# ---------------------------------------------------------------------------
# Option C: SimpleITK B-spline via ImageRegistrationMethod
# ---------------------------------------------------------------------------

def register_sitk_bspline(fixed_path: str, moving_path: str) -> dict:
    """
    B-spline registration using SimpleITK's ImageRegistrationMethod.
    For comparison with Elastix and Demons.
    """
    import SimpleITK as sitk

    fixed_rgb = sitk.ReadImage(fixed_path)
    moving_rgb = sitk.ReadImage(moving_path)

    if fixed_rgb.GetNumberOfComponentsPerPixel() > 1:
        fixed_np = sitk.GetArrayFromImage(fixed_rgb).astype(np.float32)
        moving_np = sitk.GetArrayFromImage(moving_rgb).astype(np.float32)
        fixed = sitk.GetImageFromArray(np.mean(fixed_np, axis=-1))
        moving = sitk.GetImageFromArray(np.mean(moving_np, axis=-1))
    else:
        fixed = sitk.Cast(fixed_rgb, sitk.sitkFloat32)
        moving = sitk.Cast(moving_rgb, sitk.sitkFloat32)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    t0 = time.perf_counter()

    # Set up B-spline transform
    grid_physical_spacing = [50.0, 50.0]
    image_physical_size = [
        size * spacing
        for size, spacing in zip(fixed.GetSize(), fixed.GetSpacing())
    ]
    mesh_size = [
        int(image_size / grid_spacing + 0.5)
        for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    ]
    initial_transform = sitk.BSplineTransformInitializer(
        image1=fixed, transformDomainMeshSize=mesh_size, order=3
    )

    registration_method = sitk.ImageRegistrationMethod()

    # Mean squares metric — ideal for same-modality same-contrast images
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.25)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # LBFGSB optimizer works well with B-spline
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)

    elapsed = time.perf_counter() - t0

    # Convert to displacement field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(fixed)
    displacement_field = displacement_filter.Execute(final_transform)
    field_np = sitk.GetArrayFromImage(displacement_field)

    # Warp
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    warped = resampler.Execute(moving)
    warped_np = sitk.GetArrayFromImage(warped)

    print(f"[SimpleITK B-spline] Time: {elapsed:.2f}s")
    print(f"  Optimizer stop: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"  Final metric: {registration_method.GetMetricValue():.6f}")
    print(f"  Displacement field shape: {field_np.shape}")
    print(f"  Max displacement: {np.max(np.abs(field_np)):.2f} px")

    return {
        "displacement_field": field_np,
        "warped_image": warped_np,
        "elapsed": elapsed,
        "method": "sitk_bspline",
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def create_synthetic_test_pair(shape=(864, 1200)):
    """
    Create a pair of flat-colored brain region maps with known deformation.
    Useful for testing when real images aren't handy.
    """
    from PIL import Image

    h, w = shape
    # Simple flat-color regions (simulate brain atlas regions)
    fixed = np.zeros((h, w, 3), dtype=np.uint8)

    # Create some colored regions
    fixed[:h//3, :, :] = [200, 100, 100]          # top: red-ish
    fixed[h//3:2*h//3, :w//2, :] = [100, 200, 100]  # middle-left: green
    fixed[h//3:2*h//3, w//2:, :] = [100, 100, 200]  # middle-right: blue
    fixed[2*h//3:, :, :] = [200, 200, 100]        # bottom: yellow

    # Add an elliptical region in center
    yy, xx = np.mgrid[:h, :w]
    ellipse = ((yy - h/2)**2 / (h/4)**2 + (xx - w/2)**2 / (w/3)**2) < 1
    fixed[ellipse] = [150, 50, 150]

    # Create a warped version with a known deformation
    # Simple sinusoidal warp
    from scipy.ndimage import map_coordinates

    flow_y = 8 * np.sin(2 * np.pi * xx / w) * np.cos(np.pi * yy / h)
    flow_x = 6 * np.cos(2 * np.pi * yy / h) * np.sin(np.pi * xx / w)

    coords_y = yy + flow_y
    coords_x = xx + flow_x

    moving = np.zeros_like(fixed)
    for c in range(3):
        moving[:, :, c] = map_coordinates(
            fixed[:, :, c], [coords_y, coords_x],
            order=1, mode='nearest'
        ).astype(np.uint8)

    # Save
    Image.fromarray(fixed).save("test_runs/synth_fixed.png")
    Image.fromarray(moving).save("test_runs/synth_moving.png")

    print(f"Created synthetic test pair: {w}x{h}")
    print(f"  Known max deformation: ~{max(8, 6):.0f} px")

    return "test_runs/synth_fixed.png", "test_runs/synth_moving.png"


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        fixed_path = sys.argv[1]
        moving_path = sys.argv[2]
    else:
        print("No images provided, creating synthetic test pair...\n")
        fixed_path, moving_path = create_synthetic_test_pair()

    print(f"\nFixed:  {fixed_path}")
    print(f"Moving: {moving_path}\n")
    print("=" * 60)

    results = []

    # --- SimpleITK Demons variants ---
    try:
        import SimpleITK
        print("\n--- SimpleITK Demons (Diffeomorphic, single-scale) ---")
        r = register_demons(fixed_path, moving_path, variant="diffeomorphic")
        results.append(r)

        print("\n--- SimpleITK Demons (Fast Symmetric, single-scale) ---")
        r = register_demons(fixed_path, moving_path, variant="fast_symmetric")
        results.append(r)

        print("\n--- SimpleITK Demons (Diffeomorphic, multi-scale) ---")
        r = register_demons_multiscale(fixed_path, moving_path)
        results.append(r)

        print("\n--- SimpleITK B-spline ---")
        r = register_sitk_bspline(fixed_path, moving_path)
        results.append(r)

    except ImportError:
        print("SimpleITK not installed. Run: pip install SimpleITK")

    # --- ITK-Elastix ---
    try:
        import itk
        print("\n--- ITK-Elastix B-spline ---")
        r = register_elastix(fixed_path, moving_path)
        results.append(r)
    except ImportError:
        print("\nitk-elastix not installed. Run: pip install itk-elastix")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        field = r["displacement_field"]
        mag = np.sqrt(np.sum(field**2, axis=-1))
        print(f"  {r['method']:40s}  {r['elapsed']:6.2f}s  "
              f"max_disp={np.max(np.abs(field)):6.2f}  "
              f"mean_mag={np.mean(mag):6.2f}")
