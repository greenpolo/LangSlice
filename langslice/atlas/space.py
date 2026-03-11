from dataclasses import dataclass

from brainglobe_space import AnatomicalSpace


@dataclass(frozen=True)
class AtlasSpaceContext:
    atlas_name: str
    orientation: str
    shape: tuple[int, int, int]
    resolution_um: tuple[float, float, float]
    space: AnatomicalSpace
    ap_axis_index: int
    dv_axis_index: int
    ml_axis_index: int
    ap_resolution_mm: float


def _shape3d(volume: object) -> tuple[int, int, int]:
    shape = getattr(volume, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"Expected a 3D atlas volume, got shape={shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _resolution3d_um(resolution: object) -> tuple[float, float, float]:
    if not isinstance(resolution, (tuple, list)) or len(resolution) != 3:
        raise ValueError(f"Expected 3-axis atlas resolution, got {resolution!r}")
    return float(resolution[0]), float(resolution[1]), float(resolution[2])


def atlas_space_context(atlas: object) -> AtlasSpaceContext:
    atlas_name = str(getattr(atlas, "atlas_name", "unknown"))
    orientation = str(getattr(atlas, "orientation", ""))
    shape = _shape3d(getattr(atlas, "reference", None))
    resolution_um = _resolution3d_um(getattr(atlas, "resolution", None))
    space = AnatomicalSpace(
        origin=orientation,
        shape=shape,
        resolution=resolution_um,
    )
    ap_axis_index = space.get_axis_idx("sagittal")
    dv_axis_index = space.get_axis_idx("vertical")
    ml_axis_index = space.get_axis_idx("frontal")
    ap_axis_letter = space.origin[ap_axis_index]
    if ap_axis_letter != "a":
        raise ValueError(
            f"Atlas '{atlas_name}' orientation '{space.origin_string}' is not supported. "
            + "LangSlice currently requires AP axis to increase from anterior to posterior."
        )
    return AtlasSpaceContext(
        atlas_name=atlas_name,
        orientation=space.origin_string,
        shape=shape,
        resolution_um=resolution_um,
        space=space,
        ap_axis_index=ap_axis_index,
        dv_axis_index=dv_axis_index,
        ml_axis_index=ml_axis_index,
        ap_resolution_mm=resolution_um[ap_axis_index] / 1000.0,
    )


def require_coronal_layout(context: AtlasSpaceContext) -> AtlasSpaceContext:
    if context.ap_axis_index != 0 or context.dv_axis_index != 1 or context.ml_axis_index != 2:
        raise ValueError(
            f"Atlas '{context.atlas_name}' orientation '{context.orientation}' is not supported. "
            + "LangSlice currently requires coronal layout with AP/DV/ML mapped to axes 0/1/2."
        )
    return context
