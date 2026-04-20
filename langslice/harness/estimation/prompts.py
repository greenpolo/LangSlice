"""Plane-aware system-instruction builders for the position-estimation agents."""

from __future__ import annotations

from langslice.atlas.space import Plane

_PLANE_BOILERPLATE: dict[str, str] = {
    "coronal":
        "Coordinate system: 0.0 mm is the anterior edge (olfactory bulb); "
        "larger mm moves posterior toward the cerebellum.",
    "sagittal":
        "Coordinate system: 0.0 mm is the left hemisphere lateral edge; "
        "larger mm moves to the right.",
    "horizontal":
        "Coordinate system: 0.0 mm is dorsal (top of brain); "
        "larger mm moves ventral.",
}

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


def build_single_slice_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
) -> str:
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given a histology brain "
        f"slice image and must determine its {axis} position within a "
        f"reference atlas. {boilerplate}\n\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm.\n\n"
        f"You have tools to fetch atlas reference images, zoom into regions "
        f"of interest, view side-by-side comparisons, and submit your final "
        f"estimate.\n\n"
        f"RECOMMENDED STRATEGY:\n"
        f"1. Call `fetch_atlas` with broadly spaced positions "
        f"(e.g., [2, 4, 6, 8, 10]) to find the general region.\n"
        f"2. Call `fetch_atlas` with tighter positions around your best match.\n"
        f"3. Call `fetch_atlas` with very fine positions (~0.1-0.2mm apart) to pinpoint.\n"
        f"4. When a specific landmark is unclear, call `zoom` with a bounding box "
        f"[y1, x1, y2, x2] (0-1000) on 'target' or 'atlas:<mm>'.\n"
        f"5. To directly compare two sections, call `side_by_side` with two sources.\n"
        f"6. Verify neighbors, then call `submit_estimate`.\n\n"
        f"If atlas images don't look similar to the target, DO NOT keep narrowing "
        f"in the same area. Go back and try a different region.\n\n"
        f"Think carefully before each tool call, but always follow up with an action."
    )


def build_group_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
) -> str:
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given {n_slices} consecutive "
        f"histology brain slice images from the same brain, ordered along the "
        f"{axis} axis (Slice 1 = lowest {axis}, Slice {n_slices} = highest).\n\n"
        f"Section parameters:\n"
        f"- Slice thickness: {thickness_um} um\n"
        f"- Section interval: {interval_mm:.3f} mm (center-to-center)\n\n"
        f"{boilerplate}\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm.\n\n"
        f"Your task: determine the {axis} position of EACH slice.\n\n"
        f"STRATEGY:\n"
        f"1. Examine all slices; describe 2-3 prominent landmarks visible in "
        f"Slice 1 and Slice {n_slices}.\n"
        f"2. Call `fetch_atlas` with broadly spaced positions to find the "
        f"general area.\n"
        f"3. Narrow down by comparing atlas slices with your input slices.\n"
        f"4. Use the known {interval_mm:.3f} mm interval as a constraint - once "
        f"you confidently match ANY slice, derive approximate positions for "
        f"the others.\n"
        f"5. Use `zoom` to examine specific features and `side_by_side` to "
        f"compare any two sources directly.\n"
        f"6. Submit all {n_slices} positions via `submit_group_estimate`.\n\n"
        f"If atlas images don't match your slices, try a different region - "
        f"restart rather than commit to the wrong neighborhood."
    )
