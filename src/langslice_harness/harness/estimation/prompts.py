"""Plane-aware system-instruction builders for the position-estimation agents."""

from __future__ import annotations

from langslice_harness.atlas.space import Plane

_PLANE_BOILERPLATE: dict[str, str] = {
    "coronal":
        "Coordinate system: 0.0 mm is the anterior edge (olfactory bulb); "
        "larger mm moves posterior toward the cerebellum.",
    "sagittal":
        "Coordinate system: 0.0 mm is at the lateral edge of one hemisphere; "
        "larger mm moves toward the midline. The two hemispheres are mirror "
        "images, so estimate the canonical (single-hemisphere) ML position; "
        "your estimate should fall in the canonical hemisphere range.",
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
    """Single-slice system instruction.

    Closely mirrors the legacy ``ap_single_slice.py`` / ``ap_tool_use.py`` prompt
    that drove the 0.14 mm M01 baseline: explicit first-response contract
    (landmarks + broad sweep simultaneously), explicit VERIFY step after the
    broad sweep, and a hard "restart rather than narrow in the wrong place"
    rule. Dropping any of these moves MAE by 0.5-1 mm on Flash.
    """
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given a histology brain "
        f"slice image and must determine its {axis} position within a "
        f"reference atlas. {boilerplate}\n\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm.\n\n"
        f"You have tools to fetch atlas reference images and submit your final estimate.\n\n"
        f"STUDENT-SIZED STRATEGY:\n"
        f"1. IN YOUR FIRST RESPONSE, do two things simultaneously:\n"
        f"   a) Describe 2-3 prominent anatomical landmarks in the target "
        f"slice (e.g., 'anterior commissure visible', 'hippocampus forming', "
        f"'corpus callosum is thick', 'ventricles are large').\n"
        f"   b) Based on those landmarks, call `fetch_atlas` with broadly "
        f"spaced positions (e.g., [2, 4, 6, 8, 10]) to find the general "
        f"{axis} neighborhood.\n"
        f"2. COMPARE: after each fetch, briefly say which atlas section best "
        f"matches the target and which landmark evidence drove that choice.\n"
        f"3. RECOVER: if the broad sweep is clearly wrong, call `fetch_atlas` "
        f"with a different broad {axis} range instead of narrowing in the "
        f"wrong neighborhood.\n"
        f"4. NARROW: once one area is close, call `fetch_atlas` with tighter "
        f"positions around your best candidate.\n"
        f"5. OPTIONAL: if the narrow result is still ambiguous, make one finer "
        f"`fetch_atlas` call; otherwise call `submit_estimate` with concise "
        f"reasoning.\n\n"
        f"If atlas images don't match the target, DO NOT keep narrowing in "
        f"the same area — try a completely different {axis} range."
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
    """Multi-slice group system instruction.

    Restores the legacy ``ap_multi_slice.py`` prompt structure that drove the
    pre-ADK 0.19 mm M01 baseline: explicit first-response contract (landmarks
    on Slice 1 AND Slice N + broad sweep simultaneously), VERIFY step after
    the broad sweep, and an "interval is approximate" caveat so the model
    does not over-constrain on the nominal spacing.
    """
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given {n_slices} "
        f"consecutive histology brain slice images from the same brain, "
        f"ordered along the {axis} axis (Slice 1 = lowest {axis}, "
        f"Slice {n_slices} = highest).\n\n"
        f"Section parameters:\n"
        f"- Slice thickness: {thickness_um} um\n"
        f"- Section interval: {interval_mm:.3f} mm (center-to-center)\n\n"
        f"{boilerplate}\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm.\n\n"
        f"Your task: determine the {axis} position of EACH slice.\n\n"
        f"STUDENT-SIZED STRATEGY:\n"
        f"1. IN YOUR FIRST RESPONSE, do two things simultaneously:\n"
        f"   a) Describe 2-3 prominent anatomical landmarks in Slice 1 AND "
        f"Slice {n_slices} (e.g., 'anterior commissure visible', "
        f"'hippocampus forming', 'corpus callosum is thick', "
        f"'ventricles are large').\n"
        f"   b) Based on those landmarks, call `fetch_atlas` with broadly "
        f"spaced positions (e.g., [2, 4, 6, 8, 10]) to find the general "
        f"{axis} area spanned by the group.\n"
        f"2. COMPARE: after each fetch, briefly say which atlas sections best "
        f"match the first and last target slices, and which landmarks drove "
        f"that choice.\n"
        f"3. RECOVER: if the broad sweep is clearly wrong, call `fetch_atlas` "
        f"with a different broad {axis} range instead of narrowing in the "
        f"wrong neighborhood.\n"
        f"4. NARROW: once one area is close, fetch tighter positions around "
        f"the best matched slice or group center. Use the known "
        f"{interval_mm:.3f} mm interval as a guide — once you confidently "
        f"match ANY slice you can derive approximate positions for the "
        f"others.\n"
        f"5. OPTIONAL: if the narrow result is still ambiguous, make one finer "
        f"`fetch_atlas` call; otherwise submit all {n_slices} positions via "
        f"`submit_group_estimate`.\n\n"
        f"IMPORTANT: The interval is approximate — actual spacing may vary "
        f"slightly due to tissue preparation. Use it as a guide, not an "
        f"absolute constraint.\n\n"
        f"If atlas images don't match your slices, DO NOT keep narrowing in "
        f"the same area — try a completely different {axis} range."
    )
