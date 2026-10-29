"""The JSON form of a section's parts, both ways.

A section's affine, deformation, border trace, landmarks, made images,
masks, layers and cutting angles are written to JSON in one form, here:
the state file, the undo history, the registration listings and every
picture's record of the stack it was drawn from all hold it, and digests
of a placement hash it.

Each part is read back strictly (:mod:`langslice.core.stack.json_reads`):
a key of the wrong type is a ``ValueError`` naming it. A part's keys that
its shape does not declare are kept in its ``extra`` and written back
after the declared ones, so a file an earlier LangSlice or a host wrote
reads back the same. An optional field that is None is not written.
"""

from __future__ import annotations

from collections.abc import Sequence

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable
from langslice.core.stack.affine import (
    Calibration,
    FitRegions,
    PhysicalAffine,
    StoredAffine,
)
from langslice.core.stack.deformation import DeformationRef, LandmarkWarp
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.json_reads import (
    as_number,
    flag_at,
    integer_at,
    list_at,
    number_at,
    numbers_at,
    object_at,
    objects_at,
    optional_at,
    other_keys,
    point_at,
    set_items,
    text_at,
    text_map_at,
    texts_at,
)
from langslice.core.stack.section import Angles
from langslice.core.stack.trace import BorderTrace

#: A JSON object as this module writes it.
Written = dict[str, JsonValue]


# --- the affine --------------------------------------------------------------

#: The keys of an affine's knobs.
PHYSICAL_KEYS = (
    "rotation_deg", "scale_x", "scale_y", "translate_x_mm", "translate_y_mm",
    "shear", "pivot",
)  # fmt: skip
#: The keys of a stored affine, as :class:`StoredAffine` names them.
AFFINE_KEYS = (
    "kind", "params", "physical", "calibration", "mirrored", "note",
    "in_plane", "regions", "fit_atlas", "orientation", "stale",
)  # fmt: skip


def physical_to_json(physical: PhysicalAffine) -> Written:
    """Return an affine's knobs in their JSON form.

    Parameters
    ----------
    physical
        The knobs: rotation, scales, shifts, shear and pivot.

    Returns
    -------
    dict
        ``{"rotation_deg", "scale_x", "scale_y", "translate_x_mm",
        "translate_y_mm", "shear", "pivot"}``, each knob that is set.
    """
    form = set_items(
        [
            ("rotation_deg", physical.rotation_deg),
            ("scale_x", physical.scale_x),
            ("scale_y", physical.scale_y),
            ("translate_x_mm", physical.translate_x_mm),
            ("translate_y_mm", physical.translate_y_mm),
            ("shear", physical.shear),
            ("pivot", physical.pivot),
        ]
    )
    form.update(to_jsonable(physical.extra))
    return form


def physical_from_json(data: JsonObject) -> PhysicalAffine:
    """Return the knobs a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`physical_to_json` writes.

    Returns
    -------
    PhysicalAffine
        The knobs; one the object does not give is None.
    """
    return PhysicalAffine(
        rotation_deg=optional_at(data, "rotation_deg", number_at),
        scale_x=optional_at(data, "scale_x", number_at),
        scale_y=optional_at(data, "scale_y", number_at),
        translate_x_mm=optional_at(data, "translate_x_mm", number_at),
        translate_y_mm=optional_at(data, "translate_y_mm", number_at),
        shear=optional_at(data, "shear", number_at),
        pivot=optional_at(data, "pivot", point_at),
        extra=other_keys(data, PHYSICAL_KEYS),
    )


def calibration_to_json(calibration: Calibration) -> Written:
    """Return an affine's calibration in its JSON form.

    Parameters
    ----------
    calibration
        The section scale the affine's millimetres are measured at.

    Returns
    -------
    dict
        ``{"section_um_per_px", "source"}``.
    """
    form: Written = {
        "section_um_per_px": calibration.section_um_per_px,
        "source": calibration.source,
    }
    return form


def calibration_from_json(data: JsonObject) -> Calibration:
    """Return the calibration a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`calibration_to_json` writes.

    Returns
    -------
    Calibration
        The calibration.
    """
    return Calibration(
        section_um_per_px=_optional_number(data, "section_um_per_px"),
        source=text_at(data, "source"),
    )


def _optional_number(data: JsonObject, key: str) -> float | None:
    if data.get(key) is None:
        return None
    return number_at(data, key)


def _calibration(data: JsonObject, key: str) -> Calibration:
    return calibration_from_json(object_at(data, key))


def _regions(data: JsonObject, key: str) -> FitRegions:
    choice = object_at(data, key)
    return FitRegions(
        include=texts_at(choice, "include"),
        exclude=texts_at(choice, "exclude"),
    )


def _physical(data: JsonObject, key: str) -> PhysicalAffine:
    return physical_from_json(object_at(data, key))


def _regions_json(regions: FitRegions | None) -> Written | None:
    if regions is None:
        return None
    form: Written = {
        "include": list[JsonValue](regions.include),
        "exclude": list[JsonValue](regions.exclude),
    }
    return form


def affine_to_json(affine: StoredAffine) -> Written:
    """Return a stored affine in its JSON form.

    An imported affine writes its reading on the render (``in_plane``)
    and ``mirrored`` before its calibration; every other one writes the
    calibration first.

    Parameters
    ----------
    affine
        The section's in-plane affine.

    Returns
    -------
    dict
        Its declared keys that are set, in order, then its other keys.
    """
    physical = None
    if affine.physical is not None:
        physical = physical_to_json(affine.physical)
    calibration = None
    if affine.calibration is not None:
        calibration = calibration_to_json(affine.calibration)
    middle: list[tuple[str, Json]] = [
        ("calibration", calibration),
        ("mirrored", affine.mirrored),
    ]
    if affine.in_plane is not None:
        middle = [
            ("in_plane", affine.in_plane),
            ("mirrored", affine.mirrored),
            ("calibration", calibration),
        ]
    form = set_items(
        [
            ("kind", affine.kind),
            ("params", affine.params),
            ("physical", physical),
            *middle,
            ("note", affine.note),
            ("regions", _regions_json(affine.regions)),
            ("fit_atlas", affine.fit_atlas),
            ("orientation", affine.orientation),
            ("stale", affine.stale),
        ]
    )
    form.update(to_jsonable(affine.extra))
    return form


def affine_from_json(data: JsonObject) -> StoredAffine:
    """Return the stored affine a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`affine_to_json` writes, or a host's transform
        dictionary.

    Returns
    -------
    StoredAffine
        The affine.
    """
    return StoredAffine(
        kind=optional_at(data, "kind", text_at),
        params=numbers_at(data, "params"),
        physical=optional_at(data, "physical", _physical),
        calibration=optional_at(data, "calibration", _calibration),
        mirrored=optional_at(data, "mirrored", flag_at),
        note=optional_at(data, "note", text_at),
        in_plane=optional_at(data, "in_plane", object_at),
        regions=optional_at(data, "regions", _regions),
        fit_atlas=optional_at(data, "fit_atlas", text_at),
        orientation=optional_at(data, "orientation", object_at),
        stale=optional_at(data, "stale", flag_at),
        extra=other_keys(data, AFFINE_KEYS),
    )


# --- the deformation ---------------------------------------------------------

#: The keys of a deformation reference.
DEFORMATION_KEYS = (
    "keep_linear", "record", "key", "linear_key", "steps", "summary",
    "inverse_source", "landmark_revision", "landmark_pairs",
    "landmark_folds", "landmark_holds", "landmark_baseline_key",
    "landmark_baseline_steps",
)  # fmt: skip


def deformation_to_json(deformation: DeformationRef) -> Written:
    """Return a deformation reference in its JSON form.

    A landmark warp writes its key first and its steps last, around its
    landmark facts; every other fit writes its record folder first, and a
    kept-linear marker its note first.

    Parameters
    ----------
    deformation
        The section's reference to its applied deformation.

    Returns
    -------
    dict
        Its declared keys that are set, in order, then its other keys.
    """
    warp = deformation.landmark
    if warp is None:
        form = set_items(
            [
                ("keep_linear", deformation.kept_linear),
                ("record", deformation.record),
                ("key", deformation.key),
                ("linear_key", deformation.linear_key),
                ("steps", deformation.steps),
                ("summary", deformation.summary),
                ("inverse_source", deformation.inverse_source),
            ]
        )
    else:
        form = set_items(
            [
                ("key", deformation.key),
                ("record", deformation.record),
                ("linear_key", deformation.linear_key),
                ("summary", deformation.summary),
                ("inverse_source", deformation.inverse_source),
                ("landmark_revision", warp.revision),
                ("landmark_pairs", warp.pairs),
                ("landmark_folds", warp.folds),
                ("landmark_holds", warp.holds),
            ]
        )
        # a warp built on the linear placement writes its baseline as null
        form["landmark_baseline_key"] = warp.baseline_key
        form["landmark_baseline_steps"] = to_jsonable(warp.baseline_steps)
        form["steps"] = to_jsonable(deformation.steps)
    form.update(to_jsonable(deformation.extra))
    return form


def _baseline_key(data: JsonObject) -> str | None:
    if data.get("landmark_baseline_key") is None:
        return None
    return text_at(data, "landmark_baseline_key")


def _landmark_warp(data: JsonObject) -> LandmarkWarp | None:
    if "landmark_revision" not in data:
        return None
    return LandmarkWarp(
        revision=text_at(data, "landmark_revision"),
        pairs=text_map_at(data, "landmark_pairs"),
        folds=object_at(data, "landmark_folds"),
        baseline_key=_baseline_key(data),
        baseline_steps=objects_at(data, "landmark_baseline_steps"),
        holds=optional_at(data, "landmark_holds", list_at),
    )


def deformation_from_json(data: JsonObject) -> DeformationRef:
    """Return the deformation reference a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`deformation_to_json` writes.

    Returns
    -------
    DeformationRef
        The reference.
    """
    kept_linear = optional_at(data, "keep_linear", text_at)
    key = optional_at(data, "key", text_at)
    steps = optional_at(data, "steps", objects_at)
    if kept_linear is None:
        # every fit's reference names its saved fit and the steps that made it
        key = text_at(data, "key")
        steps = objects_at(data, "steps")
    return DeformationRef(
        key=key,
        linear_key=text_at(data, "linear_key"),
        steps=steps,
        record=optional_at(data, "record", text_at),
        summary=optional_at(data, "summary", object_at),
        inverse_source=optional_at(data, "inverse_source", text_at),
        landmark=_landmark_warp(data),
        kept_linear=kept_linear,
        extra=other_keys(data, DEFORMATION_KEYS),
    )


# --- the border trace --------------------------------------------------------

#: The keys of a border trace, in the order its record is written.
TRACE_KEYS = (
    "id", "geometry_fingerprint", "prompt_edited", "provider", "model",
    "profile", "untested", "include", "exclude", "output_kind",
    "fit_performed", "artifact_dir", "artifact_paths", "cached", "attempt",
    "raw_received", "status", "route", "raw_size", "model_border_pixels",
    "error", "message",
)  # fmt: skip


def trace_to_json(trace: BorderTrace) -> Written:
    """Return a border trace in its JSON form.

    Parameters
    ----------
    trace
        The section's image-model border trace.

    Returns
    -------
    dict
        Its set keys in the order the trace's record is written, then its
        other keys.
    """
    form = set_items(
        [
            ("id", trace.id),
            ("geometry_fingerprint", trace.geometry_fingerprint),
            ("prompt_edited", trace.prompt_edited),
            ("provider", trace.provider),
            ("model", trace.model),
            ("profile", trace.profile),
            ("untested", trace.untested),
            ("include", trace.include),
            ("exclude", trace.exclude),
            ("output_kind", trace.output_kind),
            ("fit_performed", trace.fit_performed),
            ("artifact_dir", trace.artifact_dir),
            ("artifact_paths", trace.artifact_paths),
            ("cached", trace.cached),
            ("attempt", trace.attempt),
            ("raw_received", trace.raw_received),
            ("status", trace.status),
            ("route", trace.route),
            ("raw_size", trace.raw_size),
            ("model_border_pixels", trace.model_border_pixels),
            ("error", trace.error),
            ("message", trace.message),
        ]
    )
    form.update(to_jsonable(trace.extra))
    return form


def _size(data: JsonObject, key: str) -> tuple[int, int]:
    values = list_at(data, key)
    if len(values) != 2:
        raise ValueError(f"{key} must be two whole numbers, not {values!r}")
    width = integer_at({key: values[0]}, key)
    height = integer_at({key: values[1]}, key)
    return width, height


def trace_from_json(data: JsonObject) -> BorderTrace:
    """Return the border trace a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`trace_to_json` writes: the trace call's result
        record.

    Returns
    -------
    BorderTrace
        The trace.
    """
    return BorderTrace(
        id=text_at(data, "id"),
        geometry_fingerprint=text_at(data, "geometry_fingerprint"),
        status=text_at(data, "status"),
        prompt_edited=optional_at(data, "prompt_edited", flag_at),
        provider=optional_at(data, "provider", text_at),
        model=optional_at(data, "model", text_at),
        profile=optional_at(data, "profile", text_at),
        untested=optional_at(data, "untested", flag_at),
        include=optional_at(data, "include", texts_at),
        exclude=optional_at(data, "exclude", texts_at),
        output_kind=optional_at(data, "output_kind", text_at),
        fit_performed=optional_at(data, "fit_performed", flag_at),
        artifact_dir=optional_at(data, "artifact_dir", text_at),
        artifact_paths=optional_at(data, "artifact_paths", text_map_at),
        cached=optional_at(data, "cached", flag_at),
        attempt=optional_at(data, "attempt", integer_at),
        raw_received=optional_at(data, "raw_received", flag_at),
        route=optional_at(data, "route", text_at),
        raw_size=optional_at(data, "raw_size", _size),
        model_border_pixels=optional_at(
            data, "model_border_pixels", integer_at
        ),
        error=optional_at(data, "error", text_at),
        message=optional_at(data, "message", text_at),
        extra=other_keys(data, TRACE_KEYS),
    )


# --- made images, masks and layers -------------------------------------------

#: The keys of a made image's entry.
IMAGE_KEYS = ("source", "file", "digest", "made")
#: The keys of a mask's entry.
MASK_KEYS = (
    "kind", "section_image", "atlas_image", "params", "inputs",
    "registration", "file", "left_out",
)  # fmt: skip
#: The keys of a layer's reference.
LAYER_KEYS = ("id", "name", "kind", "frame", "file", "tool", "call", "created")


def image_to_json(image: ImageRef) -> Written:
    """Return a made image's entry in its JSON form.

    Parameters
    ----------
    image
        The reference to the image file.

    Returns
    -------
    dict
        ``{"source", "file", "digest", "made"}``, then its other keys.
    """
    form = set_items(
        [
            ("source", image.source),
            ("file", image.file),
            ("digest", image.digest),
            ("made", image.made),
        ]
    )
    form.update(to_jsonable(image.extra))
    return form


def image_from_json(data: JsonObject) -> ImageRef:
    """Return the made image's entry a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`image_to_json` writes.

    Returns
    -------
    ImageRef
        The entry.
    """
    return ImageRef(
        source=object_at(data, "source"),
        file=text_at(data, "file"),
        digest=text_at(data, "digest"),
        made=text_at(data, "made"),
        extra=other_keys(data, IMAGE_KEYS),
    )


def mask_to_json(mask: MaskRef) -> Written:
    """Return a mask's entry in its JSON form.

    Parameters
    ----------
    mask
        The reference to the mask file.

    Returns
    -------
    dict
        Its keys in order, then its other keys.
    """
    form = set_items(
        [
            ("kind", mask.kind),
            ("section_image", mask.section_image),
            ("atlas_image", mask.atlas_image),
            ("params", mask.params),
            ("inputs", mask.inputs),
            ("registration", mask.registration),
            ("file", mask.file),
            ("left_out", mask.left_out),
        ]
    )
    form.update(to_jsonable(mask.extra))
    return form


def mask_from_json(data: JsonObject) -> MaskRef:
    """Return the mask's entry a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`mask_to_json` writes.

    Returns
    -------
    MaskRef
        The entry.
    """
    return MaskRef(
        kind=text_at(data, "kind"),
        section_image=text_at(data, "section_image"),
        atlas_image=text_at(data, "atlas_image"),
        params=object_at(data, "params"),
        inputs=text_at(data, "inputs"),
        file=text_at(data, "file"),
        left_out=number_at(data, "left_out"),
        registration=optional_at(data, "registration", text_at),
        extra=other_keys(data, MASK_KEYS),
    )


def layer_to_json(layer: LayerRef) -> Written:
    """Return a layer's reference in its JSON form.

    Parameters
    ----------
    layer
        The reference to the layer file.

    Returns
    -------
    dict
        Its keys in order, then its other keys.
    """
    form = set_items(
        [
            ("id", layer.id),
            ("name", layer.name),
            ("kind", layer.kind),
            ("frame", layer.frame),
            ("file", layer.file),
            ("tool", layer.tool),
            ("call", layer.call),
            ("created", layer.created),
        ]
    )
    form.update(to_jsonable(layer.extra))
    return form


def layer_from_json(data: JsonObject) -> LayerRef:
    """Return the layer's reference a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`layer_to_json` writes.

    Returns
    -------
    LayerRef
        The reference.
    """
    return LayerRef(
        id=text_at(data, "id"),
        name=text_at(data, "name"),
        kind=text_at(data, "kind"),
        frame=text_at(data, "frame"),
        file=text_at(data, "file"),
        tool=text_at(data, "tool"),
        call=text_at(data, "call"),
        created=text_at(data, "created"),
        extra=other_keys(data, LAYER_KEYS),
    )


# --- cutting angles ----------------------------------------------------------


def angles_to_json(angles: Angles) -> Written:
    """Return cutting angles in their JSON form.

    Parameters
    ----------
    angles
        ``(pitch_deg, yaw_deg)``.

    Returns
    -------
    dict
        ``{"pitch", "yaw"}`` in degrees.
    """
    form: Written = {"pitch": angles[0], "yaw": angles[1]}
    return form


def angles_from_json(data: JsonObject) -> Angles:
    """Return the cutting angles a JSON object holds.

    Parameters
    ----------
    data
        ``{"pitch", "yaw"}`` in degrees; a missing part is 0.

    Returns
    -------
    Angles
        ``(pitch_deg, yaw_deg)``, as stored.
    """
    pitch = as_number("pitch", data.get("pitch", 0.0))
    yaw = as_number("yaw", data.get("yaw", 0.0))
    return pitch, yaw


def affine_fields(affine: StoredAffine | None, keys: Sequence[str]) -> Written:
    """Return some keys of a stored affine's JSON form, null where unset.

    What a digest of a placement hashes (its six numbers, its calibration).

    Parameters
    ----------
    affine
        The section's in-plane affine; None for a section without one.
    keys
        Keys of :data:`AFFINE_KEYS`.

    Returns
    -------
    dict
        Each key with its value in :func:`affine_to_json`'s form, or null.
    """
    form: Written = {} if affine is None else affine_to_json(affine)
    return {key: form.get(key) for key in keys}
