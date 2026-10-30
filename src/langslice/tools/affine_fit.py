"""The Elastix affine fit: the ``elastix_affine`` tool and ``elastix_fit``.

:func:`elastix_affine` refines each section's in-plane transform against an
atlas image and writes every fit that succeeded as one undo step
(the transforms' one write, :func:`fit_transform` records). Each
row carries what the fit read, how far it turned the section and the flags of
an implausible change (:func:`plausibility_flags`); a fit restricted to
regions carries their box, which its picture zooms to.
``elastix_fit`` is one fit written nowhere, for scripts.

``elastix_affine_tool`` answers the tool's call with a row per section,
worded from the fit's numbers (the turn, and the flags in
``affine_flag_text``'s words), and shows each fitted section under its new
registration with the atlas borders on it, drawn on the section image the
fit read, unless the call's ``view`` is false and the host does not force
it. With ``restrict_to`` those regions' borders are drawn thick and the
others faint, and the picture is zoomed to the regions' box when that
magnifies it at least ``fit_regions.MIN_ZOOM_GAIN`` times.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from langslice.core.atlas.images import resolve_image
from langslice.core.fits import affine as core_affine
from langslice.core.fits.affine import FittedAffine
from langslice.core.fits.engine_log import engine_output, failure_message
from langslice.core.fits.flags import FitFlag
from langslice.core.fits.inputs import InputsBlock
from langslice.core.geometry.affine import IDENTITY_PARAMS
from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.damage import exclusions
from langslice.core.stack import codec
from langslice.core.stack.affine import FitRegions, StoredAffine
from langslice.core.stack.codec import Written
from langslice.job.settings.needs import marked, render
from langslice.tools import masks as ops_masks
from langslice.tools.deformable import (
    check_atlas_image,
    inputs_block,
    nothing_fitted,
    refuse_unknown_inputs,
)
from langslice.tools.fit_regions import (
    region_box,
    region_entries,
    zoom_window,
)
from langslice.tools.images import check_image
from langslice.tools.look import show_result
from langslice.tools.refusal import (
    check_section_count,
    refused_row,
    reworded,
    section_refusal,
    unknown_sections,
)
from langslice.tools.run import Run, as_list, drawn_image, pictured
from langslice.tools.stale import section_inputs, stale_refusal
from langslice.tools.transforms import (
    NO_POSITION,
    mirror_message,
    physical_knobs,
    reverses_orientation,
    set_transforms,
)

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


def fit_transform(
    fit: FittedAffine,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    fit_atlas: str = "template",
) -> StoredAffine:
    """The stored record of one Elastix affine fit (kind ``elastix``).

    *fit* is :func:`langslice.core.fits.affine.fit_elastix`'s fitted affine.
    Regions and a non-default atlas image are recorded only when they were
    used.
    """
    regions = None
    if include or exclude:
        regions = FitRegions(include=tuple(include), exclude=tuple(exclude))
    atlas = None if fit_atlas == "template" else fit_atlas
    return StoredAffine(
        kind="elastix",
        params=fit.params,
        physical=fit.physical,
        calibration=fit.calibration,
        mirrored=fit.mirrored,
        regions=regions,
        fit_atlas=atlas,
    )


#: A fit that turns a section further than this from its previous transform
#: reports the turn in its row (``turn_deg``).
LARGE_TURN_DEG = 45.0


def _turn_deg(previous: StoredAffine | None, fit: FittedAffine) -> float:
    """How far a fit turns the section from its previous transform, in degrees (0-180)."""
    before = float(physical_knobs(previous).rotation_deg or 0.0)
    after = float(fit.physical.rotation_deg or 0.0)
    return abs((after - before + 180.0) % 360.0 - 180.0)


#: An affine fit whose scale along either axis changes by more than this
#: ratio either way from the previous transform is flagged ``SCALE_CHANGED``
#: (a section shrinks or swells in processing by much less than that).
SCALE_CHANGE_LIMITS = (0.8, 1.25)
#: An affine fit that changes the shear by more than this is flagged ``SHEAR``.
SHEAR_CHANGE_LIMIT = 0.15


def plausibility_flags(
    previous: StoredAffine | None, fit: FittedAffine
) -> list[dict[str, Any]]:
    """An affine fit's implausible changes from the section's *previous*
    transform (none: the identity), as the deformable fits' flags are given:
    ``SCALE_CHANGED`` (``scale``, ``previous``, ``limits`` per axis ratio) and
    ``SHEAR`` (``shear``, ``previous``, ``limit`` on the change)."""
    before = physical_knobs(previous)
    after = fit.physical
    scale = [float(value or 1.0) for value in (after.scale_x, after.scale_y)]
    held = [float(value or 1.0) for value in (before.scale_x, before.scale_y)]
    low, high = SCALE_CHANGE_LIMITS
    flags: list[dict[str, Any]] = []
    if any(
        not low <= new / old <= high
        for new, old in zip(scale, held, strict=True)
        if old
    ):
        flags.append(
            {
                "code": FitFlag.SCALE_CHANGED,
                "scale": [round(v, 3) for v in scale],
                "previous": [round(v, 3) for v in held],
                "limits": [low, high],
            }
        )
    shear, was = float(after.shear or 0.0), float(before.shear or 0.0)
    if abs(shear - was) > SHEAR_CHANGE_LIMIT:
        flags.append(
            {
                "code": FitFlag.SHEAR,
                "shear": round(shear, 3),
                "previous": round(was, 3),
                "limit": SHEAR_CHANGE_LIMIT,
            }
        )
    return flags


# --- elastix_affine ----------------------------------------------------------


@dataclass(frozen=True)
class AffineRow:
    """One section's answer from :func:`elastix_affine`: its fit, or why not.

    A row without a refusal holds the fit that was written.
    """

    id: str
    #: Why the section was not fitted or not written (``FROZEN``,
    #: ``HAS_DEFORMATION``, the fitter's code, ``STALE_INPUT``); None for a
    #: fit written.
    refusal: Refused | None = None
    fitted: FittedAffine | None = None
    #: What the fit read.
    inputs: InputsBlock | None = None
    #: How far the fit turns the section, when more than
    #: :data:`LARGE_TURN_DEG`.
    turn_deg: float | None = None
    #: The fit's implausible changes (:func:`plausibility_flags`).
    flags: tuple[JsonObject, ...] = ()
    #: The regions' box on the section's canvas, for a restricted fit
    #: (:func:`langslice.tools.fit_regions.region_box`).
    restrict_box: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AffineFit:
    """Hold what :func:`elastix_affine` did, per section and written.

    ``rows``: one :class:`AffineRow` per section asked, in order;
    ``fitted``: the sections written.
    """

    rows: list[AffineRow] = field(default_factory=list[AffineRow])
    fitted: list[str] = field(default_factory=list)


def refit_refusal(section: Section) -> Refused:
    """Refuse a linear refit of a section that holds a deformation.

    Parameters
    ----------
    section
        The section asked for, which holds a deformation.

    Returns
    -------
    Refused
        ``HAS_DEFORMATION``, saying to undo the deformation first.
    """
    return Refused(
        RefusalCode.HAS_DEFORMATION,
        f"{section.id} holds a deformation, which a new transform would "
        "leave misplaced; undo the deformation first, or fit a warp on it.",
    )


def _fit_elastix(
    job: Job,
    workspace: Workspace,
    records: Sequence[Section],
    *,
    fit_atlas: str,
    restrict_to: tuple[str, ...],
    section_image: str = "preprocessed",
    asked_atlas: str = "",
    mask: str = "none",
    edges: bool = True,
) -> AffineFit:
    """Fit an Elastix affine per section and write each fit as its transform.

    Each section's current placement is refined against the atlas image
    *fit_atlas* (:func:`langslice.core.fits.affine.fit_elastix`), by the
    *restrict_to* regions only (empty: every region); each section's marked
    regions are left out on their own
    (:func:`langslice.core.sections.damage.exclusions`). The fit reads
    *section_image* (a section image, by name) with the section mask *mask* by
    name (``"none"`` or one the section holds; a stale one is that section's
    row, ``STALE_MASK``) and, with *edges*, the edge pair; every ok row carries
    the ``inputs`` the fit read
    (:class:`langslice.core.fits.inputs.InputsBlock`) and ``estimated`` (null:
    Elastix estimates no weighting). A section that holds a deformation is
    refused (``HAS_DEFORMATION``), and so is one whose affine is frozen, and so
    is a fit that turns over a section whose orientation is frozen
    (:func:`reverses_orientation`; its row says ``FROZEN``). Every fit that
    succeeded is written as ONE undo step (:func:`set_transforms`,
    :func:`fit_transform` records).

    The fits run outside the job's write lock, from the state as it stood; the
    write takes the lock (:meth:`~langslice.job.job.Job.writing`) and refuses a
    section whose inputs changed meanwhile
    (``STALE_INPUT``, that section's row), writing
    the others.
    """
    state = job.state
    rows: list[AffineRow] = []
    fits: list[
        tuple[int, Section, FittedAffine, tuple[str, ...], tuple[str, ...]]
    ] = []
    expected: dict[str, str] = {}
    for record in records:
        expected[record.id] = section_inputs(state, record)
        frozen = section_refusal(job, record.id, "affine")
        if frozen is not None:
            rows.append(AffineRow(record.id, refusal=frozen))
            continue
        if record.deformation:
            rows.append(AffineRow(record.id, refusal=refit_refusal(record)))
            continue
        kept, dropped = exclusions(record, restrict_to)
        try:
            weights = ops_masks.section_weights(job, record, mask)
        except Refused as refusal:
            rows.append(AffineRow(record.id, refusal=refusal))
            continue
        try:
            with engine_output(
                job.engine_log, f"Elastix affine of {record.id}"
            ):
                outcome = core_affine.fit_elastix(
                    state,
                    workspace,
                    record,
                    include=kept,
                    exclude=dropped,
                    atlas_image=fit_atlas,
                    section_image=section_image,
                    edges=edges,
                    section_weights=weights,
                )
        except Exception as exc:  # nothing is written for this section
            logger.warning("elastix_affine failed for %s: %s", record.id, exc)
            failure = _fit_failure(job, record.id, exc)
            rows.append(AffineRow(record.id, refusal=failure))
            continue
        if isinstance(outcome, Refused):
            rows.append(
                AffineRow(record.id, refusal=_told(job, record.id, outcome))
            )
            continue
        turn_deg = _turn_deg(record.transform, outcome)
        turn = round(turn_deg, 1) if turn_deg > LARGE_TURN_DEG else None
        rows.append(
            AffineRow(
                record.id,
                fitted=outcome,
                inputs=inputs_block(
                    state,
                    workspace,
                    record,
                    section_image=section_image,
                    atlas_image=asked_atlas or fit_atlas,
                    mask=mask,
                    damage=dropped,
                    restrict_to=restrict_to,
                    resolved=fit_atlas,
                ),
                turn_deg=turn,
                flags=tuple(plausibility_flags(record.transform, outcome)),
            )
        )
        fits.append((len(rows) - 1, record, outcome, kept, dropped))
    with job.writing():
        records_out: dict[str, StoredAffine] = {}
        for index, record, outcome, kept, dropped in fits:
            now = state.by_id(record.id)
            if (
                now is None
                or section_inputs(state, now) != expected[record.id]
            ):
                rows[index] = AffineRow(record.id, refusal=stale_refusal())
                continue
            made = fit_transform(
                outcome, include=kept, exclude=dropped, fit_atlas=fit_atlas
            )
            if "orientation" in job.frozen_for(
                record.id
            ) and reverses_orientation(now, made):
                frozen = section_refusal(job, record.id, "orientation")
                assert frozen is not None  # the orientation is frozen
                told = reworded(frozen, mirror_message(record.id))
                rows[index] = AffineRow(record.id, refusal=told)
                continue
            records_out[record.id] = made
        written = set_transforms(job, records_out)
    return AffineFit(rows=rows, fitted=written)


def _told(job: Job, section_id: str, refusal: Refused) -> Refused:
    """Return the fitter's refusal as the row says it.

    A failed fit gets the engine's last words; a message naming another
    tool is rendered for the job's tools.
    """
    if refusal.code == RefusalCode.FIT_FAILED:
        return reworded(
            refusal,
            failure_message(
                refusal.message,
                job.engine_log,
                f"the Elastix affine of {section_id} failed",
            ),
        )
    if marked(refusal.message):  # the message names another tool
        return reworded(
            refusal, render(refusal.message, job.spec.switched_off)
        )
    return refusal


def _fit_failure(job: Job, section_id: str, exc: Exception) -> Refused:
    """Return the row's refusal of a fit that raised: the exception's own code
    and text, or ``FIT_FAILED`` with the engine's last words.
    """
    code = getattr(exc, "code", RefusalCode.FIT_FAILED)
    message = str(exc)
    if code == RefusalCode.FIT_FAILED:
        message = failure_message(
            f"{type(exc).__name__}: {exc}",
            job.engine_log,
            f"the Elastix affine of {section_id} failed",
        )
    return Refused(code, message=message)


def elastix_fit(
    job: Job,
    workspace: Workspace,
    section: Any,
    *,
    restrict_to: Sequence[str] = (),
    exclude: Sequence[str] = (),
    atlas_image: str = "nissl",
    mask: str = "none",
    marked: bool = True,
    section_image: str = "preprocessed",
    edges: bool = True,
) -> core_affine.ElastixFit:
    """One Elastix affine of the section from its current placement, written
    nowhere (a script's ``job.context.fit_elastix``):
    :class:`langslice.core.fits.affine.ElastixFit` (``params``, the six
    normalized numbers; ``report``; ``engine``; ``inputs``, what it read).
    *restrict_to* / *exclude*
    are region entries (checked, ``"CTX:left"`` for one side); *marked* adds
    the section's marked damage regions to *exclude*, as ``elastix_affine``
    leaves them out; *mask* is ``"none"`` or the name of one of the section's
    masks (:mod:`langslice.tools.masks`), its weights under 0.5 left out.

    Refused: ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``, ``BAD_ARGS`` (an image
    the atlas does not have), ``UNKNOWN_IMAGE``, ``UNKNOWN_MASK``,
    ``STALE_MASK``, ``HAS_DEFORMATION``, ``UNKNOWN_REGIONS``, ``NO_SIDES``,
    ``REGION_TOO_SMALL``, ``FIT_FAILED``.
    """
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    if record.position_mm is None:
        raise Refused(
            RefusalCode.NO_POSITION, message=NO_POSITION.format(record.id)
        )
    if record.deformation:
        raise refit_refusal(record)
    try:
        resolved = resolve_image(workspace.atlas, str(atlas_image))
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    section_image = check_image(job, workspace, [record], section_image)
    include = region_entries(workspace, job.state, list(restrict_to))
    dropped = region_entries(workspace, job.state, list(exclude))
    weights = ops_masks.section_weights(job, record, mask)
    marks: tuple[str, ...] = ()
    if marked:
        include, marks = exclusions(record, include)
        dropped = tuple(dict.fromkeys((*dropped, *marks)))
    stored = None if record.transform is None else record.transform.params
    start = (
        list(stored)
        if stored is not None and len(stored) == 6
        else list(IDENTITY_PARAMS)
    )
    try:
        with engine_output(job.engine_log, f"Elastix affine of {record.id}"):
            fit = core_affine.elastix_affine(
                job.state,
                workspace,
                record,
                start,
                include=include,
                exclude=dropped,
                atlas_image=resolved,
                section_image=section_image,
                edges=edges,
                section_weights=weights,
            )
    except Refused as refusal:  # its own code, as the tool's row carries it
        told = render(refusal.message, job.spec.switched_off)
        if not told:
            raise
        raise reworded(refusal, told) from refusal
    except Exception as exc:  # noqa: BLE001 - the engine's failure is the caller's refusal
        raise Refused(
            RefusalCode.FIT_FAILED,
            message=f"{record.id}: "
            + failure_message(
                f"{type(exc).__name__}: {exc}",
                job.engine_log,
                f"the Elastix affine of {record.id} failed",
            ),
        ) from exc
    block = inputs_block(
        job.state,
        workspace,
        record,
        section_image=section_image,
        atlas_image=str(atlas_image),
        mask=str(mask or ops_masks.NO_MASK).strip(),
        damage=dropped,
        restrict_to=include,
        resolved=resolved,
    )
    return replace(fit, inputs=block)


# --- the tool's operation ----------------------------------------------------


def elastix_affine(
    job: Job,
    workspace: Workspace,
    sections: Sequence[str],
    restrict_to: Sequence[str] = (),
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    edges: bool = True,
) -> AffineFit:
    """Fit an in-plane affine with Elastix, per section; ONE undo step for the fits.

    Each section's current placement is refined by matching its section image
    *section_image* against *atlas_image* (one of the atlas's images,
    :func:`langslice.core.atlas.images.atlas_images`; ``nissl`` is its Nissl
    image where it has one, else its template), with the section mask *mask* by
    name (``"none"`` or one each section holds) and, with *edges*, the edge
    pair. *sections* are one or more filenames (``Stack.resolve``).
    *restrict_to* fits by those atlas regions only (sides allowed; empty: every
    region). Each section's marked damage regions are left out automatically
    (:func:`langslice.core.sections.damage.exclusions`). A section that holds a
    deformation is that section's row ``HAS_DEFORMATION``.

    Every ok row of a call with *restrict_to* carries ``restrict_box``: the
    regions' bounding box on that section's canvas as ``[x0, y0, x1, y1]``
    fractions (``core.pictures.canvas.zoom_box``'s zoom), so a later picture
    can zoom on them; absent when the regions are not in that plane. Refused
    (nothing fitted): ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (``atlas_image`` not
    one of the atlas's images, a region that is not text, no section named),
    ``UNKNOWN_IMAGE``, ``UNKNOWN_MASK``.
    """
    if not sections:
        raise Refused(
            RefusalCode.BAD_ARGS,
            message="sections must name one or more sections",
        )

    asked = str(atlas_image or "nissl").strip()
    try:
        image = resolve_image(workspace.atlas, asked)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    if any(not isinstance(name, str) for name in restrict_to):
        raise Refused(
            RefusalCode.BAD_ARGS, message="restrict_to lists atlas regions"
        )
    records = []
    for ref in sections:
        record = job.state.resolve(ref)
        if record is None:
            raise unknown_sections(job.state, [str(ref)])
        records.append(record)
    named = refuse_unknown_inputs(job, workspace, records, section_image, mask)
    label = str(mask or "none").strip()
    result = _fit_elastix(
        job,
        workspace,
        records,
        fit_atlas=image,
        restrict_to=tuple(restrict_to),
        section_image=named,
        asked_atlas=asked,
        mask=label,
        edges=bool(edges),
    )
    if not restrict_to:
        return result
    rows: list[AffineRow] = []
    for row in result.rows:
        record = job.state.by_id(row.id)
        box = None
        if row.refusal is None and record is not None:
            box = region_box(workspace, job.state, record, restrict_to)
        if box is None:
            rows.append(row)
        else:
            rows.append(replace(row, restrict_box=tuple(box)))
    return replace(result, rows=rows)


# --- the tool ----------------------------------------------------------------


def affine_flag_text(flag: Mapping[str, Any]) -> str:
    """Return one ``elastix_affine`` plausibility flag as a sentence.

    Parameters
    ----------
    flag
        One of the fit's flags
        (:func:`plausibility_flags`).

    Returns
    -------
    str
        The flag's code and what it means for the fit; the bare code for a
        flag without words.
    """
    if flag.get("code") == FitFlag.SCALE_CHANGED:
        scale = flag.get("scale") or []
        previous = flag.get("previous") or []
        return (
            f"SCALE_CHANGED: this fit scales the section to {scale} from "
            f"{previous}, more than a section shrinks or swells; it may have "
            "stretched the section over atlas it does not hold. Compare its "
            "picture with the previous one."
        )
    if flag.get("code") == FitFlag.SHEAR:
        return (
            f"SHEAR: this fit shears the section by {flag.get('shear')} (it "
            f"held {flag.get('previous')}); a section is rarely sheared that "
            "much. Compare its picture with the previous one."
        )
    return str(flag.get("code"))


def elastix_affine_tool(
    run: Run,
    sections: Sequence[str],
    restrict_to: Sequence[str],
    *,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    edges: bool = True,
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``elastix_affine`` tool: fit, answer, picture.

    Parameters
    ----------
    run
        The run the tool is called in.
    sections, restrict_to, section_image, atlas_image, mask, edges, view
        The declared arguments as the model sent them
        (``tools/descriptions/fitting.py``).

    Returns
    -------
    dict
        The reply: ``status``, the call's ``settings``, one row per section
        (``results``) and the deformations the fit made stale.
    """
    job, workspace = run.job, run.workspace
    regions = region_entries(workspace, job.state, as_list(restrict_to))
    check_atlas_image(workspace, atlas_image)
    named = as_list(sections)
    if not named:
        raise Refused(
            RefusalCode.BAD_ARGS, "sections must name one or more sections"
        )
    check_section_count(job, "elastix_affine", len(named))
    done = elastix_affine(
        job,
        workspace,
        named,
        regions,
        str(section_image or "preprocessed"),
        str(atlas_image or "nissl"),
        str(mask or "none"),
        bool(edges),
    )
    rows = [affine_row_to_json(row) for row in done.rows]
    fits = [row for row in done.rows if row.refusal is None]
    status = "ok" if fits else "error"
    result: dict[str, Any] = {"status": status}
    if not fits:
        result.update(nothing_fitted(rows))
    result["settings"] = {
        "section_image": str(section_image or "preprocessed"),
        "atlas_image": str(atlas_image or "nissl"),
        "mask": str(mask or "none"),
        "restrict_to": list(regions),
        "edges": bool(edges),
    }
    result["results"] = rows
    if not fits or not run.shows(view):
        return result

    # A deformation this fit made stale goes before the picture is drawn.
    cleared = job.clear_stale_deformations()
    if cleared:
        result["deformation_cleared"] = cleared
    zooms: dict[str, list[float]] = {}
    for row in fits:
        window = zoom_window(row.restrict_box)
        if window:
            zooms[row.id] = window
    shown = show_result(
        job,
        workspace,
        "elastix_affine",
        "overlay",
        [row.id for row in fits],
        zooms=zooms,
        highlight=regions,
        images=drawn_image(section_image),
    )
    return pictured(result, shown)


def affine_row_to_json(row: AffineRow) -> Written:
    """Return one section's row of the ``elastix_affine`` reply.

    Parameters
    ----------
    row
        The section's answer.

    Returns
    -------
    dict
        Its refusal (:func:`langslice.tools.refusal.refused_row`), or the
        fit as every reply gives a fit: ``status``, ``id``,
        ``position_mm``, ``mirrored``, ``calibration``, ``regions`` (a
        restricted fit), ``inputs``, ``estimated`` (null: Elastix estimates
        no weighting), ``flags``, ``restrict_box``, the knobs as its
        ``transform`` (the six raw numbers stay host-side) and the turn
        and flags in words as its ``warning``.
    """
    fitted = row.fitted
    if row.refusal is not None or fitted is None:
        assert row.refusal is not None  # a row holds a fit or a refusal
        return refused_row(row.id, row.refusal)
    out: Written = {
        "status": "ok",
        "id": row.id,
        "position_mm": fitted.position_mm,
        "mirrored": fitted.mirrored,
        "calibration": {
            "section_um_per_px": fitted.calibration.section_um_per_px,
            "source": fitted.calibration.source,
        },
    }
    if fitted.regions is not None:
        out["regions"] = to_jsonable(fitted.regions)
    if row.inputs is not None:
        out["inputs"] = to_jsonable(row.inputs.payload())
    out["estimated"] = None
    if row.flags:
        out["flags"] = to_jsonable(list(row.flags))
    if row.restrict_box is not None:
        out["restrict_box"] = list(row.restrict_box)
    out["transform"] = codec.physical_to_json(fitted.physical)
    words = [affine_flag_text(flag) for flag in row.flags]
    if row.turn_deg is not None:
        words.insert(
            0,
            f"This fit turns the section {row.turn_deg:.0f} degrees "
            "from its previous transform.",
        )
    if words:
        out["warning"] = " ".join(words)
    return out
