"""The pictures of a deformable fit: its borders, and a traced section's trace.

:func:`draw_fits` draws, after
:func:`langslice.tools.deformable.fit_deformable` has applied the fits, each
fit's final atlas borders on the image it read and each traced section's
trace on the section, through the core
(:mod:`langslice.core.pictures.deformation`); a packaged trace's landing
draws them with its work (:func:`langslice.tools.traces.land_trace`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.fits import deformation
from langslice.core.fits.deformable.render import drawn_border_coverage
from langslice.core.pictures import deformation as pictures_deformation
from langslice.core.pictures import drawn as pictures_drawn
from langslice.core.pictures.canvas import (
    border_words,
    resolve_border_color,
    validate_border_thickness,
)
from langslice.core.sections import images as images_
from langslice.core.stack.section import Section
from langslice.tools.deformable import DeformableFit, Fitted, fit_regions

if TYPE_CHECKING:
    from langslice.core.pictures.options import DisplayOptions
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


def draw_fits(
    workspace: Workspace,
    done: DeformableFit,
    options: DisplayOptions,
) -> DeformableFit:
    """*done* with its pictures: per drawn row, the final atlas borders on the
    image the fit read (:func:`langslice.core.pictures.deformation.picture`),
    titled
    with the section, engine, smoothness and what was fitted. Then each
    traced section's trace on the section
    (:func:`~langslice.core.pictures.deformation.trace_picture`).
    Included or ``view.regions`` borders strong, excluded ones pink. Each
    drawn row gets ``image_indexes``; a picture that fails is listed in
    ``render_failed``.
    """
    parts: list[Image.Image] = []
    failed: list[dict[str, str]] = []
    thickness = validate_border_thickness(options.border_thickness)
    # `color` is the image model's own fixed yellow (`trace_picture` below
    # draws it plain: its prompt says yellow and its reply is read back by
    # colour). Each fitted-atlas picture gets its cased core (`core`) per row
    # below, resolved against that row's own tissue -- never one for the whole
    # batch, since different sections in one call can need different ones.
    style = pictures_deformation.Style(
        zoom=() if options.full_view else tuple(options.zoom),
        highlight=tuple(name for name, _ids in options.regions),
        marked=(),
        outlines=options.layer,
        color=(255, 255, 0),
        thickness=thickness,
        atlas_opacity=options.atlas_opacity,
        depth=options.depth,
        long_edge=options.long_edge,
    )
    for fitted in done.fitted:
        row, fit, outcome = fitted.row, fitted.fit, fitted.outcome
        record = fit.grid.record
        include, exclude = fit_regions(fit)
        fit_style = replace(
            style, highlight=style.highlight or include, marked=exclude
        )
        coverage = drawn_border_coverage(
            outcome,
            workspace.atlas,
            highlight=fit_style.highlight,
            marked=fit_style.marked,
            warped=True,
            outlines=fit_style.outlines,
            depth=fit_style.depth,
            width_px=fit_style.thickness,
        )
        core, _ = resolve_border_color(
            options.border_color,
            options.border_thickness,
            rgb=np.asarray(fit.image.convert("RGB")),
            coverage=coverage,
        )
        fit_style = replace(fit_style, core=core)
        first = len(parts)
        try:
            parts.append(
                pictures_deformation.picture(
                    workspace,
                    fit.image,
                    outcome,
                    warped=True,
                    style=fit_style,
                    atlas_images=options.atlas_images,
                    title=_fit_title(fitted, include, exclude),
                    note={
                        "sections": (record.id,),
                        "mode": options.mode,
                        "deformation": fitted.folder,
                        "caption": fit_caption(
                            record,
                            fit,
                            row,
                            include,
                            exclude,
                            border_words(core),
                        ),
                    },
                )
            )
        except Exception as exc:
            logger.warning(
                "fit_deformable picture failed for %s",
                record.id,
                exc_info=True,
            )
            del parts[first:]
            failed.append({"id": record.id, "message": str(exc)})
            continue
        row["image_indexes"] = list(range(first, len(parts)))
    # Each traced section's trace, once per call, so it can be reviewed.
    traces: list[dict[str, Any]] = []
    where = {
        fitted.fit.grid.record.id: fitted.fit.grid.record
        for fitted in done.fitted
    }
    for section_id, (image, lines) in done.traced.items():
        record = where.get(section_id)
        try:
            picture = _trace_picture(section_id, image, lines, style, record)
        except Exception as exc:
            logger.warning(
                "trace picture failed for %s", section_id, exc_info=True
            )
            failed.append({"id": section_id, "message": str(exc)})
            continue
        traces.append({"id": section_id, "image_indexes": [len(parts)]})
        parts.append(picture)
    return replace(done, pictures=parts, traces=traces, render_failed=failed)


def _fit_title(
    fitted: Fitted, include: Sequence[str], exclude: Sequence[str]
) -> str:
    """Return a fit picture's title.

    The section, the engine and smoothness, then what was fitted against
    what, from where, by which regions.
    """
    fit, outcome = fitted.fit, fitted.outcome
    heading = (
        f"{fit.grid.record.id}  applied: {outcome.engine['name']} "
        f"smoothness {fit.choice.smoothness_um:g} um"
    )
    start = "current" if fit.previous is not None else "linear"
    detail_line = (
        f"{fit.choice.fit_section} vs {fit.choice.fit_atlas}, start {start}"
    )
    if include:
        detail_line += f", include {','.join(include)}"
    if exclude:
        detail_line += f", exclude {','.join(exclude)}"
    return f"{heading}\n{detail_line}"


def _trace_picture(
    section_id: str,
    image: Image.Image,
    lines: np.ndarray,
    style: pictures_deformation.Style,
    section: Section | None,
) -> Image.Image:
    """Return a traced section's trace drawn on it, with its index caption."""
    picture = pictures_deformation.trace_picture(
        image,
        lines,
        style=style,
        title=f"{section_id}  trace_borders result: the image model's lines",
    )
    at = ""
    if section is not None and section.position_mm is not None:
        at = f", at {float(section.position_mm):.2f} mm"
    caption = (
        f"{section_id} trace_borders: the image model's traced lines drawn "
        f"on the section{at}"
    )
    return pictures_drawn.note(
        picture, sections=(section_id,), mode="trace", caption=caption
    )


#: How a fit picture's caption names what of the section a fit read.
_FIT_SECTION_WORDS = {
    images_.PREPROCESSED: "the preprocessed image",
    deformation.TRACED_BORDERS: "the image model's traced borders",
}


def fit_caption(
    record: Section,
    fit: deformation.Job,
    row: dict[str, Any],
    include: Sequence[str],
    exclude: Sequence[str],
    border_word: str = "",
) -> str:
    """The index caption of a deformable fit's picture: the section, the
    fit (engine, smoothness, what it read against which atlas image), the
    position, the regions drawn thick or left out, the colour the fitted
    borders are drawn in, and the fit's numbers."""
    choice = fit.choice
    read = _FIT_SECTION_WORDS.get(choice.fit_section, choice.fit_section)
    at = (
        f", at {float(record.position_mm):.2f} mm"
        if record.position_mm is not None
        else ""
    )
    numbers = cast("dict[str, object]", row.get("displacement_mm") or {})
    facts = (
        (
            f"displacement median {numbers.get('median')} mm, max {numbers.get('max')} mm; "
            f"fold fraction {row.get('fold_fraction')}"
        )
        if numbers
        else ""
    )
    engine = getattr(fit.result, "engine", {}).get("name")
    return (
        f"{record.id} deformation fit ({engine} smoothness {choice.smoothness_um:g} um, "
        f"{read} against the atlas {choice.fit_atlas}){at}; the fitted atlas borders "
        + (f"({border_word}) " if border_word else "")
        + "on the image the fit read"
        + (
            f"; regions {', '.join(include)} drawn thick, the other borders faint"
            if include
            else ""
        )
        + (
            f"; left out (second colour): {', '.join(exclude)}"
            if exclude
            else ""
        )
        + (f"; {facts}" if facts else "")
    )
