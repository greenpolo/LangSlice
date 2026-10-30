"""The ``elastix_affine`` tool: the Elastix affine fit, its reply and picture.

The fit itself, one undo step for the call, is
:func:`langslice.tools.transforms.elastix_affine`. The tool answers with a
row per section, worded from the fit's numbers (how far it turned the
section, the flags of an implausible change: :func:`affine_flag_text`), and
shows each fitted section under its new registration with the atlas
borders on it (:func:`langslice.tools.look.show_result`), drawn on the
section image the fit read, unless the call's ``view`` is false and the
host does not force it. With ``restrict_to`` those regions' borders are
drawn thick and the others faint, and the picture is zoomed to the
regions' box when that magnifies it at least
:data:`~langslice.tools.transforms.MIN_ZOOM_GAIN` times.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langslice.core.fits.flags import FitFlag
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.tools import transforms
from langslice.tools.catalog import check_section_count
from langslice.tools.deformable import (
    check_atlas_image,
    nothing_fitted,
    region_entries,
)
from langslice.tools.look import show_result
from langslice.tools.run import Run, as_list, drawn_image, pictured


def affine_flag_text(flag: Mapping[str, Any]) -> str:
    """Return one ``elastix_affine`` plausibility flag as a sentence.

    Parameters
    ----------
    flag
        One of the fit's flags
        (:func:`langslice.tools.transforms.plausibility_flags`).

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
        The run the tool belongs to.
    sections, restrict_to, section_image, atlas_image, mask, edges
        The call's arguments, as the fit takes them
        (:func:`langslice.tools.transforms.elastix_affine`).
    view
        Whether to answer with the picture.

    Returns
    -------
    dict
        The reply: ``status``, the call's ``settings``, one row per section
        (``results``) and the deformations the fit made stale.
    """
    job, workspace = run.job, run.ctx
    regions = region_entries(workspace, job.state, as_list(restrict_to))
    check_atlas_image(workspace, atlas_image)
    named = as_list(sections)
    if not named:
        raise Refused(
            RefusalCode.BAD_ARGS, "sections must name one or more sections"
        )
    check_section_count(job, "elastix_affine", len(named))
    done = transforms.elastix_affine(
        job,
        workspace,
        named,
        regions,
        str(section_image or "preprocessed"),
        str(atlas_image or "nissl"),
        str(mask or "none"),
        bool(edges),
    )
    # A fit's row as every reply gives a fit: the six raw numbers stay
    # host-side, the knobs are its transform, the turn and flags its warning.
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for row in done.rows:
        if row.get("status") != ReplyStatus.OK:
            rows.append(row)
            continue
        worded = {
            key: value
            for key, value in row.items()
            if key not in ("params", "physical", "turn_deg")
        }
        worded["transform"] = row.get("physical")
        words = [affine_flag_text(flag) for flag in row.get("flags") or []]
        turn = row.get("turn_deg")
        if turn is not None:
            words.insert(
                0,
                f"This fit turns the section {turn:.0f} degrees "
                "from its previous transform.",
            )
        if words:
            worded["warning"] = " ".join(words)
        rows.append(worded)
        fits.append(worded)
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
        window = transforms.zoom_window(row.get("restrict_box"))
        if window:
            zooms[str(row["id"])] = window
    shown = show_result(
        job,
        workspace,
        "elastix_affine",
        "overlay",
        [str(row["id"]) for row in fits],
        zooms=zooms,
        highlight=regions,
        images=drawn_image(section_image),
    )
    return pictured(result, shown)
