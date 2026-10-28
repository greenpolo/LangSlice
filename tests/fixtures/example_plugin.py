"""A small tool plugin on synthetic data: one read tool and two write tools.

It is the example ``docs/developers/plugins.md`` points to and what the plugin tests load
(``tests.fixtures.example_plugin:tools``). ``section_brightness`` reads each
section's image and says how bright it is; ``nudge_sections`` writes an
in-plane shift through the context, so it is one undo step and obeys what the
job froze; ``ripple_sections`` writes a deformation, the full map from the fit
grid to the native atlas plane that a deformable engine would return. None
needs an engine; a real plugin would call one as a subprocess from its handler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.plugin_api import ToolContext, ToolResult, ToolSpec

PLUGIN_API_VERSION = 1


def section_brightness(sections: list[str] = []) -> dict[str, Any]:  # noqa: B006
    """Report how bright each section is. Writes nothing.

    Args:
        sections: Filenames; empty is every section.

    Returns:
        One row per section with `mean` brightness (0 to 255) and `position_mm`.
    """
    ...


def nudge_sections(sections: list[str], dx: float = 0.01) -> dict[str, Any]:  # noqa: B006
    """Shift sections sideways in the plane by dx of their width (1 is one width).

    Args:
        sections: Filenames of the sections to shift.
        dx: The shift as a fraction of the section's width.

    Returns:
        One row per section: `ok`, or the reason it was left alone.
    """
    ...


def ripple_sections(sections: list[str], amplitude_px: float = 1.0) -> dict[str, Any]:
    """Warp sections by a gentle ripple on top of their linear placement.

    Args:
        sections: Filenames of the sections to warp.
        amplitude_px: The ripple's height in fit-grid pixels.

    Returns:
        One row per section: `ok` with the warp's numbers, or the reason it was left alone.
    """
    ...


def ripple_map(frame: Any, amplitude_px: float) -> np.ndarray:
    """The full map a deformable engine started from the section's current
    placement would return for *frame*: each fit-grid pixel centre's native atlas
    pixel, a sideways ripple of *amplitude_px* fit-grid pixels followed by the
    linear placement."""
    height, width = frame.image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    moved_x = xx + amplitude_px * np.sin(2 * np.pi * yy / max(height, 1))
    placed = frame.placed_atlas_to_section
    to_native = np.linalg.inv(frame.atlas_to_section if placed is None else placed)
    return np.stack([to_native[0, 0] * moved_x + to_native[0, 1] * yy + to_native[0, 2],
                     to_native[1, 0] * moved_x + to_native[1, 1] * yy + to_native[1, 2]],
                    axis=-1)


def _brightness(ctx: ToolContext, sections: list[str] | None = None, **_: Any) -> ToolResult:
    names = list(sections or []) or [view.id for view in ctx.sections()]
    rows = []
    for name in names:
        view = ctx.section(name)
        mean = float(ctx.section_image(name, level=64).mean())
        rows.append({"id": view.id, "mean": round(mean, 1), "position_mm": view.position_mm})
    return ToolResult({"rows": rows})


def _nudge(ctx: ToolContext, sections: list[str], dx: float = 0.01) -> ToolResult:
    rows = [{"id": name, "params": [1.0, 0.0, float(dx), 0.0, 1.0, 0.0]} for name in sections]
    done = ctx.set_transforms(rows)
    reply = done.reply()
    pictures = []
    if done.written:
        pictures.append(ctx.overlay_picture(done.written[0]))
    return ToolResult(reply, pictures)


def _ripple(ctx: ToolContext, sections: list[str], amplitude_px: float = 1.0) -> ToolResult:
    rows = [{"id": name, "map": ripple_map(ctx.fit_frame(name), float(amplitude_px)),
             "note": f"ripple {amplitude_px} px"} for name in sections]
    done = ctx.set_deformations(rows)
    pictures = [ctx.overlay_picture(done.written[0])] if done.written else []
    return ToolResult(done.reply(), pictures)


def tools() -> list[ToolSpec]:
    return [
        ToolSpec("section_brightness", section_brightness, _brightness, kind="read",
                 group="Common", version="1"),
        ToolSpec("nudge_sections", nudge_sections, _nudge, kind="write", group="Linear",
                 task="transform", writes=("affine",), limits={"sections": 4}, version="1"),
        ToolSpec("ripple_sections", ripple_sections, _ripple, kind="write", group="Nonlinear",
                 task="nonlinear", writes=("deformation",), limits={"sections": 4},
                 version="1"),
    ]
