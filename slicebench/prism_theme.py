"""GraphPad-Prism-style theme for plotnine.

Hand-crafted on top of `theme_classic()`. plotnine has no built-in Prism
theme; this matches the signature visual cues:
  - white background, no grid
  - thick black axis lines on bottom and left only
  - inward-pointing axis ticks
  - bold sans-serif axis labels and titles
  - no legend frame
"""

from __future__ import annotations

from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_classic,
)


def theme_prism(base_size: int = 12, base_family: str = "Arial") -> theme:
    """Return a plotnine theme that emulates GraphPad-Prism's look."""
    return theme_classic(base_size=base_size, base_family=base_family) + theme(
        # Backgrounds
        panel_background=element_rect(fill="white", color="white"),
        plot_background=element_rect(fill="white", color="white"),

        # No grid (Prism never shows grid)
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),

        # Thick black axis lines, bottom + left only (theme_classic already drops top/right)
        axis_line=element_line(color="black", size=0.9),

        # Inward-pointing ticks: plotnine 0.15 expresses this via NEGATIVE length
        axis_ticks=element_line(color="black", size=0.7),
        axis_ticks_length=-4,

        # Bold black axis text and titles
        axis_text=element_text(weight="bold", color="black", size=base_size),
        axis_title=element_text(weight="bold", color="black", size=base_size + 1),

        # Legend: no box, no key background
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_title=element_text(weight="bold", size=base_size),
        legend_text=element_text(size=base_size - 1),

        # Facet strips: bold but no boxed background (Prism rarely uses facets)
        strip_background=element_blank(),
        strip_text=element_text(weight="bold", size=base_size),

        # Plot title
        plot_title=element_text(weight="bold", size=base_size + 2, ha="left"),
    )


# Named color palette tuned to be readable when models are bar/line groups.
PRISM_PALETTE = [
    "#2E86AB",  # blue
    "#E63946",  # red
    "#F4A261",  # amber
    "#2A9D8F",  # teal
    "#6A4C93",  # purple
    "#E76F51",  # orange
    "#264653",  # dark teal
    "#A8DADC",  # pale teal
]
