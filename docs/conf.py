"""Sphinx configuration for the LangSlice documentation."""

project = "LangSlice"
author = "LangSlice contributors"
copyright = "LangSlice contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",  # the Installation page's tab per AI app
]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 6

exclude_patterns = [
    "requirements.txt", "_build", "_extra", "_includes",
    # Reference for agents and developers: kept at these paths in the repository (code, the connectors' skills,
    # the CLAUDE.md guides and the Installation page's message for an agent cite them) and read on GitHub; the
    # site carries only what a user needs.
    "agents/*", "reference/*", "developers/*",
]
templates_path = ["_templates"]
html_extra_path = ["_extra"]  # llms.txt at the site root

html_theme = "pydata_sphinx_theme"
html_title = "LangSlice"
html_baseurl = "https://langslice.readthedocs.io/"
html_favicon = "../assets/logo/logomark.svg"
html_static_path = ["_static"]
# The look of the ASCII card on the front page: one monospace face on black, the logo's colours.
html_css_files = ["langslice.css"]
html_js_files = [("langslice.js", {"defer": "defer"})]  # names the region under the pointer on the front page
html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    # Paths are relative to docs/; the theme copies the files into _static.
    "logo": {
        # The site is dark in either colour mode, so both modes show the lockup drawn for black.
        "image_light": "../assets/logo/lockup_dark.svg",
        "image_dark": "../assets/logo/lockup_dark.svg",
        "alt_text": "LangSlice - Home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/greenpolo/LangSlice",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "navbar_align": "left",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    # The top bar holds the logo, search and GitHub; the pages are listed in the left bar.
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"],  # no light/dark switch: the docs have one look
    "primary_sidebar_end": [],
    # The front page's hero needs the width, so it has no "On this page" column.
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page"],
        "index": [],
    },
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_context = {
    "github_user": "greenpolo",
    "github_repo": "LangSlice",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "dark",
}
# Every page, the front page included, lists the whole site on the left, grouped by category
# (_templates/sidebar-categories.html).
html_sidebars = {"**": ["sidebar-categories"]}
