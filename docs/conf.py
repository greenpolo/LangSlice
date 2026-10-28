"""Sphinx configuration for the LangSlice documentation."""

project = "LangSlice"
author = "LangSlice contributors"
copyright = "LangSlice contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 6

exclude_patterns = ["requirements.txt", "_build"]

html_theme = "pydata_sphinx_theme"
html_title = "LangSlice"
html_baseurl = "https://langslice.readthedocs.io/"
html_theme_options = {
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
}
html_context = {
    "github_user": "greenpolo",
    "github_repo": "LangSlice",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars = {"**": []}  # the top bar carries the nav; the left sidebar would be empty
