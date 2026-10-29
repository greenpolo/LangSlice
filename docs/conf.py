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

exclude_patterns = ["requirements.txt", "_build", "_extra"]
html_extra_path = ["_extra"]  # llms.txt at the site root

html_theme = "pydata_sphinx_theme"
html_title = "LangSlice"
html_baseurl = "https://langslice.readthedocs.io/"
html_favicon = "../assets/logo/logomark.svg"
html_static_path = ["_static"]
html_css_files = ["brand.css"]
html_theme_options = {
    # Paths are relative to docs/; the theme copies the files into _static.
    "logo": {
        "image_light": "../assets/logo/lockup_light.svg",
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
}
html_context = {
    "github_user": "greenpolo",
    "github_repo": "LangSlice",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars = {"**": []}  # the top bar carries the nav; the left sidebar would be empty
