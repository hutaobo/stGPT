from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

project = "stGPT"
author = "Taobo Hu"
copyright = "2026, Taobo Hu"


def _read_version() -> str:
    init_file = ROOT / "src" / "stgpt" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_file.read_text(encoding="utf-8"))
    return match.group(1) if match else "0.0.0"


release = _read_version()
version = release

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "stGPT"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/hutaobo/stGPT",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "navigation_with_keys": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "home_page_in_toc": True,
}

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]

nb_execution_mode = "off"
nb_execution_timeout = 600
nb_merge_streams = True
