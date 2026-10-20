"""Sphinx configuration for the engiNIR documentation set."""

from pathlib import Path
import sys
import tomllib

# Make the src layout visible to autodoc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "engiNIR"
author = "engiNIR contributors"
copyright = "2026, engiNIR contributors"
_pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
release = str(_pyproject["project"]["version"])
version = release

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy / Google docstring styles
    "sphinx.ext.viewcode",  # [source] links
    "sphinx.ext.intersphinx",  # cross-links to numpy / sklearn docs
    "myst_parser",  # Markdown support (.md files)
]

autosummary_generate = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # Attributes sections must not clash with documented dataclass fields

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "examples/**",
]

suppress_warnings = ["app.add_directive", "ref.duplicate"]
