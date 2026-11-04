# Sphinx configuration of the healpix-stl-ia documentation.
#
# The API pages (autosummary) and the command-line pages (sphinx-argparse) are
# generated from the code at every build, so that they cannot drift from it;
# the CI builds with warnings as errors on every push.
import datetime as dt
from importlib.metadata import PackageNotFoundError, version as _version

project = "healpix-stl-ia"
author = "Healpix-STL-IA developers"
copyright = f"2026-{dt.datetime.now().year}, {author}"
try:
    release = _version("healpix-stl-ia")
except PackageNotFoundError:
    release = "dev"
version = release

root_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "myst_parser",
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
mathjax3_config = {"tex": {"macros": {"nside": r"N_{\rm side}"}}}
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# The STL-Dev-MPI engine is not a PyPI package: the documentation builds without it.
autodoc_mock_imports = ["stl_port"]
autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "pydata_sphinx_theme"
html_title = "healpix-stl-ia"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/jmdelouis/Healpix-STL-IA",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
