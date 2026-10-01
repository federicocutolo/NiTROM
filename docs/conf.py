"""Sphinx configuration for the NiTROM documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nitrom import __version__ as nitrom_version  # noqa: E402

# -- Project information -----------------------------------------------------

project = "NiTROM"
author = "Alberto Padovan, Cole Errico, and contributors"
copyright = "2026, the NiTROM developers"
version = nitrom_version
release = nitrom_version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True
nitpick_ignore_regex = [
    # Prose used in :type: fields, not importable targets.
    ("py:class", r"backend (array|dtype)"),
    ("py:class", r"array-like"),
    ("py:class", r"optional"),
    ("py:class", r"scipy sparse matrix"),
    ("py:attr", r"xp"),
    # Private helpers referenced from public docstrings.
    ("py:.*", r"_.*"),
    ("py:.*", r".*\._.*"),
    # Members defined on several classes: a bare reference is ambiguous, so the
    # suffix resolver (see below) correctly refuses to pick one.
    ("py:attr", r"(.*\.)?param_names"),
    ("py:meth", r"(.*\.)?(inner_params|update_params)"),
    # Documented via :param: fields rather than as attributes.
    ("py:attr", r"(.*\.)?(forcing_exists|RegisteredParam\.shared|shared)"),
    ("py:attr", r"X"),
    # torch registers this as torch.nn.parameter.Parameter in intersphinx.
    ("py:class", r"torch\.nn\.Parameter"),
]

# -- Autodoc / autosummary ---------------------------------------------------

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = ["mpi4py"]
autosummary_generate = False

# -- Napoleon ----------------------------------------------------------------
# Docstrings are predominantly reST field lists (passed through untouched);
# NumPy style is enabled for the stragglers still being converted.

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Sphinx-Gallery ----------------------------------------------------------
# The toymodel scripts form a pipeline, so they must run in this order (the
# training scripts populate models_continuous_adjoint/, which read_results.py
# then loads).

from sphinx_gallery.sorting import FunctionSortKey  # noqa: E402
from toymodel_order import toymodel_sort_key  # noqa: E402

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/toymodel"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"\.py",
    "ignore_pattern": r"(fom_class|time_calls|_paper|sweep|check_gradients)",
    "within_subsection_order": FunctionSortKey(toymodel_sort_key),
    "remove_config_comments": True,
    "download_all_examples": False,
    # NITROM_DOCS_NO_GALLERY=1 skips executing the examples (used by the
    # linkcheck CI job; the pages are still generated, without outputs).
    "plot_gallery": not os.environ.get("NITROM_DOCS_NO_GALLERY"),
}

# -- Linkcheck ---------------------------------------------------------------
# PyTorch's docs attach anchors via JavaScript, so linkcheck cannot see them.

linkcheck_anchors_ignore_for_url = [r"https://(docs\.)?pytorch\.org/.*"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- Diagrams ----------------------------------------------------------------

graphviz_output_format = "svg"
inheritance_graph_attrs = {"rankdir": "TB", "bgcolor": "transparent"}

# -- Short cross-reference resolution ----------------------------------------
# Docstrings reference siblings by short name (:class:`PolynomialModel`) and
# narrative pages by re-export path (nitrom.projections.LinearProjection),
# while autodoc registers every object under its leaf module. Resolve a
# missing py reference by unique suffix match against the documented objects.


def _resolve_short_ref(app, env, node, contnode):
    if node.get("refdomain") != "py":
        return None
    from sphinx.util.nodes import make_refnode

    objects = env.domains["py"].objects
    parts = node["reftarget"].split(".")
    for k in range(len(parts)):
        suffix = ".".join(parts[k:])
        matches = {
            name: entry
            for name, entry in objects.items()
            if name == suffix or name.endswith("." + suffix)
        }
        if len(matches) == 1:
            name, entry = next(iter(matches.items()))
            return make_refnode(
                app.builder, node["refdoc"], entry.docname, entry.node_id,
                contnode, name,
            )
        if len(matches) > 1:
            return None  # ambiguous: leave unresolved rather than guess
    return None


def setup(app):
    app.connect("missing-reference", _resolve_short_ref)


# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"NiTROM {nitrom_version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
