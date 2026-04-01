import datetime
import inspect

import brainsets

author = "neuro-galaxy Team"
project = "brainsets"
version = brainsets.__version__
copyright = f"{datetime.datetime.now().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "temporaldata": ("https://temporaldata.readthedocs.io/en/latest/", None),
    "torch_brain": ("https://torch-brain.readthedocs.io/en/latest/", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
]

pygments_style = "default"

html_css_files = [
    "style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
]

html_js_files = [
    "citation.js",
]
html_copy_source = False
html_show_sourcelink = True
html_logo = "_static/brainsets_logo.png"
html_favicon = "_static/brainsets_logo.png"

import brainsets.taxonomy

taxonomy_classes = [
    name
    for name, obj in inspect.getmembers(brainsets.taxonomy, inspect.isclass)
    if obj.__module__.startswith("brainsets.taxonomy") and not name.startswith("_")
]


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, "templates"):
        rst_context = {
            "brainsets": brainsets,
            "taxonomy": brainsets.taxonomy,
            "taxonomy_classes": taxonomy_classes,
        }
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect("source-read", rst_jinja_render)