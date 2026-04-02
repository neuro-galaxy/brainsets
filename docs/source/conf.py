import os
import datetime
import glob
import jinja2

import brainsets
import brainsets.taxonomy
import brainsets.descriptions
import brainsets.core
import brainsets.utils.mat_utils
import brainsets.utils.dandi_utils
import brainsets.utils.dir_utils
import brainsets.utils.split
import brainsets.processing.signal

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

add_module_names = True
autodoc_member_order = "bysource"
autosummary_generate = True

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "temporaldata": ("https://temporaldata.readthedocs.io/en/latest/", None),
    "torch_brain": ("https://torch-brain.readthedocs.io/en/latest/", None),
    "pydantic": ("https://pydantic.dev", None),
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


# Remove stale stubs so deleted or renamed symbols don't persist across builds
_generated_dir = os.path.join(os.path.dirname(__file__), "_generated")
os.makedirs(_generated_dir, exist_ok=True)
for _stale in glob.glob(os.path.join(_generated_dir, "*.rst")):
    os.remove(_stale)

# Generate stubs at conf.py import time so they exist before autosummary scans
# source files (autosummary pre-generation runs before source-read events fire,
# so Jinja2-rendered content would otherwise be invisible to it).
for _mod_name, _mod in [
    ("brainsets.taxonomy", brainsets.taxonomy),
    ("brainsets.descriptions", brainsets.descriptions),
    ("brainsets.core", brainsets.core),
    ("brainsets.utils.mat_utils", brainsets.utils.mat_utils),
    ("brainsets.utils.dandi_utils", brainsets.utils.dandi_utils),
    ("brainsets.utils.dir_utils", brainsets.utils.dir_utils),
    ("brainsets.utils.split", brainsets.utils.split),
    ("brainsets.processing.signal", brainsets.processing.signal),
]:
    for _name in getattr(_mod, "_classes", []):
        with open(os.path.join(_generated_dir, f"{_mod_name}.{_name}.rst"), "w") as _f:
            _f.write(f"{_name}\n{'=' * len(_name)}\n\n")
            _f.write(f".. currentmodule:: {_mod_name}\n\n")
            _f.write(f".. autoclass:: {_name}\n")
            _f.write(
                f"   :members:\n   :show-inheritance:\n   :undoc-members:\n   :member-order: bysource\n"
            )
    for _name in getattr(_mod, "_functions", []):
        with open(os.path.join(_generated_dir, f"{_mod_name}.{_name}.rst"), "w") as _f:
            _f.write(f"{_name}\n{'=' * len(_name)}\n\n")
            _f.write(f".. currentmodule:: {_mod_name}\n\n")
            _f.write(f".. autofunction:: {_name}\n")


def rst_jinja_render(app, _, source):
    rst_context = {"brainsets": brainsets}
    source[0] = jinja2.Environment().from_string(source[0]).render(rst_context)


def setup(app):
    app.connect("source-read", rst_jinja_render)
