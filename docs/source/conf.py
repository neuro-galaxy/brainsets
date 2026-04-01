import os
import datetime
import inspect

import brainsets
import brainsets.taxonomy
import brainsets.descriptions
import brainsets.utils.mat_utils
import brainsets.utils.dandi_utils
import brainsets.utils.dir_utils
import brainsets.utils.split
import glob as glob
import brainsets.processing

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
autosummary_generate = True

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


taxonomy_classes = [
    name
    for name, obj in inspect.getmembers(brainsets.taxonomy, inspect.isclass)
    if obj.__module__.startswith("brainsets.taxonomy") and not name.startswith("_")
]

description_classes = [
    name
    for name, obj in inspect.getmembers(brainsets.descriptions, inspect.isclass)
    if obj.__module__ == "brainsets.descriptions" and not name.startswith("_")
]


def _get_module_fns(module):
    return [
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if obj.__module__ == module.__name__ and not name.startswith("_")
    ]


mat_utils_fns = _get_module_fns(brainsets.utils.mat_utils)
dandi_utils_fns = _get_module_fns(brainsets.utils.dandi_utils)
dir_utils_fns = _get_module_fns(brainsets.utils.dir_utils)
split_fns = _get_module_fns(brainsets.utils.split)
signal_processing_fns = _get_module_fns(brainsets.processing.signal)


_generated_dir = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(_generated_dir, exist_ok=True)

# Remove stale stubs so deleted or renamed symbols don't persist across builds
for _stale in glob.glob(os.path.join(_generated_dir, "*.rst")):
    os.remove(_stale)


def _write_class_stub(generated_dir, module, name):
    stub_path = os.path.join(generated_dir, f"{module}.{name}.rst")
    underline = "=" * len(name)
    with open(stub_path, "w") as f:
        f.write(f"{name}\n{underline}\n\n")
        f.write(f".. currentmodule:: {module}\n\n")
        f.write(f".. autoclass:: {name}\n")
        f.write(f"   :members:\n")
        f.write(f"   :show-inheritance:\n")
        f.write(f"   :undoc-members:\n")
        f.write(f"   :member-order: bysource\n")


for _name in description_classes:
    _write_class_stub(_generated_dir, "brainsets.descriptions", _name)

for _name in taxonomy_classes:
    _write_class_stub(_generated_dir, "brainsets.taxonomy", _name)


def _write_function_stub(generated_dir, module, name):
    stub_path = os.path.join(generated_dir, f"{module}.{name}.rst")
    underline = "=" * len(name)
    with open(stub_path, "w") as f:
        f.write(f"{name}\n{underline}\n\n")
        f.write(f".. currentmodule:: {module}\n\n")
        f.write(f".. autofunction:: {name}\n")


for _name in mat_utils_fns:
    _write_function_stub(_generated_dir, "brainsets.utils.mat_utils", _name)
for _name in dandi_utils_fns:
    _write_function_stub(_generated_dir, "brainsets.utils.dandi_utils", _name)
for _name in dir_utils_fns:
    _write_function_stub(_generated_dir, "brainsets.utils.dir_utils", _name)
for _name in split_fns:
    _write_function_stub(_generated_dir, "brainsets.utils.split", _name)
for _name in signal_processing_fns:
    _write_function_stub(_generated_dir, "brainsets.processing.signal", _name)


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, "templates"):
        rst_context = {
            "brainsets": brainsets,
            "taxonomy": brainsets.taxonomy,
            "taxonomy_classes": taxonomy_classes,
            "description_classes": description_classes,
            "mat_utils_fns": mat_utils_fns,
            "dandi_utils_fns": dandi_utils_fns,
            "dir_utils_fns": dir_utils_fns,
            "split_fns": split_fns,
            "signal_processing_fns": signal_processing_fns,
        }
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect("source-read", rst_jinja_render)
