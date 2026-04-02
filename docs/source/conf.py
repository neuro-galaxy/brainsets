import os
import datetime
import inspect
import importlib

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


def _get_module_classes(module):
    return [
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__.startswith(module.__name__) and not name.startswith("_")
    ]


def _get_module_fns(module):
    return [
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if obj.__module__.startswith(module.__name__) and not name.startswith("_")
    ]


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


def _write_function_stub(generated_dir, module, name):
    stub_path = os.path.join(generated_dir, f"{module}.{name}.rst")
    underline = "=" * len(name)
    with open(stub_path, "w") as f:
        f.write(f"{name}\n{underline}\n\n")
        f.write(f".. currentmodule:: {module}\n\n")
        f.write(f".. autofunction:: {name}\n")


modules = [
    "brainsets.descriptions",
    "brainsets.taxonomy",
    "brainsets.core",
    "brainsets.utils.mat_utils",
    "brainsets.utils.dandi_utils",
    "brainsets.utils.dir_utils",
    "brainsets.utils.split",
    "brainsets.processing.signal",
]

_generated_dir = os.path.join(os.path.dirname(__file__), "_generated")
os.makedirs(_generated_dir, exist_ok=True)

# Remove stale stubs so deleted or renamed symbols don't persist across builds
for _stale in glob.glob(os.path.join(_generated_dir, "*.rst")):
    os.remove(_stale)

rst_context = {
    "brainsets": brainsets,
    "taxonomy": brainsets.taxonomy,
}

for module_name in modules:
    m = importlib.import_module(module_name)
    context_name = "_".join(module_name.split(".")[1:])

    classes = _get_module_classes(m)
    for c in classes:
        _write_class_stub(_generated_dir, module_name, c)
    rst_context[f"{context_name}_classes"] = classes

    fns = _get_module_fns(m)
    for f in fns:
        _write_function_stub(_generated_dir, module_name, f)
    rst_context[f"{context_name}_fns"] = fns


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, "templates"):
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect("source-read", rst_jinja_render)
