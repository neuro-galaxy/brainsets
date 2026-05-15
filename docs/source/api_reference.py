from importlib import import_module
from pathlib import Path
import jinja2

import brainsets
import brainsets.pipeline
import brainsets.datasets
import brainsets.descriptions
import brainsets.taxonomy
import brainsets.core
import brainsets.processing.signal
import brainsets.utils.dandi_utils
import brainsets.utils.split

"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to the modules's __api_ref__. Each module's
__api_ref__ consists of the following components:

description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

See `torch_brain/docs/source/api_reference.py` for the full template documentation.
"""


# Modules to include in API reference.
API_MODS = [
    "brainsets.pipeline",
    "brainsets.datasets",
    "brainsets.descriptions",
    "brainsets.taxonomy",
    "brainsets.core",
    "brainsets.processing.signal",
    "brainsets.utils.dandi_utils",
    "brainsets.utils.split",
]

API_REFERENCE = {m: import_module(m).__api_ref__ for m in API_MODS}


def build_api_rst():
    generated = Path(__file__).parent / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "api").mkdir(exist_ok=True)

    rst_templates: list[dict] = [
        {
            "template_path": "api/index.rst.template",
            "target_path": "generated/api/index.rst",
            "kwargs": {"API_REFERENCE": API_REFERENCE.items()},
        }
    ]

    for module in API_REFERENCE:
        rst_templates.append(
            {
                "template_path": "api/module.rst.template",
                "target_path": f"generated/api/{module}.rst",
                "kwargs": {"module": module, "module_info": API_REFERENCE[module]},
            }
        )

    for template in rst_templates:
        with open(template["template_path"], "r") as f:
            t = jinja2.Template(f.read())

        with open(template["target_path"], "w") as f:
            f.write(t.render(**template["kwargs"]))
