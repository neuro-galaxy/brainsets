from . import dandi_utils
from . import split
from . import bids_utils
from . import mne_utils
from . import s3_utils
from . import openneuro
from . import misc_utils

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "brainsets.utils.dandi_utils",
            "autosummary": [f"dandi_utils.{name}" for name in dandi_utils._functions],
        },
        {
            "title": "brainsets.utils.split",
            "autosummary": [f"split.{name}" for name in split._functions],
        },
        {
            "title": "brainsets.utils.misc_utils",
            "autosummary": [f"misc_utils.{name}" for name in misc_utils._functions],
        },
    ],
}
