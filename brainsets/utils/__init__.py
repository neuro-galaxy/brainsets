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
            "autosummary": [
                "dandi_utils.extract_subject_from_nwb",
                "dandi_utils.extract_spikes_from_nwbfile",
                "dandi_utils.download_file",
                "dandi_utils.get_nwb_asset_list",
            ],
        },
        {
            "title": "brainsets.utils.split",
            "autosummary": [
                "split.generate_stratified_folds",
                "split.generate_string_kfold_assignment",
            ],
        },
        {
            "title": "brainsets.utils.misc_utils",
            "autosummary": [
                "misc_utils.calculate_sampling_rate",
            ],
        },
    ],
}
