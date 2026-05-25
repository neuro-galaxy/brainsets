_electrophysiology_datasets = [
    "PerichMillerPopulation2018",
    "PeiPandarinathNLB2021",
    "FlintSlutzkyAccurate2012",
    "ChurchlandShenoyNeural2012",
    "OdohertySabesNonhuman2017",
    "VollanMoserAlternating2025",
    "ShiraziHBNR1DS005505",
    "KlinzingSleepDS005555",
]

_calcium_imaging_datasets = [
    "AllenVisualCodingOphys2016",
]

_ieeg_datasets = [
    "Neuroprobe2025",
    "KochiVisualNamingDS006914",
]

_psg_datasets = [
    "KempSleepEDF2013",
]

__all__ = (
    _electrophysiology_datasets
    + _calcium_imaging_datasets
    + _ieeg_datasets
    + _psg_datasets
    + ["OpenNeuroDataset", "OpenNeuroSplitType"]
)

from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType

from .PerichMillerPopulation2018 import PerichMillerPopulation2018
from .PeiPandarinathNLB2021 import PeiPandarinathNLB2021
from .FlintSlutzkyAccurate2012 import FlintSlutzkyAccurate2012
from .ChurchlandShenoyNeural2012 import ChurchlandShenoyNeural2012
from .OdohertySabesNonhuman2017 import OdohertySabesNonhuman2017
from .AllenVisualCodingOphys2016 import AllenVisualCodingOphys2016
from .KempSleepEDF2013 import KempSleepEDF2013
from .Neuroprobe2025 import Neuroprobe2025
from .KlinzingSleepDS005555 import KlinzingSleepDS005555
from .KochiVisualNamingDS006914 import KochiVisualNamingDS006914
from .ShiraziHBNR1DS005505 import ShiraziHBNR1DS005505
from .VollanMoserAlternating2025 import VollanMoserAlternating2025

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Electrophysiology Datasets",
            "template": "dataset.rst",
            "autosummary": _electrophysiology_datasets,
        },
        {
            "title": "Calcium Imaging Datasets",
            "template": "dataset.rst",
            "autosummary": _calcium_imaging_datasets,
        },
        {
            "title": "iEEG Datasets",
            "template": "dataset.rst",
            "autosummary": _ieeg_datasets,
        },
        {
            "title": "PSG Datasets",
            "template": "dataset.rst",
            "autosummary": _psg_datasets,
        },
    ],
}
