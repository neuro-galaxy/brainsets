from .PerichMillerPopulation2018 import PerichMillerPopulation2018
from .PeiPandarinathNLB2021 import PeiPandarinathNLB2021
from .FlintSlutzkyAccurate2012 import FlintSlutzkyAccurate2012
from .ChurchlandShenoyNeural2012 import ChurchlandShenoyNeural2012
from .OdohertySabesNonhuman2017 import OdohertySabesNonhuman2017
from .AllenVisualCodingOphys2016 import AllenVisualCodingOphys2016
from .KempSleepEDF2013 import KempSleepEDF2013
from .Neuroprobe2025 import Neuroprobe2025

_classes_electrophysiology = [
    "PerichMillerPopulation2018",
    "PeiPandarinathNLB2021",
    "FlintSlutzkyAccurate2012",
    "ChurchlandShenoyNeural2012",
    "OdohertySabesNonhuman2017",
]

_classes_calcium_imaging = [
    "AllenVisualCodingOphys2016",
]

_classes_ieeg = [
    "Neuroprobe2025",
]

_classes_psg = [
    "KempSleepEDF2013",
]

__all__ = [
    *_classes_electrophysiology,
    *_classes_calcium_imaging,
    *_classes_ieeg,
    *_classes_psg,
]
