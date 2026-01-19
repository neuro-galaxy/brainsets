# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.0.0",
#   "boto3~=1.26.0",
#   "requests~=2.28.0",
# ]
# ///

"""Klinzing Sleep dataset pipeline (OpenNeuro ds005555).

This pipeline processes EEG sleep recordings from the Klinzing et al. dataset.
Dataset URL: https://openneuro.org/datasets/ds005555
"""

from brainsets.utils.openneuro import OpenNeuroEEGPipeline

ELECTRODE_RENAME = {
    "PSG_F3": "F3",
    "PSG_F4": "F4",
    "PSG_C3": "C3",
    "PSG_C4": "C4",
    "PSG_O1": "O1",
    "PSG_O2": "O2",
    "PSG_EOG": "EOG",
    "PSG_EMG": "EMG",
    "PSG_PULSE": "PULSE",
    "PSG_BEAT": "BEAT",
    "PSG_SPO2": "SPO2",
}

MODALITY_CHANNELS = {
    "EEG": ["F3", "F4", "C3", "C4", "O1", "O2"],
    "EOG": ["EOG"],
    "EMG": ["EMG"],
    "PULSE": ["PULSE"],
    "BEAT": ["BEAT"],
    "SPO2": ["SPO2"],
}


class Pipeline(OpenNeuroEEGPipeline):
    """Pipeline for processing Klinzing Sleep EEG dataset."""

    brainset_id = "klinzing_sleep_ds005555_2024"
    dataset_id = "ds005555"
    description = "Sleep EEG recordings from Klinzing et al."

    ELECTRODE_RENAME = ELECTRODE_RENAME
    MODALITY_CHANNELS = MODALITY_CHANNELS
