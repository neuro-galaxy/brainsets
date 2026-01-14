"""Klinzing Sleep dataset pipeline (OpenNeuro ds005555).

This pipeline processes EEG sleep recordings from the Klinzing et al. dataset.
Dataset URL: https://openneuro.org/datasets/ds005555
"""

from brainsets.utils.open_neuro_pipeline import OpenNeuroEEGPipeline

ELECTRODE_RENAME = {
    "PSG F3": "F3",
    "PSG F4": "F4",
    "PSG C3": "C3",
    "PSG C4": "C4",
    "PSG O1": "O1",
    "PSG O2": "O2",
    "PSG EOG1": "EOG_L",
    "PSG EOG2": "EOG_R",
    "PSG EMG1": "EMG_Chin",
}

MODALITY_CHANNELS = {
    "EEG": ["F3", "F4", "C3", "C4", "O1", "O2"],
    "EOG": ["EOG_L", "EOG_R"],
    "EMG": ["EMG_Chin"],
}


class Pipeline(OpenNeuroEEGPipeline):
    """Pipeline for processing Klinzing Sleep EEG dataset."""

    brainset_id = "klinzing_sleep_ds005555_2024"
    dataset_id = "ds005555"
    description = "Sleep EEG recordings from Klinzing et al."

    ELECTRODE_RENAME = ELECTRODE_RENAME
    MODALITY_CHANNELS = MODALITY_CHANNELS

    def process(self, download_output):
        """Process the downloaded EEG data using default pipeline."""
        super().process(download_output)
