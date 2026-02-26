# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from brainsets.utils.openneuro import OpenNeuroEEGPipeline

MODALITY_CHANNELS = {"EEG": [f"E{i}" for i in range(1, 129)] + ["Cz"]}


class Pipeline(OpenNeuroEEGPipeline):
    brainset_id = "shirazi_hbnr1_ds005505_2024"
    dataset_id = "ds005505"
    description = (
        "Healthy Brain Network (HBN) EEG dataset containing recordings from "
        "participants performing various passive and active tasks including "
        "resting state, movie watching, and cognitive tasks."
    )
    MODALITY_CHANNELS = MODALITY_CHANNELS
