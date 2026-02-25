# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from brainsets.utils.openneuro import OpenNeuroIEEGPipeline


class Pipeline(OpenNeuroIEEGPipeline):
    brainset_id = "kochi_visualnaming_ds006914_2025"
    dataset_id = "ds006914"
    description = (
        "Visual Naming EC - A large-scale intracranial EEG (iEEG) dataset with 110 subjects "
        "and 353 recordings from ECoG electrodes during picture naming task. "
        "Includes comprehensive electrode localization (MNI coordinates) and channel metadata "
        "from BIDS sidecar files."
    )
