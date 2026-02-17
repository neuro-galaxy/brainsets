"""OpenNeuro utilities subpackage.

This package provides utilities for working with OpenNeuro datasets:
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- Data extraction utilities for MNE Raw objects
- OpenNeuroEEGPipeline base class for building EEG pipelines
"""

from .data_extraction import (
    extract_brainset_description,
    extract_device_description,
    extract_session_description,
    extract_subject_description,
)
from .dataset import (
    OPENNEURO_S3_BUCKET,
    check_recording_files_exist,
    construct_s3_url_from_path,
    download_dataset_description,
    download_recording,
    fetch_all_filenames,
    fetch_eeg_recordings,
    fetch_ieeg_recordings,
    fetch_participants_tsv,
    validate_dataset_id,
)
from .pipeline import OpenNeuroEEGPipeline, OpenNeuroIEEGPipeline, OpenNeuroPipeline
from brainsets.utils.split import generate_train_valid_splits_one_epoch

__all__ = [
    # dataset.py
    "OPENNEURO_S3_BUCKET",
    "validate_dataset_id",
    "fetch_all_filenames",
    "fetch_eeg_recordings",
    "fetch_ieeg_recordings",
    "fetch_participants_tsv",
    "construct_s3_url_from_path",
    "download_recording",
    "download_dataset_description",
    "check_recording_files_exist",
    # data_extraction.py
    "extract_brainset_description",
    "extract_subject_description",
    "extract_session_description",
    "extract_device_description",
    # split.py
    "generate_train_valid_splits_one_epoch",
    # pipeline.py
    "OpenNeuroPipeline",
    "OpenNeuroEEGPipeline",
    "OpenNeuroIEEGPipeline",
]
