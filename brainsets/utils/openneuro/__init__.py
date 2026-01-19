"""OpenNeuro utilities subpackage.

This package provides utilities for working with OpenNeuro datasets:
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- Data extraction utilities for MNE Raw objects
- OpenNeuroEEGPipeline base class for building EEG pipelines
"""

from .data_extraction import (
    extract_brainset_description,
    extract_channels,
    extract_device_description,
    extract_meas_date,
    extract_session_description,
    extract_signal,
    extract_subject_description,
    generate_train_valid_splits_one_epoch,
)
from .dataset import (
    OPENNEURO_S3_BUCKET,
    check_recording_files_exist,
    construct_s3_url_from_path,
    download_recording,
    fetch_all_filenames,
    fetch_eeg_recordings,
    validate_dataset_id,
)
from .pipeline import OpenNeuroEEGPipeline

__all__ = [
    # dataset.py
    "OPENNEURO_S3_BUCKET",
    "validate_dataset_id",
    "fetch_all_filenames",
    "fetch_eeg_recordings",
    "construct_s3_url_from_path",
    "download_recording",
    "check_recording_files_exist",
    # data_extraction.py
    "extract_brainset_description",
    "extract_subject_description",
    "extract_session_description",
    "extract_device_description",
    "extract_meas_date",
    "extract_signal",
    "extract_channels",
    "generate_train_valid_splits_one_epoch",
    # pipeline.py
    "OpenNeuroEEGPipeline",
]
