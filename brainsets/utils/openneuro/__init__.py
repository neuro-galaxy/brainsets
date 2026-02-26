"""OpenNeuro utilities subpackage.

This package provides utilities for working with OpenNeuro datasets:
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- OpenNeuroPipeline base class and EEG/iEEG subclasses for building pipelines
"""

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

__all__ = [
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
    "OpenNeuroPipeline",
    "OpenNeuroEEGPipeline",
    "OpenNeuroIEEGPipeline",
]
