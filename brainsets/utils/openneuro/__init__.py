"""OpenNeuro utilities subpackage.

This package provides utilities for working with OpenNeuro datasets:
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- OpenNeuroPipeline base class and EEG/iEEG subclasses for building pipelines
"""

from .openneuro_s3 import (
    OPENNEURO_S3_BUCKET,
    construct_s3_url_from_path,
    download_dataset_description,
    download_recording,
    fetch_all_filenames,
    fetch_participants_tsv,
    fetch_species,
    validate_dataset_id,
    validate_dataset_version,
    validate_subject_ids,
)
from .pipeline import OpenNeuroEEGPipeline, OpenNeuroIEEGPipeline, OpenNeuroPipeline

__all__ = [
    "OPENNEURO_S3_BUCKET",
    "construct_s3_url_from_path",
    "download_dataset_description",
    "download_recording",
    "fetch_all_filenames",
    "fetch_participants_tsv",
    "fetch_species",
    "validate_dataset_id",
    "validate_dataset_version",
    "validate_subject_ids",
    "OpenNeuroPipeline",
    "OpenNeuroEEGPipeline",
    "OpenNeuroIEEGPipeline",
]
