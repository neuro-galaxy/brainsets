"""OpenNeuro dataset utilities.

This module provides functions for dataset validation, file listing,
and downloading from OpenNeuro's S3 bucket.
"""

from pathlib import Path
from typing import Optional

from brainsets.utils.bids_utils import EEG_EXTENSIONS, parse_bids_eeg_filename
from brainsets.utils.s3_utils import (
    download_prefix_from_url,
    get_cached_s3_client,
    list_objects,
)

OPENNEURO_S3_BUCKET = "openneuro.org"


def validate_dataset_id(dataset_id: str) -> str:
    """Validate and normalize an OpenNeuro dataset ID.

    OpenNeuro dataset IDs follow the format 'ds' followed by exactly 6 digits,
    where the numeric portion ranges from 000001 to 009999.

    Args:
        dataset_id: The dataset identifier in various accepted formats:
            - Numeric only: "5085" -> "ds005085"
            - With prefix: "ds5085" -> "ds005085"
            - Already normalized: "ds005085" -> "ds005085"

    Returns:
        Normalized dataset ID in format "dsXXXXXX" (6 digits, zero-padded)

    Raises:
        ValueError: If the dataset ID format is invalid or numeric part exceeds 9999
    """
    dataset_id = dataset_id.strip()

    if dataset_id.lower().startswith("ds"):
        numeric_part = dataset_id[2:]
        if numeric_part.isdigit():
            num = int(numeric_part)
            if num <= 9999:
                return f"ds{num:06d}"
            raise ValueError(
                f"Dataset ID '{dataset_id}' has too many digits. Maximum is 4 digits after 'ds'."
            )
        raise ValueError(
            f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by digits only."
        )

    if dataset_id.isdigit():
        num = int(dataset_id)
        if num <= 9999:
            return f"ds{num:06d}"
        raise ValueError(
            f"Dataset ID '{dataset_id}' has too many digits. Maximum is 4 digits."
        )

    raise ValueError(
        f"Invalid dataset ID format: '{dataset_id}'. Expected numeric ID or 'ds' + digits."
    )


def fetch_all_filenames(dataset_id: str, tag: Optional[str] = None) -> list[str]:
    """Fetch all filenames for a given OpenNeuro dataset using AWS S3.

    Note: The S3 bucket only contains the latest version of each dataset,
    so the tag parameter is currently ignored.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        tag: The dataset version tag (currently ignored for S3 access)

    Returns:
        List of relative filenames in the dataset (excluding directories)
    """
    dataset_id = validate_dataset_id(dataset_id)
    prefix = f"{dataset_id}/"

    filenames = list_objects(OPENNEURO_S3_BUCKET, prefix)

    if len(filenames) == 0:
        raise RuntimeError(
            f"No files found for dataset {dataset_id}. "
            "The dataset may not exist or may be empty."
        )

    return filenames


def fetch_eeg_recordings(dataset_id: str) -> list[dict]:
    """Discover all EEG recordings in a dataset by parsing BIDS filenames.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        List of dicts with keys:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acq_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - eeg_file: Relative path to EEG file
    """
    dataset_id = validate_dataset_id(dataset_id)
    all_files = fetch_all_filenames(dataset_id)

    recordings = []
    seen_recording_ids = set()

    for filepath in all_files:
        ext = Path(filepath).suffix.lower()
        if ext not in EEG_EXTENSIONS:
            continue

        parsed = parse_bids_eeg_filename(filepath)
        if not parsed:
            continue

        components = [parsed["subject_id"]]
        if parsed["session_id"]:
            components.append(parsed["session_id"])
        components.append(f"task-{parsed['task_id']}")
        if parsed["acq_id"]:
            components.append(f"acq-{parsed['acq_id']}")
        if parsed["run_id"]:
            components.append(f"run-{parsed['run_id']}")

        recording_id = "_".join(components)

        if recording_id in seen_recording_ids:
            continue
        seen_recording_ids.add(recording_id)

        recordings.append(
            {
                "recording_id": recording_id,
                "subject_id": parsed["subject_id"],
                "session_id": parsed["session_id"],
                "task_id": parsed["task_id"],
                "acq_id": parsed["acq_id"],
                "run_id": parsed["run_id"],
                "eeg_file": filepath,
            }
        )

    return recordings


def construct_s3_url_from_path(dataset_id: str, eeg_file_path: str) -> str:
    """Construct S3 URL directly from a known file path.

    This avoids S3 API calls by using the path that was already discovered.

    Args:
        dataset_id: OpenNeuro dataset identifier
        eeg_file_path: Relative path to the EEG file within the dataset

    Returns:
        S3 URL prefix for downloading the recording files
    """
    dataset_id = validate_dataset_id(dataset_id)
    parent_dir = str(Path(eeg_file_path).parent)
    recording_id = Path(eeg_file_path).stem.replace("_eeg", "")
    return f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"


def download_recording(s3_url: str, target_dir: Path) -> list[Path]:
    """Download all files matching an S3 prefix pattern for a recording.

    Args:
        s3_url: S3 URL prefix pattern (e.g., 's3://openneuro.org/ds005555/sub-1/eeg/sub-1_task-Sleep')
        target_dir: Local directory to download files to

    Returns:
        List of downloaded file paths

    Raises:
        RuntimeError: If download fails
    """
    return download_prefix_from_url(s3_url, target_dir)


def check_recording_files_exist(recording_id: str, subject_dir: Path) -> bool:
    """Check if EEG files matching the recording_id pattern exist locally.

    Args:
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
        subject_dir: Subject directory to search in

    Returns:
        True if at least one recording file is found, False otherwise
    """
    if not subject_dir.exists():
        return False

    eeg_patterns = [
        f"**/{recording_id}_eeg.edf",
        f"**/{recording_id}_eeg.fif",
        f"**/{recording_id}_eeg.set",
        f"**/{recording_id}_eeg.bdf",
        f"**/{recording_id}_eeg.vhdr",
        f"**/{recording_id}_eeg.eeg",
    ]

    for pattern in eeg_patterns:
        if list(subject_dir.glob(pattern)):
            return True

    return False
