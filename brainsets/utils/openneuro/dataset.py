"""OpenNeuro dataset utilities.

This module provides functions for dataset validation, file listing,
and downloading from OpenNeuro's S3 bucket.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from botocore.exceptions import ClientError

    BOTO_AVAILABLE = True
except ImportError:
    ClientError = None
    BOTO_AVAILABLE = False

from brainsets.utils.bids_utils import (
    EEG_EXTENSIONS,
    IEEG_EXTENSIONS,
    parse_bids_eeg_filename,
    parse_bids_ieeg_filename,
)
from brainsets.utils.s3_utils import (
    download_prefix_from_url,
    get_cached_s3_client,
    get_object_list,
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

    filenames = get_object_list(OPENNEURO_S3_BUCKET, prefix)

    if len(filenames) == 0:
        raise RuntimeError(
            f"No files found for dataset {dataset_id}. "
            "The dataset may not exist or may be empty."
        )

    return filenames


def _fetch_recordings(
    dataset_id: str,
    extensions: set[str],
    parse_fn,
    data_file_key: str,
) -> list[dict]:
    """Discover recordings in a dataset by parsing BIDS filenames.

    Internal helper function used by fetch_eeg_recordings and fetch_ieeg_recordings.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        extensions: Set of file extensions to accept (e.g., EEG_EXTENSIONS)
        parse_fn: Function to parse BIDS filename (e.g., parse_bids_eeg_filename)
        data_file_key: Key name for data file path in returned dict (e.g., 'eeg_file')

    Returns:
        List of dicts with recording metadata
    """
    dataset_id = validate_dataset_id(dataset_id)
    all_files = fetch_all_filenames(dataset_id)

    recordings = []
    seen_recording_ids = set()

    for filepath in all_files:
        ext = Path(filepath).suffix.lower()
        if ext not in extensions:
            continue

        parsed = parse_fn(filepath)
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
                data_file_key: filepath,
            }
        )

    return recordings


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
    return _fetch_recordings(
        dataset_id, EEG_EXTENSIONS, parse_bids_eeg_filename, "eeg_file"
    )


def fetch_ieeg_recordings(dataset_id: str) -> list[dict]:
    """Discover all iEEG recordings in a dataset by parsing BIDS filenames.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        List of dicts with keys:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-VisualNaming')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'VisualNaming')
            - acq_id: Acquisition identifier or None (e.g., 'ecog')
            - run_id: Run identifier or None (e.g., '01')
            - ieeg_file: Relative path to iEEG file
    """
    return _fetch_recordings(
        dataset_id, IEEG_EXTENSIONS, parse_bids_ieeg_filename, "ieeg_file"
    )


def fetch_participants_tsv(dataset_id: str) -> Optional[pd.DataFrame]:
    """Fetch and parse participants.tsv from OpenNeuro S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        DataFrame with participant information, or None if file doesn't exist.
        The DataFrame has 'participant_id' as the index and columns like 'age', 'sex', etc.
    """
    dataset_id = validate_dataset_id(dataset_id)
    s3_client = get_cached_s3_client()

    key = f"{dataset_id}/participants.tsv"

    try:
        response = s3_client.get_object(Bucket=OPENNEURO_S3_BUCKET, Key=key)
        content = response["Body"].read()

        df = pd.read_csv(
            BytesIO(content),
            sep="\t",
            na_values=["n/a", "N/A"],
            keep_default_na=True,
        )

        if "participant_id" not in df.columns:
            return None

        df = df.set_index("participant_id")
        return df

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404"):
            return None
        raise


def construct_s3_url_from_path(dataset_id: str, data_file_path: str) -> str:
    """Construct S3 URL directly from a known file path.

    This avoids S3 API calls by using the path that was already discovered.
    Automatically detects and strips both _eeg and _ieeg suffixes.

    Args:
        dataset_id: OpenNeuro dataset identifier
        data_file_path: Relative path to the EEG/iEEG file within the dataset

    Returns:
        S3 URL prefix for downloading the recording files
    """
    dataset_id = validate_dataset_id(dataset_id)
    parent_dir = str(Path(data_file_path).parent)
    stem = Path(data_file_path).stem
    if stem.endswith("_eeg"):
        recording_id = stem[:-4]  # Remove "_eeg"
    elif stem.endswith("_ieeg"):
        recording_id = stem[:-5]  # Remove "_ieeg"
    else:
        recording_id = stem
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


def download_dataset_description(dataset_id: str, target_dir: Path) -> Path:
    """Download dataset_description.json from OpenNeuro S3.

    This file is required for mne-bids to recognize a valid BIDS dataset.
    If the file already exists locally, it is not re-downloaded.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        target_dir: Local directory to download to

    Returns:
        Path to the downloaded or existing dataset_description.json file

    Raises:
        RuntimeError: If download fails or file doesn't exist on S3
    """
    dataset_id = validate_dataset_id(dataset_id)
    target_dir = Path(target_dir)
    target_path = target_dir / "dataset_description.json"

    if target_path.exists():
        return target_path

    s3_client = get_cached_s3_client()
    key = f"{dataset_id}/dataset_description.json"

    try:
        response = s3_client.get_object(Bucket=OPENNEURO_S3_BUCKET, Key=key)
        content = response["Body"].read()

        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(content)

        return target_path

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404"):
            raise RuntimeError(
                f"dataset_description.json not found for {dataset_id} on OpenNeuro S3"
            ) from e
        raise RuntimeError(
            f"Failed to download dataset_description.json for {dataset_id}: {e}"
        ) from e


def check_recording_files_exist(recording_id: str, subject_dir: Path) -> bool:
    """Check if data files matching the recording_id pattern exist locally.

    Searches for any data file (EEG or iEEG) in the BIDS-structured directory.
    Supports all BIDS-compliant formats (.edf, .vhdr, .set, .bdf, .eeg, .nwb)
    plus .fif for MNE-processed files.

    Args:
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
        subject_dir: Subject directory to search in

    Returns:
        True if at least one data file is found, False otherwise
    """
    if not subject_dir.exists():
        return False

    supported_extensions = {".edf", ".set", ".bdf", ".vhdr", ".eeg", ".nwb", ".fif"}

    for file in subject_dir.rglob(f"{recording_id}_*"):
        if file.suffix.lower() in supported_extensions:
            return True

    return False
