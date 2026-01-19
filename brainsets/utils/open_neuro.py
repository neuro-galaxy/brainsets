"""OpenNeuro S3/API utilities for EEG data access.

This module provides functions to interact with OpenNeuro datasets:
- Dataset validation and metadata queries via GraphQL API
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- Electrode mapping utilities for MNE Raw objects
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import mne

OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
OPENNEURO_S3_BUCKET = "openneuro.org"

EEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf"}

MODALITY_TO_MNE_TYPE = {
    "EEG": "eeg",
    "SCALP": "eeg",
    "SCALP_EEG": "eeg",
    "EOG": "eog",
    "HEOG": "eog",
    "VEOG": "eog",
    "EMG": "emg",
    "MUSCLE": "emg",
    "ECG": "ecg",
    "EKG": "ecg",
    "CARDIAC": "ecg",
    "ECOG": "ecog",
    "SEEG": "seeg",
    "STEREO_EEG": "seeg",
    "RESP": "resp",
    "RESPIRATORY": "resp",
    "RESPIRATION": "resp",
    "BREATHING": "resp",
    "TMP": "temperature",
    "TEMP": "temperature",
    "THERM": "temperature",
    "TEMPERATURE": "temperature",
    "MEG": "meg",
    "MAG": "meg",
    "REF_MEG": "ref_meg",
    "MEG_REF": "ref_meg",
    "STIM": "stim",
    "STI": "stim",
    "EVENTS": "stim",
    "TRIGGER": "stim",
    "GSR": "gsr",
    "SKIN": "gsr",
    "GALVANIC": "gsr",
    "GALVANIC_SKIN_RESPONSE": "gsr",
    "BIO": "bio",
    "CHPI": "chpi",
    "DBS": "dbs",
    "DIPOLE": "dipole",
    "EXCI": "exci",
    "EYETRACK": "eyetrack",
    "FNIRS": "fnirs",
    "GOF": "gof",
    "GOODNESS_OF_FIT": "gof",
    "IAS": "ias",
    "SYST": "syst",
    "SYSTEM": "syst",
    "MISC": "misc",
}


@lru_cache(maxsize=1)
def _get_s3_client():
    """Get a cached S3 client with retry configuration for unsigned OpenNeuro access.

    Uses boto3's standard retry mode which includes:
    - Exponential backoff (base 2) with random jitter
    - Max backoff capped at ~20 seconds
    - Automatic retries on transient errors, throttling (429), and 5xx status codes
    """
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={
                "mode": "standard",
                "total_max_attempts": 3,
            },
        ),
    )


BIDS_EEG_PATTERN = re.compile(
    r"^(?P<subject>sub-[^_]+)"
    r"(?:_(?P<session>ses-[^_]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_eeg"
    r"\.(?P<ext>\w+)$"
)

METADATA_QUERY = """
    query Query($datasetId: ID!) {
        dataset(id: $datasetId) {
            created
            metadata {
                datasetId
                datasetName
                datasetUrl
                modalities
                species
                tasksCompleted
                studyDomain
                studyDesign
                openneuroPaperDOI
                associatedPaperDOI
                seniorAuthor
                adminUsers
            }
        }
    }
"""

README_QUERY = """
    query Query($datasetId: ID!) {
        dataset(id: $datasetId) {
            draft {
                readme
            }
        }
    }
"""

VERSION_TAGS_QUERY = """
    query Query($datasetId: ID!) {
        dataset(id: $datasetId) {
            snapshots {
                tag
            }
        }
    }
"""

PARTICIPANTS_QUERY = """
    query Query($datasetId: ID!, $tag: String!) {
        snapshot(datasetId: $datasetId, tag: $tag) {
            summary {
                subjects
            }
        }
    }
"""


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


def _format_error_from_response(response: requests.Response) -> str:
    """Extract an error message from an HTTP response."""
    try:
        root = ET.fromstring(response.text)
        return root.find("Message").text
    except Exception:
        return "Error message could not be formatted."


def _graphql_query(query: str, variables: dict) -> dict:
    """Execute a GraphQL query against the OpenNeuro API."""
    response = requests.post(
        OPENNEURO_GRAPHQL_URL, json={"query": query, "variables": variables}
    )

    if response.status_code == 200:
        if "errors" in response.json():
            raise RuntimeError(
                f"Error while fetching with query {query} and variables {variables}: "
                f"{response.json()['errors'][0]['message']}"
            )
        return response.json()["data"]

    raise RuntimeError(
        f"Error while fetching with query {query} and variables {variables}: "
        f"{response.status_code}: {_format_error_from_response(response)}"
    )


def fetch_metadata(dataset_id: str) -> dict:
    """Fetch metadata for a given OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., "ds000001")

    Returns:
        Dictionary containing metadata fields including datasetId, datasetName,
        datasetUrl, modalities, species, tasksCompleted, studyDomain, studyDesign,
        openneuroPaperDOI, associatedPaperDOI, seniorAuthor, adminUsers, and created
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(METADATA_QUERY, variables)

    result = raw_result["dataset"]["metadata"]
    result["created"] = raw_result["dataset"]["created"]
    return result


def fetch_readme(dataset_id: str) -> str:
    """Retrieve the README content for a given OpenNeuro dataset.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        The README content as a string
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(README_QUERY, variables)
    return raw_result["dataset"]["draft"]["readme"]


def fetch_latest_version_tag(dataset_id: str) -> str:
    """Get the latest version tag for a given OpenNeuro dataset.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')

    Returns:
        The latest version tag for the dataset (e.g., '1.0.0')
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(VERSION_TAGS_QUERY, variables)
    return raw_result["dataset"]["snapshots"][-1]["tag"]


def fetch_all_version_tags(dataset_id: str) -> list[str]:
    """Get all version tags for a given OpenNeuro dataset.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')

    Returns:
        List of version tags for the dataset (e.g., ['1.0.0', '1.0.1'])
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(VERSION_TAGS_QUERY, variables)
    return [tag["tag"] for tag in raw_result["dataset"]["snapshots"]]


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
    s3_client = _get_s3_client()

    prefix = f"{dataset_id}/"
    filenames = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=OPENNEURO_S3_BUCKET, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith("/"):
                    relative_path = key[len(prefix) :]
                    if relative_path:
                        filenames.append(relative_path)

    except Exception as e:
        raise RuntimeError(
            f"Error fetching filenames for dataset {dataset_id}: {str(e)}"
        )

    if len(filenames) == 0:
        raise RuntimeError(
            f"No files found for dataset {dataset_id}. "
            "The dataset may not exist or may be empty."
        )

    return filenames


def fetch_participants(dataset_id: str, tag: Optional[str] = None) -> list[str]:
    """Retrieve a list of participant IDs from an OpenNeuro dataset.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        tag: The dataset version tag to query (defaults to latest)

    Returns:
        List of participant IDs with 'sub-' prefix (e.g., ['sub-01', 'sub-02'])
    """
    dataset_id = validate_dataset_id(dataset_id)

    if tag is None:
        tag = fetch_latest_version_tag(dataset_id)

    variables = {"datasetId": dataset_id, "tag": tag}
    raw_result = _graphql_query(PARTICIPANTS_QUERY, variables)

    result = raw_result["snapshot"]["summary"]["subjects"]
    return [subj if subj.startswith("sub-") else f"sub-{subj}" for subj in result]


def parse_bids_eeg_filename(filename: str) -> Optional[dict]:
    """Parse a BIDS-compliant EEG filename to extract components.

    Args:
        filename: The filename to parse (e.g., 'sub-01_task-Sleep_eeg.edf')

    Returns:
        Dictionary with keys: subject_id, session_id, task_id, acq_id, run_id
        Returns None if the filename doesn't match BIDS EEG pattern
    """
    basename = Path(filename).name
    match = BIDS_EEG_PATTERN.match(basename)
    if not match:
        return None

    return {
        "subject_id": match.group("subject"),
        "session_id": match.group("session"),
        "task_id": match.group("task"),
        "acq_id": match.group("acq"),
        "run_id": match.group("run"),
    }


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


def construct_s3_url(dataset_id: str, recording_id: str) -> str:
    """Construct the S3 URL for a recording matching the given pattern.

    Args:
        dataset_id: OpenNeuro dataset identifier
        recording_id: Recording identifier to be used as a prefix pattern

    Returns:
        S3 URL for the recording files matching this prefix

    Raises:
        ValueError: If subject_id cannot be parsed from recording_id
        RuntimeError: If no files matching the recording_id pattern are found
    """
    dataset_id = validate_dataset_id(dataset_id)

    parts = recording_id.split("_")
    subject_id = None
    session_id = None

    for part in parts:
        if "sub-" in part:
            match = re.search(r"sub-([^_]+)", part)
            if match:
                subject_id = f"sub-{match.group(1)}"
        if "ses-" in part:
            match = re.search(r"ses-([^_]+)", part)
            if match:
                session_id = f"ses-{match.group(1)}"

    if not subject_id:
        raise ValueError(
            f"Could not parse subject_id from recording_id: {recording_id}"
        )

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    datatypes = [
        "eeg",
        "ieeg",
        "anat",
        "func",
        "meg",
        "fmap",
        "dwi",
        "beh",
        "perf",
        "pet",
        "micr",
        "nirs",
        "motion",
        "mrs",
    ]

    subject_prefix = f"{dataset_id}/{subject_id}/"

    if session_id:
        try:
            session_prefix = f"{subject_prefix}{session_id}/"
            for datatype in datatypes:
                search_prefix = f"{session_prefix}{datatype}/{recording_id}"
                try:
                    paginator = s3_client.get_paginator("list_objects_v2")
                    for page in paginator.paginate(
                        Bucket=OPENNEURO_S3_BUCKET, Prefix=search_prefix, MaxKeys=1
                    ):
                        if "Contents" in page and len(page["Contents"]) > 0:
                            base_path = f"s3://openneuro.org/{dataset_id}"
                            return f"{base_path}/{subject_id}/{session_id}/{datatype}/{recording_id}"
                except Exception:
                    continue
        except Exception:
            pass

    for datatype in datatypes:
        search_prefix = f"{subject_prefix}{datatype}/{recording_id}"
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=OPENNEURO_S3_BUCKET, Prefix=search_prefix, MaxKeys=1
            ):
                if "Contents" in page and len(page["Contents"]) > 0:
                    base_path = f"s3://openneuro.org/{dataset_id}"
                    return f"{base_path}/{subject_id}/{datatype}/{recording_id}"
        except Exception:
            continue

    session_info = f" and session {session_id}" if session_id else ""
    raise RuntimeError(
        f"No files found matching recording_id pattern '{recording_id}' "
        f"for subject {subject_id}{session_info} in dataset {dataset_id}"
    )


def get_s3_file_size(dataset_id: str, file_path: str) -> int:
    """Get the size of a file in an OpenNeuro dataset using S3 HEAD request.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        file_path: The relative path to the file within the dataset

    Returns:
        File size in bytes

    Raises:
        RuntimeError: If the file doesn't exist or cannot be accessed
    """
    dataset_id = validate_dataset_id(dataset_id)
    s3_client = _get_s3_client()
    s3_key = f"{dataset_id}/{file_path}"

    try:
        response = s3_client.head_object(Bucket=OPENNEURO_S3_BUCKET, Key=s3_key)
        return response["ContentLength"]
    except Exception as e:
        raise RuntimeError(
            f"Error getting file size for {file_path} in dataset {dataset_id}: {str(e)}"
        )


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


def download_file_from_s3(
    dataset_id: str, file_path: str, target_dir: str, force: bool = False
) -> str:
    """Download a single file from an OpenNeuro dataset using AWS S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        file_path: The relative path to the file within the dataset
        target_dir: Directory where the file will be downloaded
        force: If True, re-download even if file exists locally

    Returns:
        The local path to the downloaded file

    Raises:
        RuntimeError: If the file doesn't exist or cannot be downloaded
    """
    import os

    dataset_id = validate_dataset_id(dataset_id)
    local_path = os.path.join(target_dir, file_path)

    if os.path.exists(local_path) and not force:
        return local_path

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_key = f"{dataset_id}/{file_path}"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3_client.download_file(OPENNEURO_S3_BUCKET, s3_key, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Error downloading {file_path} from dataset {dataset_id}: {str(e)}"
        )


def download_prefix_from_s3(s3_url: str, target_dir: Path) -> None:
    """Download all files matching an S3 prefix pattern.

    The s3_url should be a prefix pattern like:
    's3://openneuro.org/ds005555/sub-1/eeg/sub-1_task-Sleep_acq-headband'

    This will download all files matching that prefix.

    Args:
        s3_url: S3 URL prefix pattern to match files
        target_dir: Local directory to download files to

    Raises:
        RuntimeError: If download fails
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URL: {s3_url}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = _get_s3_client()

    downloaded_files = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                obj_key = obj["Key"]
                if obj_key.endswith("/"):
                    continue

                dataset_id = prefix.split("/")[0]
                if obj_key.startswith(f"{dataset_id}/"):
                    rel_path = obj_key[len(f"{dataset_id}/") :]
                else:
                    continue

                local_path = target_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    s3.download_file(bucket, obj_key, str(local_path))
                    downloaded_files.append(local_path)
                except ClientError as e:
                    raise RuntimeError(f"Failed to download {obj_key}: {e}") from e

        if not downloaded_files:
            raise RuntimeError(
                f"No files found matching prefix pattern '{prefix}' in bucket '{bucket}'"
            )

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to download from {s3_url}: {e}")


def _modality_to_mne_type(modality: str) -> str:
    """Map modality string to MNE channel type."""
    return MODALITY_TO_MNE_TYPE.get((modality or "").upper(), "misc")


def rename_electrodes(raw: "mne.io.Raw", rename_map: dict[str, str]) -> None:
    """Rename channels in an MNE Raw object.

    Args:
        raw: The MNE Raw object to modify (modified in place)
        rename_map: Dictionary mapping old channel names to new names

    Example:
        >>> rename_map = {"PSG_F3": "F3", "PSG_F4": "F4"}
        >>> rename_electrodes(raw, rename_map)
    """
    if not rename_map:
        return

    current_ch_names = raw.ch_names
    valid_rename = {
        old: new for old, new in rename_map.items() if old in current_ch_names
    }

    if valid_rename:
        raw.rename_channels(valid_rename, allow_duplicates=False)


def set_channel_modalities(
    raw: "mne.io.Raw", modality_map: dict[str, list[str]]
) -> None:
    """Set channel types from a modality mapping.

    Args:
        raw: The MNE Raw object to modify (modified in place)
        modality_map: Dictionary mapping modality types to lists of channel names

    Example:
        >>> modality_map = {
        ...     "EEG": ["F3", "F4", "C3", "C4"],
        ...     "EOG": ["EOG_L", "EOG_R"],
        ... }
        >>> set_channel_modalities(raw, modality_map)
    """
    if not modality_map:
        return

    current_ch_names = set(raw.ch_names)
    type_dict = {}

    for modality, channel_names in modality_map.items():
        mne_type = _modality_to_mne_type(modality)
        for ch_name in channel_names:
            if ch_name in current_ch_names:
                type_dict[ch_name] = mne_type

    if type_dict:
        raw.set_channel_types(type_dict)
