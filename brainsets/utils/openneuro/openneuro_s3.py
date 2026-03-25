"""OpenNeuro dataset utilities.

This module provides functions for dataset validation, file listing,
and downloading from OpenNeuro's S3 bucket.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional
import logging
import requests
import pandas as pd

try:
    from botocore.exceptions import ClientError

    BOTO_AVAILABLE = True
except ImportError:
    ClientError = Exception
    BOTO_AVAILABLE = False

from brainsets.utils.s3_utils import (
    download_prefix_from_url,
    get_cached_s3_client,
    get_object_list,
)

OPENNEURO_S3_BUCKET = "openneuro.org"
GRAPHQL_ENDPOINT = "https://openneuro.org/crn/graphql"


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


def validate_dataset_version(dataset_id: str, dataset_version: str) -> str:
    """
    Validate and normalize a dataset version against OpenNeuro.

    This function checks the provided dataset version (`dataset_version`) for the given
    OpenNeuro dataset (`dataset_id`) by querying the OpenNeuro GraphQL API for the latest
    available version (snapshot tag). If the versions do not match, a warning is emitted
    that highlights a potential version discrepancy. Regardless, the latest available
    snapshot tag is always returned (current OpenNeuro S3 only contains latest).

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds005555').
        dataset_version: The version (snapshot tag) of the dataset (e.g., '1.0.0') as
            used during pipeline creation.

    Returns:
        The latest snapshot tag/name available on OpenNeuro for the given dataset.

    Side Effects:
        If `dataset_version` does not match the latest version, logs a warning explaining
        possible reproducibility and consistency issues.
    """
    query = """
        query Dataset($datasetId: ID!) {
            dataset(id: $datasetId) {
                latestSnapshot {
                    tag
                }
            }
        }
    """

    variables = {
        "datasetId": dataset_id,
    }

    response = _graphql_query_openneuro(
        query,
        variables,
    )

    latest_snapshot_tag = response["data"]["dataset"]["latestSnapshot"]["tag"]
    if latest_snapshot_tag != dataset_version:
        logging.warning(
            f"Dataset version '{dataset_version}' was used to create the brainset pipeline for dataset '{dataset_id}', "
            f"but the latest available version on OpenNeuro is '{latest_snapshot_tag}'. "
            f"Downloading data or running the pipeline now will use the latest version, "
            f"which may differ from the original version used, potentially causing errors or inconsistencies. "
            f"Check the CHANGES file of the dataset for details about the differences between versions."
        )
    return latest_snapshot_tag


def fetch_all_filenames(dataset_id: str) -> list[str]:
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
            logging.warning(
                f"No participant_id column found in participants.tsv file in OpenNeuro dataset {dataset_id}. "
                "Returning None."
            )
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


def _graphql_query_openneuro(query: str, variables: dict | None = None) -> dict:
    """
    Execute a GraphQL query and return the response.

    Args:
        query: The GraphQL query to execute
        variables: The variables to pass to the query

    Returns:
        The response from the GraphQL query
    """

    def _retry(max_attempts=5, initial_wait=4, max_wait=10):
        def decorator(func):
            import time
            import random

            def wrapper(*args, **kwargs):
                attempt = 0
                wait_time = initial_wait
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, max_wait)

            return wrapper

        return decorator

    @_retry(max_attempts=5, initial_wait=4, max_wait=10)
    def _graphql_query(query, variables=None):
        response = requests.post(
            GRAPHQL_ENDPOINT, json={"query": query, "variables": variables}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with status code {response.status_code}")

    return _graphql_query(query, variables)
