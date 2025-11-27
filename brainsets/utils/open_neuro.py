import os
import logging
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from typing import Union

OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
OPENNEURO_S3_BUCKET = "openneuro.org"


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
    """Returns a valid dataset ID or throws an error if it's invalid.

    Args:
        dataset_id: The dataset identifier to validate (e.g., "5085", "ds5085", "ds005085")

    Returns:
        The validated and formatted dataset ID (e.g., "ds005085")
    """
    # Remove any whitespace
    dataset_id = dataset_id.strip()

    # If it already starts with 'ds', extract and validate the numeric part
    if dataset_id.lower().startswith("ds"):
        numeric_part = dataset_id[2:]
        if numeric_part.isdigit():
            # Must be exactly 4 digits when converted to int (no leading zeros count toward length)
            num = int(numeric_part)
            if num <= 9999:  # 4 digits max
                return f"ds{num:06d}"
            else:
                raise ValueError(
                    f"Dataset ID '{dataset_id}' has too many digits. Maximum is 4 digits after 'ds'."
                )
        else:
            raise ValueError(
                f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by digits only."
            )

    # If it's just numbers, validate and format
    if dataset_id.isdigit():
        num = int(dataset_id)
        if num <= 9999:  # 4 digits max
            return f"ds{num:06d}"
        else:
            raise ValueError(
                f"Dataset ID '{dataset_id}' has too many digits. Maximum is 4 digits."
            )

    # If we can't parse it, raise an error
    raise ValueError(
        f"Invalid dataset ID format: '{dataset_id}'. Expected numeric ID or 'ds' + digits."
    )


def _format_error_from_response(response: requests.Response) -> str:
    """Extract an error message from an HTTP response.

    Args:
        response: The HTTP response object to extract the error message from

    Returns:
        The extracted error message, or a default message if extraction fails
    """
    try:
        root = ET.fromstring(response.text)
        return root.find("Message").text
    except Exception as e:
        return "Error message could not be formatted."


def _graphql_query(query: str, variables: dict) -> requests.Response:
    """Query the OpenNeuro GraphQL API.

    Args:
        query: The GraphQL query to execute
        variables: The variables to pass to the query

    Returns:
        The response from the OpenNeuro GraphQL API
    """

    response = requests.post(
        OPENNEURO_GRAPHQL_URL, json={"query": query, "variables": variables}
    )

    if response.status_code == 200:
        if "errors" in response.json():
            raise RuntimeError(
                f"Error while fetching with query {query} and variables {variables}: {response.json()['errors'][0]['message']}"
            )
        return response.json()["data"]
    else:
        raise RuntimeError(
            f"Error while fetching with query {query} and variables {variables}: {response.status_code}: {_format_error_from_response(response)}"
        )


def fetch_metadata(dataset_id: str) -> dict:
    """Fetch metadata for a given OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., "ds000001")

    Returns:
        Dictionary containing metadata fields including datasetId, datasetName, datasetUrl,
        modalities, species, tasksCompleted, studyDomain, studyDesign, openneuroPaperDOI,
        associatedPaperDOI, seniorAuthor, adminUsers, and created
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(METADATA_QUERY, variables)

    # Extract the metadata from the response
    result = raw_result["dataset"]["metadata"]
    result["created"] = raw_result["dataset"]["created"]
    return result


def fetch_readme(dataset_id: str) -> str:
    """Retrieve the README content for a given OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        The README content as a string
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(README_QUERY, variables)

    # Extract the README content from the response
    return raw_result["dataset"]["draft"]["readme"]


def fetch_latest_version_tag(dataset_id: str) -> str:
    """Get the latest version tag for a given OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')

    Returns:
        The latest version tag for the dataset (e.g., '1.0.0')
    """
    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(VERSION_TAGS_QUERY, variables)

    # Extract the latest version tag from the response
    return raw_result["dataset"]["snapshots"][-1]["tag"]


def fetch_all_version_tags(dataset_id: str) -> list[str]:
    """Get all version tags for a given OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')

    Returns:
        List of version tags for the dataset (e.g., ['1.0.0', '1.0.1'])
    """

    dataset_id = validate_dataset_id(dataset_id)
    variables = {"datasetId": dataset_id}
    raw_result = _graphql_query(VERSION_TAGS_QUERY, variables)

    # Extract the version tags from the response
    result = raw_result["dataset"]["snapshots"]
    return [tag["tag"] for tag in result]


def fetch_all_filenames(dataset_id: str, tag: str = None) -> list[str]:
    """Fetch the filenames for a given OpenNeuro dataset using AWS S3.

    Note: The S3 bucket only contains the latest version of each dataset,
    so the tag parameter is currently ignored. All files from the dataset
    root are returned regardless of the tag value.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        tag: The dataset version tag (currently ignored for S3 access)

    Returns:
        List of relative filenames in the dataset (excluding directories)
    """

    dataset_id = validate_dataset_id(dataset_id)

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

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


def _fetch_all_filenames_latest_tag(dataset_id: str) -> list[str]:
    """Fetch the filenames for the latest version of a given OpenNeuro dataset.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        List of filenames in the latest dataset version
    """

    return fetch_all_filenames(dataset_id, tag=None)


def fetch_participants(dataset_id: str, tag: str = None) -> list[str]:
    """Retrieve a list of participant IDs from an OpenNeuro dataset using the GraphQL API.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        tag: The dataset version tag to query (e.g., '1.0.0') (optional)

    Returns:
        List of participant IDs with 'sub-' prefix (e.g., ['sub-01', 'sub-02'])
    """

    dataset_id = validate_dataset_id(dataset_id)

    if tag is None:
        tag = fetch_latest_version_tag(dataset_id)

    variables = {"datasetId": dataset_id, "tag": tag}
    raw_result = _graphql_query(PARTICIPANTS_QUERY, variables)

    # Extract the participant IDs from the response
    result = raw_result["snapshot"]["summary"]["subjects"]
    subjects = [subj if subj.startswith("sub-") else f"sub-{subj}" for subj in result]

    return subjects


def get_s3_file_size(dataset_id: str, file_path: str) -> int:
    """Get the size of a file in an OpenNeuro dataset using S3 HEAD request.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        file_path: The relative path to the file within the dataset (e.g., 'dataset_description.json')

    Returns:
        File size in bytes

    Raises:
        RuntimeError: If the file doesn't exist or cannot be accessed
    """
    dataset_id = validate_dataset_id(dataset_id)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    s3_key = f"{dataset_id}/{file_path}"

    try:
        response = s3_client.head_object(Bucket=OPENNEURO_S3_BUCKET, Key=s3_key)
        return response["ContentLength"]
    except Exception as e:
        raise RuntimeError(
            f"Error getting file size for {file_path} in dataset {dataset_id}: {str(e)}"
        )


def download_file_from_s3(
    dataset_id: str, file_path: str, target_dir: str, force: bool = False
) -> str:
    """Download a single file from an OpenNeuro dataset using AWS S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds000001')
        file_path: The relative path to the file within the dataset (e.g., 'dataset_description.json')
        target_dir: Directory where the file will be downloaded
        force: If True, re-download even if file exists locally (default: False)

    Returns:
        The local path to the downloaded file

    Raises:
        RuntimeError: If the file doesn't exist or cannot be downloaded
    """
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


def download_subject_eeg_data(
    dataset_id: str,
    subject_id: list[str],
    target_dir: str,
    path_signature: str = "*/eeg/*",
    tag: str = None,
) -> None:
    """Download data for specified subjects from an OpenNeuro dataset.

    Args:
        dataset_id: The name of the dataset on OpenNeuro (e.g., 'ds000247')
        participants_list: List of all participants in the dataset
        subject_id: List of specific subject IDs to include in the download
        target_dir: Directory where the dataset should be downloaded
        path_signature: BIDS structure of the dataset, used to only download the eeg files (e.g., '*/eeg/*')
        tag: Version tag of the dataset to download (e.g., '0.0.1')
    """

    if tag is None:
        tag = fetch_latest_version_tag(dataset_id)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    participants_list = fetch_participants(dataset_id, tag)

    # only include subjects that are not in target_dir yet
    subjects_to_include = [
        subj_id
        for subj_id in subject_id
        if not os.path.exists(os.path.join(target_dir, subj_id))
    ]

    # exclude all but subj_ids in include
    exclude = np.setdiff1d(participants_list, subjects_to_include)

    if subjects_to_include:
        include = [
            path_signature.replace("sub-*", subj_id) for subj_id in subjects_to_include
        ]
    else:
        return

    try:
        download(
            dataset=dataset_id,
            tag=tag,
            target_dir=target_dir,
            include=include,
            exclude=exclude,
            max_concurrent_downloads=5,
            max_retries=3,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to download dataset {dataset_id} (version {tag}): {str(e)} Stopping execution."
        )


def download_openneuro_data(
    dataset_id: str,
    subject_id: Union[list[str]],
    target_dir: str,
    bids_structure: Union[dict, None],
    logger: logging.Logger = None,
) -> dict:
    """Download data from an OpenNeuro dataset for specified subjects.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., "ds003645")
        subject_id: List of subject IDs to download (e.g., ["sub-01", "sub-02"])
        target_dir: Directory where the dataset will be downloaded
        bids_structure: Dictionary specifying BIDS path signatures for the dataset, or None
        logger: Logger instance for logging messages (optional)

    Returns:
        Dictionary containing metadata, readme, latest_version_tag, participants_list,
        and valid_subject_id
    """

    # validate dataset id
    dataset_id = validate_dataset_id(dataset_id)

    # fetch metadata
    metadata = fetch_metadata(dataset_id)
    readme = fetch_readme(dataset_id)

    # validate data modality
    if not any(
        "EEG" in modality.upper() for modality in metadata["modalities"].values[0]
    ):
        raise RuntimeError(
            f"Dataset {dataset_id} does not contain EEG data. Stopping execution."
        )

    # get latest version tag
    latest_version_tag = fetch_latest_version_tag(dataset_id)

    # validate subject id
    participants_list = fetch_participants(
        dataset_id,
        latest_version_tag,
    )

    # validate subject id
    if subject_id is None:
        raise RuntimeError(
            f"Subject ID must be provided. Subject ID list is None. Stopping execution."
        )

    # download data
    download_subject_eeg_data(
        dataset_id=dataset_id,
        subject_id=subject_id,
        target_dir=target_dir,
        path_signature=bids_structure[dataset_id],
        tag=latest_version_tag,
    )

    result = {
        "metadata": metadata,
        "readme": readme,
        "latest_version_tag": latest_version_tag,
        "participants_list": participants_list,
        "valid_subject_id": subject_id,
    }

    return result
