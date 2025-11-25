from pathlib import Path
import pandas as pd
import os
from functools import lru_cache
import logging
from brainsets.utils.open_neuro import validate_dataset_id

CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "context")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_default_dataset_info() -> pd.DataFrame:
    """Cache the dataset info CSV to avoid repeated disk reads."""
    logger.info("Loading default_dataset_info.csv...")
    df = pd.read_csv(os.path.join(CONTEXT_PATH, "default_dataset_info.csv"))
    logger.info(f"Loaded {len(df)} records from default_dataset_info.csv")
    return df


def task_taxonomy() -> list[dict]:
    df = pd.read_csv(os.path.join(CONTEXT_PATH, "EEG_task_taxonomy.csv"))
    return df.to_dict(orient="records")


def modalities() -> list[dict]:
    df = pd.read_csv(os.path.join(CONTEXT_PATH, "modalities.csv"))
    return df.to_dict(orient="records")


def eeg_bids_specification() -> str:
    with open(os.path.join(CONTEXT_PATH, "eeg_bids_specs.md"), "r") as file:
        return file.read()


def get_default_dataset_info(dataset_id: str) -> list[dict]:
    """
    Get default dataset information for a given dataset ID.
    This operation may take some time for large datasets.
    """
    logger.info(f"Processing request for dataset_id: {dataset_id}")

    # Clean and validate the dataset ID
    dataset_id = validate_dataset_id(dataset_id)
    logger.info(f"Cleaned dataset_id: {dataset_id}")

    # Load the data (this will use the cached version after first load)
    df = _load_default_dataset_info()

    # Filter the data
    logger.info(f"Filtering data for dataset_id: {dataset_id}")
    filtered_df = df[df["brainset"] == dataset_id]

    result = filtered_df.to_dict(orient="records")
    logger.info(f"Found {len(result)} records for dataset_id: {dataset_id}")

    return result
